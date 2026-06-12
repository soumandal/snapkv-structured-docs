"""Pre-flight checks before the overnight pilot run.

Runs on the A100 VM after Task 10 (blob mount) and Task 11 (model download).
Exits non-zero on any failure so an outer wrapper can gate the orchestrator
launch.

Cheap checks (blob mount, CUDA, model metadata, corpus dry-run) run by
default — finishes in seconds. Pass --with-forward-pass to additionally
load Llama-3.1-8B onto GPU, run one prefill, and round-trip a dump.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from kvr.config import MODEL_LLAMA, Config


def _ok(label: str) -> None:
    print(f"  [PASS] {label}")


def _fail(label: str, detail: str = "") -> None:
    print(f"  [FAIL] {label}")
    if detail:
        print(f"         {detail}")


def _warn(detail: str) -> None:
    print(f"         WARN: {detail}")


def check_blob_mount(cfg: Config) -> bool:
    p = cfg.blob_mount
    if not p.exists():
        _fail(f"blob mount missing: {p}", "see docs/runbook_blob_mount.md")
        return False
    if not p.is_dir():
        _fail(f"blob mount not a directory: {p}")
        return False
    if not p.is_mount():
        _fail(
            f"path exists but is not a mount point: {p}",
            "setup_blob_mount.sh likely failed mid-way and left a local dir. "
            "Pilot would silently write to local disk and lose data on dealloc. "
            "Re-run setup_blob_mount.sh after fixing RBAC, or `sudo rmdir` to fail-fast.",
        )
        return False
    probe = p / ".kvr_preflight"
    try:
        probe.write_text("ok")
        probe.unlink()
    except OSError as e:
        _fail(f"blob mount not writable: {p}", str(e))
        return False
    # shutil.disk_usage on a blobfuse2 mount reports the local block-cache
    # size (~8 GB), not the underlying blob capacity (effectively unbounded).
    # Don't bother warning on it.
    _ok(f"blob mount {p} writable (FUSE/blobfuse2)")
    return True


def check_cuda() -> bool:
    if not torch.cuda.is_available():
        _fail("CUDA unavailable", "pilot orchestrator requires a GPU")
        return False
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    _ok(f"CUDA: {name} ({total_gb:.0f} GB)")
    if total_gb < 40:
        _warn("<40 GB GPU; the 32k-context bucket may OOM")
    return True


def check_hf_model() -> bool:
    from transformers import AutoConfig, AutoTokenizer

    try:
        AutoConfig.from_pretrained(MODEL_LLAMA)
        AutoTokenizer.from_pretrained(MODEL_LLAMA, use_fast=True)
    except Exception as e:
        first_line = str(e).splitlines()[0] if str(e) else type(e).__name__
        _fail(f"cannot load {MODEL_LLAMA}", first_line)
        print("         Hint: `huggingface-cli login` and run scripts/00_download_models.py")
        return False
    _ok(f"model accessible: {MODEL_LLAMA}")
    return True


def check_corpus_dry_run(skip_tokenizer: bool = False) -> bool:
    from kvr.data.pilot_corpus import build_pilot_corpus
    from kvr.structure.align import align_char_spans_to_tokens
    from kvr.structure.dispatcher import label_structure

    try:
        corpus = build_pilot_corpus(n_wikitable=2, n_synthetic=2, seed=0)
    except Exception as e:
        _fail("corpus assembly failed", str(e).splitlines()[0])
        return False
    if not corpus:
        _fail("corpus assembly returned empty list")
        return False

    first = corpus[0]
    text = first.context + "\n\nQuestion: " + first.question
    spans = label_structure(text)
    if not spans:
        _fail("structure labeling produced no spans")
        return False
    if skip_tokenizer:
        _ok(f"corpus dry-run: {len(corpus)} prompts; first → {len(spans)} char spans "
            "(tokenizer skipped — hf_model check failed)")
        return True

    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(MODEL_LLAMA, use_fast=True)
    except Exception as e:
        _fail("cannot load tokenizer for token alignment", str(e).splitlines()[0])
        return False
    roles = align_char_spans_to_tokens(text, spans, tok)
    if not roles:
        _fail("token alignment produced no token roles")
        return False
    _ok(f"corpus dry-run: {len(corpus)} prompts; first → {len(roles)} token roles")
    return True


def check_forward_pass() -> bool:
    """Load Llama, run one forward pass with attention recording, save+load a dump."""
    from tempfile import TemporaryDirectory

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kvr.instrumentation.attention_hook import AttentionRecorder
    from kvr.instrumentation.dump import load_prompt_dump, save_prompt_dump
    from kvr.structure.align import align_char_spans_to_tokens
    from kvr.structure.dispatcher import label_structure

    print("  [..] loading Llama-3.1-8B (this is the slow check; ~30s)")
    tok = AutoTokenizer.from_pretrained(MODEL_LLAMA, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_LLAMA,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()

    text = '{"name": "Alice", "city": "Paris"}\n\nQuestion: what city?'
    spans = label_structure(text)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    token_roles = align_char_spans_to_tokens(text, spans, tok)

    recorder = AttentionRecorder(model)
    recorder.start()
    with torch.inference_mode():
        model(**enc, output_attentions=True, return_dict=True)
    mass = recorder.stop()

    with TemporaryDirectory() as tmp:
        dump_path = Path(tmp) / "preflight.parquet"
        save_prompt_dump(
            dump_path,
            prompt_id="preflight",
            source="preflight",
            mass=mass,
            token_roles=token_roles,
        )
        round_tripped = load_prompt_dump(dump_path)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    expected = n_layers * n_heads * len(token_roles)
    if len(round_tripped) != expected:
        _fail(
            f"forward-pass dump row count mismatch",
            f"got {len(round_tripped)}, expected {expected}",
        )
        return False
    _ok(f"forward-pass + dump round-trip: {len(round_tripped):,} rows")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--with-forward-pass",
        action="store_true",
        help="Also load Llama and run one prefill (~30s, ~20GB GPU)",
    )
    args = ap.parse_args()

    cfg = Config.from_env()
    print(f"Pre-flight checks (blob_mount={cfg.blob_mount})\n")

    checks: list[tuple[str, bool]] = []
    checks.append(("blob_mount", check_blob_mount(cfg)))
    checks.append(("cuda", check_cuda()))
    hf_ok = check_hf_model()
    checks.append(("hf_model", hf_ok))
    checks.append(("corpus_dry_run", check_corpus_dry_run(skip_tokenizer=not hf_ok)))
    if args.with_forward_pass:
        if not hf_ok:
            _fail("forward_pass skipped — hf_model failed")
            checks.append(("forward_pass", False))
        else:
            checks.append(("forward_pass", check_forward_pass()))

    failed = [name for name, ok in checks if not ok]
    print()
    if failed:
        print(f"FAILED: {len(failed)} of {len(checks)} checks did not pass: {', '.join(failed)}")
        return 1
    print(f"OK: all {len(checks)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

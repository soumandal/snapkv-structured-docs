"""Role probe (TODO #26): predict token role from token-ID window features.

Trains a multinomial logistic-regression probe (implemented as a sum of
per-offset embedding lookups, mathematically equivalent to LR on
position-aware one-hot features) that classifies tokens by role
({KEY, VALUE, DELIM, HEADER, PROSE, WS}) from a 5-token window of
token-IDs (current ± 2).

Deployment target: replace the runtime `label_structure` pipeline with
this probe so the role-conditional eviction policy ships without a
structural parser at inference time.

Success criterion (whitepaper §11): KEY F1 ≥ 0.90 with cross-corpus
generalization (train on indented JSON, test on compact JSON / XML
where the parser+labeler coupling is fragile).

Usage:
  # Smoke (5 prompts × 4 sources, ~1 min):
  .venv/bin/python scripts/17_role_probe.py --n-per-source 5
  # Full (50 × 4 sources, ~5 min):
  .venv/bin/python scripts/17_role_probe.py --n-per-source 50
"""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from kvr.config import MODEL_LLAMA
from kvr.data.pilot_corpus import build_corpus_for_source
from kvr.structure.align import align_char_spans_to_tokens
from kvr.structure.dispatcher import label_structure

WINDOW = 2  # current ± WINDOW = (2W + 1)-token feature window
SOURCES = ["synthetic_json", "synthetic_json_compact", "synthetic_xml", "wikitable_long"]
ROLES = ["KEY", "VALUE", "DELIM", "HEADER", "PROSE", "WS"]
ROLE_TO_IDX = {r: i for i, r in enumerate(ROLES)}
PAD_ID_SENTINEL = -1  # placeholder for out-of-prompt offsets


# -------- data --------
def label_prompt(text: str, tokenizer):
    spans = label_structure(text)
    token_roles = align_char_spans_to_tokens(text, spans, tokenizer)
    if not token_roles:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    token_ids = np.array([tr.token_id for tr in token_roles], dtype=np.int64)
    role_idx = np.array([ROLE_TO_IDX[tr.role.value] for tr in token_roles], dtype=np.int64)
    return token_ids, role_idx


def build_dataset(n_per_source: int, target_tokens: int, tokenizer, seed: int):
    per_prompt_ids, per_prompt_roles, sources_pp, prompt_ids_pp = [], [], [], []
    for src in SOURCES:
        corpus = build_corpus_for_source(
            src, n_prompts=n_per_source, seed=seed,
            target_tokens=target_tokens,
            tokenizer=tokenizer if src == "wikitable_long" else None,
        )[:n_per_source]
        n_src_tokens = 0
        for prompt in corpus:
            text = prompt.context + "\n\nQuestion: " + prompt.question
            ids, roles = label_prompt(text, tokenizer)
            if len(ids) == 0:
                continue
            per_prompt_ids.append(ids)
            per_prompt_roles.append(roles)
            sources_pp.append(src)
            prompt_ids_pp.append(prompt.id)
            n_src_tokens += len(ids)
        n_src_prompts = sum(1 for s in sources_pp if s == src)
        print(f"  {src:25s} {n_src_prompts:3d} prompts, {n_src_tokens:>8d} tokens")
    return per_prompt_ids, per_prompt_roles, sources_pp, prompt_ids_pp


def split_train_test(sources_pp, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for src in SOURCES:
        src_idx = [i for i, s in enumerate(sources_pp) if s == src]
        if not src_idx:
            continue
        n_test = max(1, int(round(test_frac * len(src_idx))))
        perm = rng.permutation(src_idx)
        test_idx.extend(perm[:n_test].tolist())
        train_idx.extend(perm[n_test:].tolist())
    return sorted(train_idx), sorted(test_idx)


def build_window_tensor(per_prompt_ids: list, window: int, pad_id: int):
    """Concatenate per-prompt token IDs into a (n_total, 2W+1) tensor where
    each row contains the IDs at offsets [-W..+W] around that position. Out-
    of-prompt offsets use pad_id."""
    n_offsets = 2 * window + 1
    chunks = []
    for ids in per_prompt_ids:
        L = len(ids)
        padded = np.full(L + 2 * window, pad_id, dtype=np.int64)
        padded[window : window + L] = ids
        # Build a 2D view: row i has padded[i .. i + n_offsets - 1]
        # i.e., positions (i - window) .. (i + window) in original prompt
        view = np.lib.stride_tricks.sliding_window_view(padded, window_shape=n_offsets)
        # view shape: (L, n_offsets)
        chunks.append(view.copy())
    return np.concatenate(chunks, axis=0)


# -------- model --------
class WindowLogReg(nn.Module):
    """Equivalent to multinomial LR on position-aware one-hot features. For
    each of the 2W+1 offsets we maintain a (vocab+1, n_classes) embedding
    whose lookups are summed. Index `vocab` is reserved for pad."""

    def __init__(self, vocab_size: int, n_classes: int, window: int):
        super().__init__()
        n_offsets = 2 * window + 1
        self.window = window
        self.pad_id = vocab_size  # one past the last real token id
        self.weight = nn.ParameterList([
            nn.Parameter(torch.zeros(vocab_size + 1, n_classes))
            for _ in range(n_offsets)
        ])
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def forward(self, window_ids: torch.Tensor) -> torch.Tensor:
        # window_ids: (B, 2W+1) — pad sentinel already remapped to self.pad_id
        out = self.bias.unsqueeze(0).expand(window_ids.shape[0], -1).clone()
        for k, w in enumerate(self.weight):
            out = out + F.embedding(window_ids[:, k], w)
        return out


# -------- eval --------
def per_role_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    """Return per-role precision, recall, F1, support; plus macro F1, accuracy."""
    out = {}
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        support = int((y_true == c).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec) if (prec + rec) > 0 else 0.0
        out[ROLES[c]] = {"precision": round(prec, 4), "recall": round(rec, 4),
                         "f1": round(f1, 4), "support": support}
        if support > 0:
            f1s.append(f1)
    out["__macro_f1__"] = round(float(np.mean(f1s)), 4) if f1s else 0.0
    out["__accuracy__"] = round(float((y_true == y_pred).mean()), 4)
    return out


def print_report(metrics: dict, title: str):
    print(f"\n=== {title} ===")
    print(f"  {'role':8s} {'prec':>7s} {'recall':>7s} {'f1':>7s} {'support':>9s}")
    for role in ROLES:
        m = metrics[role]
        print(f"  {role:8s} {m['precision']:7.4f} {m['recall']:7.4f} {m['f1']:7.4f} {m['support']:9d}")
    print(f"  macro F1: {metrics['__macro_f1__']:.4f}    accuracy: {metrics['__accuracy__']:.4f}")


# -------- main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-source", type=int, default=5)
    ap.add_argument("--target-tokens", type=int, default=8000)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-state", type=Path,
                    default=Path("results/plan2/role_probe_state.json"))
    ap.add_argument("--out-csv", type=Path,
                    default=Path("results/plan2/role_probe_per_corpus.csv"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    print(f"Loading tokenizer {MODEL_LLAMA}...")
    tok = AutoTokenizer.from_pretrained(MODEL_LLAMA, use_fast=True)
    vocab_size = tok.vocab_size
    print(f"  vocab_size = {vocab_size}")

    print(f"\n=== building dataset ({args.n_per_source} per source) ===")
    t0 = time.perf_counter()
    per_prompt_ids, per_prompt_roles, sources_pp, prompt_ids_pp = build_dataset(
        args.n_per_source, args.target_tokens, tok, args.seed,
    )
    n_total = sum(len(x) for x in per_prompt_ids)
    print(f"  total prompts: {len(per_prompt_ids)}; total tokens: {n_total:,}")
    print(f"  build time: {time.perf_counter() - t0:.1f}s")

    role_dist = Counter()
    for r in per_prompt_roles:
        role_dist.update(r.tolist())
    print("  global role distribution:")
    for i, role in enumerate(ROLES):
        n = role_dist.get(i, 0)
        print(f"    {role:7s} {n:>9d}  ({n / n_total:6.1%})")

    train_idx, test_idx = split_train_test(sources_pp, args.test_frac, args.seed)
    print(f"\n=== split: {len(train_idx)} train prompts, {len(test_idx)} test prompts ===")

    print("\n=== building window tensors ===")
    t0 = time.perf_counter()
    pad_id = vocab_size
    Xtr = build_window_tensor([per_prompt_ids[i] for i in train_idx], WINDOW, pad_id)
    ytr = np.concatenate([per_prompt_roles[i] for i in train_idx])
    Xte = build_window_tensor([per_prompt_ids[i] for i in test_idx], WINDOW, pad_id)
    yte = np.concatenate([per_prompt_roles[i] for i in test_idx])
    src_te = np.concatenate([
        np.full(len(per_prompt_ids[i]), sources_pp[i]) for i in test_idx
    ])
    print(f"  Xtr: {Xtr.shape}  ytr: {ytr.shape}")
    print(f"  Xte: {Xte.shape}  yte: {yte.shape}")
    print(f"  build time: {time.perf_counter() - t0:.1f}s")

    device = torch.device(args.device)
    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    Xte_t = torch.from_numpy(Xte).to(device)
    yte_t = torch.from_numpy(yte).to(device)

    model = WindowLogReg(vocab_size, len(ROLES), WINDOW).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_train = Xtr_t.shape[0]

    print(f"\n=== training ({args.epochs} epochs, batch={args.batch_size}, lr={args.lr}, device={device}) ===")
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        perm = torch.randperm(n_train, device=device)
        total_loss, total_correct = 0.0, 0
        for s in range(0, n_train, args.batch_size):
            bi = perm[s : s + args.batch_size]
            logits = model(Xtr_t[bi])
            loss = F.cross_entropy(logits, ytr_t[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(bi)
            total_correct += int((logits.argmax(-1) == ytr_t[bi]).sum())
        avg_loss = total_loss / n_train
        train_acc = total_correct / n_train
        print(f"  epoch {ep+1:2d}/{args.epochs}  loss={avg_loss:.4f}  train_acc={train_acc:.4f}")
    print(f"  train time: {time.perf_counter() - t0:.1f}s")

    # Eval
    with torch.no_grad():
        logits_te = []
        for s in range(0, Xte_t.shape[0], args.batch_size):
            logits_te.append(model(Xte_t[s : s + args.batch_size]))
        yp_te = torch.cat(logits_te).argmax(-1).cpu().numpy()

    global_metrics = per_role_metrics(yte, yp_te, len(ROLES))
    print_report(global_metrics, "global test eval")

    per_corpus = []
    for src in SOURCES:
        mask = src_te == src
        if mask.sum() == 0:
            continue
        m = per_role_metrics(yte[mask], yp_te[mask], len(ROLES))
        per_corpus.append({"source": src, "n_tokens": int(mask.sum()), **{
            f"{r}_f1": m[r]["f1"] for r in ROLES
        }, "macro_f1": m["__macro_f1__"], "acc": m["__accuracy__"]})
        print_report(m, f"per-corpus eval: {src}  (n={mask.sum()})")

    # Persist
    args.out_state.parent.mkdir(parents=True, exist_ok=True)
    args.out_state.write_text(json.dumps({
        "n_per_source": args.n_per_source,
        "vocab_size": vocab_size, "window": WINDOW,
        "epochs": args.epochs, "lr": args.lr,
        "n_total_tokens": int(n_total),
        "train_prompts": len(train_idx), "test_prompts": len(test_idx),
        "role_distribution": {ROLES[i]: int(role_dist.get(i, 0)) for i in range(len(ROLES))},
        "global_metrics": global_metrics,
        "per_corpus": per_corpus,
    }, indent=2))
    print(f"\nwrote {args.out_state}")
    with open(args.out_csv, "w", newline="") as fp:
        fieldnames = ["source", "n_tokens"] + [f"{r}_f1" for r in ROLES] + ["macro_f1", "acc"]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_corpus)
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()

"""Synthetic XML QA generator (mirrors `synthetic_json` with XML syntax).

Takes the procedurally-generated dict trees from `synthetic_json.py` and asks
gpt-4o-mini (via `external_llms/gpt_4o_mini.py`) to re-serialize each one as
XML, preserving structure and content. The question + answer are unchanged
from the JSON original so the corpora are content-matched and only the
serialization syntax varies.

Design notes:
  * Generation is gated by a verbatim-answer check — if the gold leaf value
    doesn't appear in the model's XML output, we retry once with a stricter
    system message before giving up.
  * Each (seed, target_tokens) result is cached under
    `data/cache/synthetic_xml/`. A re-run with the same seed reads from
    cache, so the corpus is deterministic across runs once generated.
  * The path notation in the question (`mike_8.charlie_42.papa_11`) is
    inherited from the JSON corpus. We do not rewrite it to XPath because
    the answer-grader is substring-EM on the leaf value; the model can
    navigate either notation.

To regenerate from scratch: delete the cache directory.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Make the sibling external_llms package importable when this module is run
# from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_EXTERNAL_DIR = _REPO_ROOT / "external_llms"
if str(_EXTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_DIR))

import urllib3  # noqa: E402
urllib3.disable_warnings()

from gpt_4o_mini import get_llm_completion  # noqa: E402

from kvr.data.synthetic_json import generate_deep_json_qa


CACHE_DIR = _REPO_ROOT / "data" / "cache" / "synthetic_xml"


_SYSTEM_PROMPT = (
    "You convert JSON to XML. Rules:\n"
    "1. Each JSON key becomes an XML element tag with the same name.\n"
    "2. Each JSON string value becomes the text content of that element.\n"
    "3. Nested JSON objects become nested XML elements.\n"
    "4. Preserve every key and value EXACTLY, character for character.\n"
    "5. Wrap the whole document in a single <root> element.\n"
    "6. Do not add comments, attributes, namespaces, or XML declarations.\n"
    "7. Indent each nested level with two spaces and put each tag on its own line.\n"
    'Return a JSON object {"xml": "<root>...</root>"} and nothing else.'
)


def _call_model(json_text: str, gold: str, retry: bool) -> str:
    # Build the retry suffix via plain concatenation to avoid str.format()
    # interpreting the literal {"xml": ...} braces in _SYSTEM_PROMPT as
    # format placeholders.
    if retry:
        system = (
            _SYSTEM_PROMPT
            + f"\nThe leaf value `{gold}` MUST appear verbatim in the XML output. "
            "Earlier attempts dropped or altered it."
        )
    else:
        system = _SYSTEM_PROMPT

    # Retry-with-backoff on transient HTTP failures (503, rate limits).
    out = ""
    for attempt in range(3):
        out = get_llm_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json_text},
            ],
            max_tokens=16000,
            response_format_json=True,
        )
        if out:
            break
        time.sleep(2.0 * (attempt + 1))
    if not out:
        return ""
    try:
        parsed = json.loads(out)
    except json.JSONDecodeError:
        # The model occasionally returns the XML directly without JSON wrapping
        # (rare, since we forced response_format=json_object).
        return out.strip()
    return parsed.get("xml", "").strip()


def generate_deep_xml_qa(
    seed: int,
    target_tokens: int,
    depth: int = 3,
    breadth: int = 4,
    *,
    cache_dir: Path | None = None,
    force_regenerate: bool = False,
) -> dict[str, str] | None:
    """Generate one XML-corpus prompt by converting the JSON sibling.

    `target_tokens` here refers to the desired XML token count (so the same
    flag value can be passed by callers that target a context bucket).
    Empirically the indented-XML re-serialization expands JSON token counts
    by ~1.2×, so the underlying JSON is generated at `target_tokens / 1.2`.

    Returns the (context, question, answer) triple on success, None on
    repeated conversion failure (the answer leaf was lost in both attempts).
    """
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"seed_{seed}_t{target_tokens}.json"

    if cache_path.exists() and not force_regenerate:
        return json.loads(cache_path.read_text())

    # 1) Generate the JSON content deterministically (same seed as
    # synthetic_json, so the dict tree is identical). Scale down by the
    # empirical XML-expansion factor so the resulting XML lands near
    # `target_tokens` after re-serialization.
    json_target = max(500, int(round(target_tokens / 1.2)))
    json_qa = generate_deep_json_qa(
        seed=seed, target_tokens=json_target, depth=depth, breadth=breadth
    )

    # 2) Ask the model to re-serialize as XML. Retry once if the gold value
    # is missing from the output.
    for attempt, retry in enumerate([False, True]):
        xml = _call_model(json_qa["context"], json_qa["answer"], retry=retry)
        if json_qa["answer"] in xml:
            break
        time.sleep(0.5)  # Light throttle between attempts.
    else:
        return None

    record = {
        "context": xml,
        "question": json_qa["question"],
        "answer": json_qa["answer"],
        "seed": seed,
        "source_json_chars": len(json_qa["context"]),
        "xml_chars": len(xml),
    }
    cache_path.write_text(json.dumps(record))
    return record


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--target-tokens", type=int, default=8000)
    ap.add_argument("--seed-base", type=int, default=0)
    args = ap.parse_args()

    failures = 0
    for i in range(args.n):
        rec = generate_deep_xml_qa(
            seed=args.seed_base * 1000 + i,
            target_tokens=args.target_tokens,
        )
        if rec is None:
            print(f"prompt {i}: FAILED (answer lost in conversion)")
            failures += 1
            continue
        print(
            f"prompt {i}: ok  "
            f"json_chars={rec['source_json_chars']:>6}  "
            f"xml_chars={rec['xml_chars']:>6}  "
            f"answer={rec['answer']!r}"
        )
    print(f"\n{args.n - failures}/{args.n} succeeded.")

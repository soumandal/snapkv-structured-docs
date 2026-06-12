"""WikiTableQuestions loader → renders tables as markdown.

The pilot needs structured-context prompts. WikiTableQuestions ships
tables as TSV; we convert to markdown so the structure detector treats
the headers as HEADER tokens.
"""
from __future__ import annotations

import random

from datasets import load_dataset


def _to_markdown_table(table: dict) -> str:
    """Convert a wikitable dict {'header': [...], 'rows': [[...], ...]} to markdown."""
    header = table["header"]
    rows = table["rows"]
    lines = []
    lines.append("| " + " | ".join(str(c) for c in header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def load_wikitable_pilot(n: int = 200, seed: int = 0) -> list[dict[str, str]]:
    # lighteval mirror: parquet-backed (the original `wikitablequestions`
    # repos ship a deprecated loader script that `datasets` refuses to run).
    ds = load_dataset("lighteval/wikitablequestions", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    out: list[dict[str, str]] = []
    for ex in ds:
        md = _to_markdown_table(ex["table"])
        context = f"{md}\n\nQuestion: {ex['question']}"
        # answers can be a list — pick first.
        ans = ex["answers"][0] if ex["answers"] else ""
        out.append({
            "context": context,
            "question": ex["question"],
            "answer": ans,
            "source": "wikitable",
            "id": ex["id"],
        })
    return out


def load_wikitable_long(
    target_tokens: int,
    n_prompts: int,
    tokenizer,
    seed: int = 0,
    pool_size: int = 4000,
    band: tuple[float, float] = (0.85, 1.15),
) -> list[dict[str, str]]:
    """Build long-context wikitable QA by concatenating distractor tables.

    Each prompt = `<distractor>\\n\\n<distractor>\\n\\n...\\n\\n<anchor>\\n\\n
    Question: <q>`. The anchor provides the question and gold answer; the
    distractors are unrelated wikitables (markdown-rendered) prepended to
    grow the prompt into `target_tokens * band`. Returned prompts that
    don't reach the band (under-filled because there are no small-enough
    remaining distractors) are dropped; the caller may need to enlarge
    `pool_size` if too few anchors qualify.

    Mirrors `load_wikitable_pilot`'s return shape; `source = "wikitable_long"`
    so downstream consumers can distinguish.
    """
    ds = load_dataset("lighteval/wikitablequestions", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(pool_size, len(ds))))

    table_md = [_to_markdown_table(ex["table"]) for ex in ds]
    table_tok_lens = [
        len(tokenizer(md, add_special_tokens=False)["input_ids"])
        for md in table_md
    ]

    rng = random.Random(seed)
    anchors = [i for i, ex in enumerate(ds) if ex["answers"]]
    rng.shuffle(anchors)

    lo = int(target_tokens * band[0])
    hi = int(target_tokens * band[1])
    sep_tokens = 2  # "\n\n" between tables

    out: list[dict[str, str]] = []
    for anchor_i in anchors:
        if len(out) >= n_prompts:
            break
        anchor_ex = ds[anchor_i]
        q_suffix = f"\n\nQuestion: {anchor_ex['question']}"
        q_suffix_tokens = len(tokenizer(q_suffix, add_special_tokens=False)["input_ids"])

        running = table_tok_lens[anchor_i] + q_suffix_tokens
        if running > hi:
            continue  # anchor itself already too long

        # Iterate distractor candidates in a fresh per-anchor random order so
        # anchor identity doesn't bias which distractors get used first.
        d_order = [i for i in range(len(ds)) if i != anchor_i]
        rng.shuffle(d_order)

        prefix_md: list[str] = []
        for d_i in d_order:
            if running >= lo:
                break
            cost = table_tok_lens[d_i] + sep_tokens
            if running + cost > hi:
                continue
            prefix_md.append(table_md[d_i])
            running += cost

        if running < lo:
            continue  # couldn't fill enough; skip this anchor

        pieces = prefix_md + [table_md[anchor_i]]
        context = "\n\n".join(pieces) + q_suffix
        out.append({
            "context": context,
            "question": anchor_ex["question"],
            "answer": anchor_ex["answers"][0],
            "source": "wikitable_long",
            "id": f"wikitable_long_{anchor_ex['id']}_t{target_tokens}",
        })

    return out

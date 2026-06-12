"""Synthetic deep-JSON QA generator (pilot version).

Generates a randomly-structured nested JSON object of approximately the
requested token size, with a randomly-chosen leaf value designated as the
"answer." The question asks: "What is the value at path X?"

The richer ITSM-style generator (incident records, audit-log
chronologies) comes later. This pilot generator is intentionally generic
so the pilot's premise check isn't biased toward domain-specific
schema shapes.
"""
from __future__ import annotations

import json
import random
import string
from typing import Any


_WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
    "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
    "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
    "victor", "whiskey", "xray", "yankee", "zulu",
]


def _random_key(rng: random.Random) -> str:
    return f"{rng.choice(_WORDS)}_{rng.randint(0, 99)}"


def _random_string(rng: random.Random) -> str:
    n = rng.randint(4, 16)
    return "".join(rng.choices(string.ascii_lowercase + string.digits, k=n))


def _build_tree(
    rng: random.Random,
    depth: int,
    breadth: int,
    target_tokens: int,
    path: list[str],
    answer_path: list[str],
) -> tuple[Any, str | None]:
    """Returns (subtree, answer_value_if_answer_path_inside_else_None)."""
    if depth == 0 or target_tokens < 50:
        # Leaf: random string. If we are at the chosen answer path, record it.
        val = _random_string(rng)
        if path == answer_path:
            return val, val
        return val, None

    obj: dict[str, Any] = {}
    found_answer: str | None = None
    for _ in range(rng.randint(2, breadth)):
        k = _random_key(rng)
        # Avoid duplicate keys at the same level.
        while k in obj:
            k = _random_key(rng)
        sub, ans = _build_tree(
            rng,
            depth - 1,
            breadth,
            target_tokens // breadth,
            path + [k],
            answer_path,
        )
        obj[k] = sub
        if ans is not None:
            found_answer = ans
    return obj, found_answer


def generate_deep_json_qa(
    seed: int,
    target_tokens: int,
    depth: int = 3,
    breadth: int = 4,
    compact: bool = False,
) -> dict[str, str]:
    """Generate a (context, question, answer) sample.

    Args:
        seed: RNG seed.
        target_tokens: rough target prompt size, in Llama-3.1 tokens.
        depth: nesting depth.
        breadth: max keys per object.
        compact: if True, render with `separators=(",", ":")` (no whitespace,
            no indentation). Used by the `synthetic_json_compact` corpus to
            isolate the role-density effect from the indentation effect.

    Returns:
        dict with keys "context" (JSON string), "question", "answer".
    """
    rng = random.Random(seed)
    # JSON content tokenizes at ~2 chars/token on Llama-3.1 (lots of short
    # tokens for braces, quotes, key separators); the older 4 chars/token
    # heuristic produced contexts ~2x the requested size.
    target_chars = target_tokens * 2

    def _render(o: Any) -> str:
        if compact:
            return json.dumps(o, separators=(",", ":"))
        return json.dumps(o, indent=2)

    # Grow a root dict by appending sibling subtrees until the rendered size
    # approximates target_chars. Each subtree has nesting depth (depth - 1),
    # so total nesting from the root matches the requested `depth`.
    root: dict[str, Any] = {}
    while True:
        k = _random_key(rng)
        while k in root:
            k = _random_key(rng)
        sub, _ = _build_tree(rng, depth - 1, breadth, target_chars, [k], answer_path=[])
        root[k] = sub
        if len(_render(root)) >= target_chars:
            break

    # Enumerate (path, leaf_value) pairs and pick one as the answer.
    leaves: list[tuple[list[str], str]] = []

    def walk(node: Any, p: list[str]) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, p + [k])
        else:
            leaves.append((p, node))

    walk(root, [])
    answer_path, answer_val = rng.choice(leaves)

    question = "What is the value at path " + ".".join(answer_path) + "?"
    context = _render(root)

    return {
        "context": context,
        "question": question,
        "answer": answer_val,
    }

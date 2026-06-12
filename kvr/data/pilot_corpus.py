"""Pilot corpus assembly.

Sources: WikiTableQuestions + synthetic deep JSON. Spider was originally in
the spec but its HF dataset doesn't ship schemas and the loader had no way
to hit the pilot's 8k-32k context buckets — deferred to a v2 corpus that
schema-packs CREATE TABLE statements from `b-mc2/sql-create-context` or
similar.

The synthetic generator's `target_tokens` parameter should be set to match
the current context bucket so each bucket gets roughly correctly-sized
prompts.
"""
from __future__ import annotations

from dataclasses import dataclass

from kvr.data.synthetic_json import generate_deep_json_qa
from kvr.data.synthetic_xml import generate_deep_xml_qa
from kvr.data.wikitable import load_wikitable_long, load_wikitable_pilot


@dataclass(frozen=True)
class PilotPrompt:
    id: str
    source: str  # "wikitable" | "synthetic_json" | "wikitable_long"
    context: str
    question: str
    answer: str


def build_corpus_for_source(
    source: str,
    n_prompts: int,
    seed: int = 0,
    target_tokens: int = 4000,
    tokenizer=None,
) -> list[PilotPrompt]:
    """Build a single-source corpus. `target_tokens` is used by length-aware
    sources (synthetic_json's generator, wikitable_long's concatenator);
    `wikitable_long` additionally requires a tokenizer.
    """
    if source == "synthetic_json":
        out: list[PilotPrompt] = []
        for i in range(n_prompts):
            ex = generate_deep_json_qa(
                seed=seed * 1000 + i,
                target_tokens=target_tokens,
                depth=3,
            )
            out.append(PilotPrompt(
                id=f"synthetic_{i:04d}",
                source="synthetic_json",
                context=ex["context"],
                question=ex["question"],
                answer=ex["answer"],
            ))
        return out
    if source == "synthetic_json_compact":
        out: list[PilotPrompt] = []
        for i in range(n_prompts):
            ex = generate_deep_json_qa(
                seed=seed * 1000 + i,
                target_tokens=target_tokens,
                depth=3,
                compact=True,
            )
            out.append(PilotPrompt(
                id=f"synthetic_compact_{i:04d}",
                source="synthetic_json_compact",
                context=ex["context"],
                question=ex["question"],
                answer=ex["answer"],
            ))
        return out
    if source == "synthetic_xml":
        # Reads from cache; the cache must be pre-populated by running
        # `python -m kvr.data.synthetic_xml --n {n_prompts} --target-tokens {target_tokens}`.
        # Prompts whose JSON->XML conversion failed (the gold leaf was lost
        # in both attempts) are silently skipped, so the returned corpus
        # may be shorter than `n_prompts`.
        out: list[PilotPrompt] = []
        for i in range(n_prompts):
            ex = generate_deep_xml_qa(
                seed=seed * 1000 + i,
                target_tokens=target_tokens,
                depth=3,
            )
            if ex is None:
                continue
            out.append(PilotPrompt(
                id=f"synthetic_xml_{i:04d}",
                source="synthetic_xml",
                context=ex["context"],
                question=ex["question"],
                answer=ex["answer"],
            ))
        return out
    if source == "wikitable":
        return [
            PilotPrompt(
                id=f"wikitable_{i:04d}",
                source="wikitable",
                context=ex["context"],
                question=ex["question"],
                answer=ex["answer"],
            )
            for i, ex in enumerate(load_wikitable_pilot(n=n_prompts, seed=seed))
        ]
    if source == "wikitable_long":
        if tokenizer is None:
            raise ValueError("wikitable_long requires a tokenizer")
        return [
            PilotPrompt(
                id=f"wikitable_long_{i:04d}_t{target_tokens}",
                source="wikitable_long",
                context=ex["context"],
                question=ex["question"],
                answer=ex["answer"],
            )
            for i, ex in enumerate(load_wikitable_long(
                target_tokens=target_tokens,
                n_prompts=n_prompts,
                tokenizer=tokenizer,
                seed=seed,
            ))
        ]
    raise ValueError(f"unknown source: {source!r}")


def build_pilot_corpus(
    n_wikitable: int = 200,
    n_synthetic: int = 200,
    seed: int = 0,
    synthetic_target_tokens: int = 4000,
    synthetic_depth: int = 3,
) -> list[PilotPrompt]:
    return (
        build_corpus_for_source("wikitable", n_wikitable, seed=seed)
        + build_corpus_for_source("synthetic_json", n_synthetic, seed=seed,
                                  target_tokens=synthetic_target_tokens)
    )

import pytest

from transformers import AutoTokenizer

from kvr.config import MODEL_LLAMA
from kvr.data.wikitable import load_wikitable_long


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL_LLAMA, use_fast=True)


def test_wikitable_long_small_target_yields_prompts_in_band(tok):
    target = 1500
    band = (0.85, 1.15)
    prompts = load_wikitable_long(
        target_tokens=target,
        n_prompts=5,
        tokenizer=tok,
        seed=0,
        pool_size=500,
        band=band,
    )
    assert len(prompts) == 5
    lo, hi = int(target * band[0]), int(target * band[1])
    for p in prompts:
        n = len(tok(p["context"], add_special_tokens=False)["input_ids"])
        assert lo <= n <= hi, f"prompt {p['id']}: {n} tokens not in [{lo}, {hi}]"
        assert p["source"] == "wikitable_long"
        assert p["answer"]


def test_wikitable_long_deterministic_under_seed(tok):
    a = load_wikitable_long(target_tokens=1500, n_prompts=3, tokenizer=tok,
                            seed=42, pool_size=500)
    b = load_wikitable_long(target_tokens=1500, n_prompts=3, tokenizer=tok,
                            seed=42, pool_size=500)
    assert [p["id"] for p in a] == [p["id"] for p in b]
    assert [p["context"] for p in a] == [p["context"] for p in b]

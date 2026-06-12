import json

from kvr.data.synthetic_json import generate_deep_json_qa


def test_returns_context_question_answer_triple():
    sample = generate_deep_json_qa(seed=0, target_tokens=512, depth=2)
    assert "context" in sample
    assert "question" in sample
    assert "answer" in sample


def test_context_is_valid_json():
    sample = generate_deep_json_qa(seed=0, target_tokens=512, depth=2)
    parsed = json.loads(sample["context"])
    assert isinstance(parsed, dict)


def test_answer_appears_verbatim_in_context():
    sample = generate_deep_json_qa(seed=42, target_tokens=1024, depth=3)
    assert sample["answer"] in sample["context"]


def test_seed_is_deterministic():
    s1 = generate_deep_json_qa(seed=7, target_tokens=512, depth=2)
    s2 = generate_deep_json_qa(seed=7, target_tokens=512, depth=2)
    assert s1 == s2


def test_target_token_size_is_approximated():
    target = 2048
    sample = generate_deep_json_qa(seed=1, target_tokens=target, depth=2)
    # The generator targets target_tokens * 2 chars (~2 chars/token for
    # JSON content on Llama-3.1). The while loop overshoots by at most
    # one final subtree, so expected: [target*2, ~target*2.5].
    actual_chars = len(sample["context"])
    assert target * 2 <= actual_chars <= target * 3

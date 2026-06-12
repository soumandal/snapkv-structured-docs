import pytest
from transformers import AutoTokenizer

from kvr.structure.align import align_char_spans_to_tokens
from kvr.structure.roles import CharRoleSpan, Role


@pytest.fixture(scope="module")
def tokenizer():
    # GPT-2 tokenizer is small and offset-mapping capable — fine for unit tests.
    return AutoTokenizer.from_pretrained("gpt2", use_fast=True)


def test_token_inherits_role_from_covering_span(tokenizer):
    s = '{"foo": 1}'
    spans = [
        CharRoleSpan(0, 1, Role.DELIM),       # {
        CharRoleSpan(1, 6, Role.KEY),         # "foo"
        CharRoleSpan(6, 7, Role.DELIM),       # :
        CharRoleSpan(7, 8, Role.WS),          # space
        CharRoleSpan(8, 9, Role.VALUE),       # 1
        CharRoleSpan(9, 10, Role.DELIM),      # }
    ]
    token_roles = align_char_spans_to_tokens(s, spans, tokenizer)
    # Each token has at least one role assigned.
    assert len(token_roles) > 0
    # The token containing "foo" must be KEY.
    enc = tokenizer(s, return_offsets_mapping=True, add_special_tokens=False)
    for tok_idx, (a, b) in enumerate(enc["offset_mapping"]):
        if "foo" in s[a:b]:
            assert token_roles[tok_idx].role == Role.KEY


def test_boundary_token_uses_role_precedence(tokenizer):
    # Construct a string where a single token spans KEY and VALUE chars.
    # Use a short ascii string; gpt-2 will often emit a single token per word.
    s = "ab"
    spans = [
        CharRoleSpan(0, 1, Role.KEY),
        CharRoleSpan(1, 2, Role.VALUE),
    ]
    token_roles = align_char_spans_to_tokens(s, spans, tokenizer)
    # "ab" is a single token; KEY has higher precedence than VALUE.
    assert token_roles[0].role == Role.KEY


def test_depth_carries_through(tokenizer):
    s = '{"a": {"b": 1}}'
    spans = [
        CharRoleSpan(1, 4, Role.KEY, depth=0),  # "a"
        CharRoleSpan(7, 10, Role.KEY, depth=1),  # "b"
    ]
    token_roles = align_char_spans_to_tokens(s, spans, tokenizer)
    enc = tokenizer(s, return_offsets_mapping=True, add_special_tokens=False)
    for tok_idx, (a, b) in enumerate(enc["offset_mapping"]):
        char_slice = s[a:b]
        if "a" in char_slice and "b" not in char_slice:
            assert token_roles[tok_idx].depth == 0
        if "b" in char_slice and "a" not in char_slice:
            assert token_roles[tok_idx].depth == 1

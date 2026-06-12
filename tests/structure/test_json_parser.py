import pytest
from kvr.structure.json_parser import label_json
from kvr.structure.roles import Role


def test_flat_object_labels_keys_and_values():
    s = '{"a": "b"}'
    spans = label_json(s)
    # Find span covering "a"
    a_span = next(sp for sp in spans if s[sp.start:sp.end] == '"a"')
    b_span = next(sp for sp in spans if s[sp.start:sp.end] == '"b"')
    assert a_span.role == Role.KEY
    assert b_span.role == Role.VALUE


def test_nested_object_keys_get_depth():
    s = '{"a": {"b": 1}}'
    spans = label_json(s)
    a_span = next(sp for sp in spans if s[sp.start:sp.end] == '"a"')
    b_span = next(sp for sp in spans if s[sp.start:sp.end] == '"b"')
    assert a_span.role == Role.KEY
    assert a_span.depth == 0
    assert b_span.role == Role.KEY
    assert b_span.depth == 1


def test_array_values_labeled_as_value():
    s = '{"k": [1, 2, 3]}'
    spans = label_json(s)
    # The array elements should be VALUE
    one_span = next(sp for sp in spans if s[sp.start:sp.end] == "1")
    assert one_span.role == Role.VALUE


def test_braces_and_commas_are_delim():
    s = '{"k": "v"}'
    spans = label_json(s)
    open_brace = next(sp for sp in spans if s[sp.start:sp.end] == "{")
    close_brace = next(sp for sp in spans if s[sp.start:sp.end] == "}")
    assert open_brace.role == Role.DELIM
    assert close_brace.role == Role.DELIM


def test_truncated_json_does_not_raise():
    # Truncation is a real failure mode in long contexts (the spec calls this out).
    s = '{"a": {"b": "trunca'
    # Should produce SOME spans, fall back to PROSE for the unparseable tail.
    spans = label_json(s)
    assert any(sp.role == Role.PROSE for sp in spans)


def test_spans_are_non_overlapping_and_cover_full_string():
    s = '{"foo": "bar", "baz": 42}'
    spans = label_json(s)
    spans_sorted = sorted(spans, key=lambda sp: sp.start)
    # Non-overlapping
    for a, b in zip(spans_sorted, spans_sorted[1:]):
        assert a.end <= b.start
    # Full coverage
    assert spans_sorted[0].start == 0
    assert spans_sorted[-1].end == len(s)

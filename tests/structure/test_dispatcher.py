from kvr.structure.dispatcher import label_structure
from kvr.structure.roles import Role


def test_pure_json_dispatched_to_json_parser():
    s = '{"a": 1}'
    spans = label_structure(s)
    a_span = next(sp for sp in spans if s[sp.start:sp.end] == '"a"')
    assert a_span.role == Role.KEY


def test_pure_markdown_table_dispatched_to_markdown_parser():
    s = "| h |\n|---|\n| v |\n"
    spans = label_structure(s)
    h_span = next(sp for sp in spans if s[sp.start:sp.end].strip() == "h")
    assert h_span.role == Role.HEADER


def test_pure_xml_dispatched_to_xml_parser():
    s = "<x>y</x>"
    spans = label_structure(s)
    open_tag = next(sp for sp in spans if s[sp.start:sp.end] == "<x>")
    assert open_tag.role == Role.KEY


def test_fenced_json_inside_prose_is_labeled():
    s = 'Some prose.\n```json\n{"a": 1}\n```\nMore prose.\n'
    spans = label_structure(s)
    # The "a" inside the fenced block should be KEY.
    a_span = next(sp for sp in spans if s[sp.start:sp.end] == '"a"')
    assert a_span.role == Role.KEY


def test_json_prefix_followed_by_prose_is_labeled():
    # Mirrors the pilot prompt shape: `{json}\n\nQuestion: ...?`.
    s = '{"a": 1, "b": "x"}\n\nQuestion: What is the value at a?'
    spans = label_structure(s)
    a_span = next(sp for sp in spans if s[sp.start:sp.end] == '"a"')
    assert a_span.role == Role.KEY
    b_span = next(sp for sp in spans if s[sp.start:sp.end] == '"b"')
    assert b_span.role == Role.KEY
    # The question text should be PROSE.
    q_spans = [sp for sp in spans if "Question" in s[sp.start:sp.end]]
    assert q_spans and all(sp.role == Role.PROSE for sp in q_spans)


def test_unrecognized_input_labeled_prose():
    s = "Just some normal prose without any structure at all."
    spans = label_structure(s)
    assert any(sp.role == Role.PROSE for sp in spans)


def test_spans_non_overlapping_and_full_coverage():
    s = 'Pre.\n```json\n{"a": 1}\n```\nPost.\n'
    spans = label_structure(s)
    spans_sorted = sorted(spans, key=lambda sp: sp.start)
    for a, b in zip(spans_sorted, spans_sorted[1:]):
        assert a.end <= b.start, f"overlap at {a} → {b}"
    assert spans_sorted[0].start == 0
    assert spans_sorted[-1].end == len(s)

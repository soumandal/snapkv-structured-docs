from kvr.structure.markdown_parser import label_markdown_table
from kvr.structure.roles import Role


def test_simple_table_labels_header_row_as_header():
    s = "| name | age |\n|---|---|\n| Alice | 30 |\n"
    spans = label_markdown_table(s)
    name_span = next(sp for sp in spans if s[sp.start:sp.end].strip() == "name")
    age_span = next(sp for sp in spans if s[sp.start:sp.end].strip() == "age")
    assert name_span.role == Role.HEADER
    assert age_span.role == Role.HEADER


def test_separator_row_is_delim():
    s = "| a | b |\n|---|---|\n| 1 | 2 |\n"
    spans = label_markdown_table(s)
    sep_spans = [sp for sp in spans if "---" in s[sp.start:sp.end]]
    assert all(sp.role == Role.DELIM for sp in sep_spans)


def test_body_cells_are_value():
    s = "| a | b |\n|---|---|\n| 1 | 2 |\n"
    spans = label_markdown_table(s)
    one_span = next(sp for sp in spans if s[sp.start:sp.end].strip() == "1")
    two_span = next(sp for sp in spans if s[sp.start:sp.end].strip() == "2")
    assert one_span.role == Role.VALUE
    assert two_span.role == Role.VALUE


def test_pipes_are_delim():
    s = "| a |\n|---|\n| 1 |\n"
    spans = label_markdown_table(s)
    pipe_spans = [sp for sp in spans if s[sp.start:sp.end] == "|"]
    assert len(pipe_spans) >= 3
    assert all(sp.role == Role.DELIM for sp in pipe_spans)


def test_non_table_text_is_prose():
    # If the input does not look like a markdown table, label everything PROSE.
    s = "This is not a table.\nJust prose.\n"
    spans = label_markdown_table(s)
    assert all(sp.role in (Role.PROSE, Role.WS) for sp in spans)

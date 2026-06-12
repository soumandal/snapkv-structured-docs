from kvr.structure.xml_parser import label_xml
from kvr.structure.roles import Role


def test_open_and_close_tags_are_key():
    s = "<foo>bar</foo>"
    spans = label_xml(s)
    open_tag = next(sp for sp in spans if s[sp.start:sp.end] == "<foo>")
    close_tag = next(sp for sp in spans if s[sp.start:sp.end] == "</foo>")
    assert open_tag.role == Role.KEY
    assert close_tag.role == Role.KEY


def test_content_between_tags_is_value():
    s = "<foo>bar</foo>"
    spans = label_xml(s)
    content = next(sp for sp in spans if s[sp.start:sp.end] == "bar")
    assert content.role == Role.VALUE


def test_nested_tags_get_depth():
    s = "<a><b>x</b></a>"
    spans = label_xml(s)
    a_open = next(sp for sp in spans if s[sp.start:sp.end] == "<a>")
    b_open = next(sp for sp in spans if s[sp.start:sp.end] == "<b>")
    assert a_open.depth == 0
    assert b_open.depth == 1


def test_malformed_xml_falls_back_to_prose():
    s = "<unclosed>oops"
    spans = label_xml(s)
    assert any(sp.role == Role.PROSE for sp in spans)

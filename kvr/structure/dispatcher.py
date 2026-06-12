"""Top-down format detection and per-block dispatch.

Strategy: split the input into blocks based on format markers, then call
the appropriate parser on each block. Blocks:
  - Fenced code blocks marked ```json → JSON parser
  - Fenced code blocks marked ```xml → XML parser
  - Markdown table regions (auto-detected by separator row) → markdown parser
  - Top-level JSON (whole input parses as JSON) → JSON parser
  - Top-level XML (whole input parses as XML-shaped) → XML parser
  - Everything else → PROSE
"""
import json
import re

from kvr.structure.json_parser import label_json
from kvr.structure.markdown_parser import (
    label_markdown_table,
    _is_separator_line,
    _is_table_row,
)
from kvr.structure.roles import CharRoleSpan, Role
from kvr.structure.xml_parser import label_xml

_FENCE_RE = re.compile(r"^```(json|xml)\s*$", re.MULTILINE)


def _is_probably_json(s: str) -> bool:
    s = s.strip()
    return (
        (s.startswith("{") and s.endswith("}"))
        or (s.startswith("[") and s.endswith("]"))
    )


def _json_prefix_end(s: str) -> int | None:
    """If `s` starts (after leading whitespace) with a syntactically valid
    JSON object/array, return the offset in `s` just past it. Otherwise None.

    Used to detect prompts of the form `{...json...}\n\nQuestion: ...?` where
    a JSON document is followed by free-form prose.
    """
    lead = len(s) - len(s.lstrip())
    body = s[lead:]
    if not body or body[0] not in "{[":
        return None
    try:
        _, end = json.JSONDecoder().raw_decode(body)
    except json.JSONDecodeError:
        return None
    return lead + end


def _is_probably_xml(s: str) -> bool:
    s = s.strip()
    return s.startswith("<") and ">" in s


def _has_markdown_table(s: str) -> bool:
    lines = s.splitlines()
    for i in range(len(lines) - 1):
        if _is_table_row(lines[i]) and _is_separator_line(lines[i + 1]):
            return True
    return False


def _shift_spans(spans: list[CharRoleSpan], offset: int) -> list[CharRoleSpan]:
    return [
        CharRoleSpan(sp.start + offset, sp.end + offset, sp.role, sp.depth)
        for sp in spans
    ]


def _label_prose(s: str, base_offset: int = 0) -> list[CharRoleSpan]:
    if not s:
        return []
    return [CharRoleSpan(base_offset, base_offset + len(s), Role.PROSE)]


def label_structure(s: str) -> list[CharRoleSpan]:
    if not s:
        return []

    # Whole-input JSON / XML
    if _is_probably_json(s.strip()):
        # Find the JSON's start offset (skip leading whitespace).
        lead = len(s) - len(s.lstrip())
        trail = len(s) - len(s.rstrip())
        inner = s[lead:len(s) - trail]
        spans = _shift_spans(label_json(inner), lead)
        if lead:
            spans = [CharRoleSpan(0, lead, Role.WS)] + spans
        if trail:
            spans.append(CharRoleSpan(len(s) - trail, len(s), Role.WS))
        return spans

    # JSON-prefix-then-prose (e.g. `{...}\n\nQuestion: ...?`).
    json_end = _json_prefix_end(s)
    if json_end is not None and json_end < len(s):
        lead = len(s) - len(s.lstrip())
        inner = s[lead:json_end]
        spans: list[CharRoleSpan] = []
        if lead:
            spans.append(CharRoleSpan(0, lead, Role.WS))
        spans.extend(_shift_spans(label_json(inner), lead))
        spans.extend(_label_prose(s[json_end:], json_end))
        return spans

    if _is_probably_xml(s.strip()) and not _has_markdown_table(s):
        lead = len(s) - len(s.lstrip())
        trail = len(s) - len(s.rstrip())
        inner = s[lead:len(s) - trail]
        spans = _shift_spans(label_xml(inner), lead)
        if lead:
            spans = [CharRoleSpan(0, lead, Role.WS)] + spans
        if trail:
            spans.append(CharRoleSpan(len(s) - trail, len(s), Role.WS))
        return spans

    # Mixed content: scan for fenced blocks + tables, treat the rest as prose.
    spans: list[CharRoleSpan] = []
    cursor = 0
    n = len(s)

    while cursor < n:
        # Look for next fence.
        fence_match = _FENCE_RE.search(s, cursor)
        # Look for next table.
        table_start = None
        lines = s[cursor:].splitlines(keepends=True)
        line_offset = cursor
        for i in range(len(lines) - 1):
            if _is_table_row(lines[i]) and _is_separator_line(lines[i + 1]):
                table_start = line_offset
                break
            line_offset += len(lines[i])

        # Decide which comes first.
        next_event: tuple[int, str] | None = None
        fence_first = fence_match and (
            table_start is None or fence_match.start() < table_start
        )
        if fence_first:
            next_event = (fence_match.start(), "fence")
        elif table_start is not None:
            next_event = (table_start, "table")

        if next_event is None:
            # Nothing else — rest is prose.
            spans.extend(_label_prose(s[cursor:], cursor))
            break

        ev_start, ev_kind = next_event
        # Prose before the event.
        if ev_start > cursor:
            spans.extend(_label_prose(s[cursor:ev_start], cursor))

        if ev_kind == "fence":
            assert fence_match is not None
            fence_kind = fence_match.group(1)
            block_start = fence_match.end()
            # Find closing fence.
            close_idx = s.find("```", block_start)
            if close_idx == -1:
                # Unterminated fence → rest is prose.
                spans.extend(_label_prose(s[ev_start:], ev_start))
                break
            inner = s[block_start:close_idx]
            # The fence lines themselves are DELIM.
            spans.append(CharRoleSpan(ev_start, block_start, Role.DELIM))
            if fence_kind == "json":
                spans.extend(_shift_spans(label_json(inner), block_start))
            elif fence_kind == "xml":
                spans.extend(_shift_spans(label_xml(inner), block_start))
            else:
                spans.extend(_label_prose(inner, block_start))
            spans.append(CharRoleSpan(close_idx, close_idx + 3, Role.DELIM))
            cursor = close_idx + 3
        else:  # table
            # Find end of contiguous table region.
            rest = s[ev_start:]
            rest_lines = rest.splitlines(keepends=True)
            end_off = ev_start
            for ln in rest_lines:
                if _is_table_row(ln) or _is_separator_line(ln):
                    end_off += len(ln)
                else:
                    break
            inner = s[ev_start:end_off]
            spans.extend(_shift_spans(label_markdown_table(inner), ev_start))
            cursor = end_off

    return spans

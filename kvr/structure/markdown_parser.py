"""Char-level role labeling for markdown tables.

A markdown table is: a row of pipe-delimited cells, then a separator row
matching `|---|---|...|`, then zero or more body rows. The dispatcher
(`kvr.structure.dispatcher`) calls this only on input that already looks
like a table. For tables embedded in prose, the dispatcher splits the
input first.
"""
import re

from kvr.structure.roles import CharRoleSpan, Role

_SEPARATOR_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")


def _is_separator_line(line: str) -> bool:
    return bool(_SEPARATOR_RE.match(line))


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return (
        stripped.startswith("|")
        and stripped.endswith("|")
        and len(stripped) >= 3
    )


def label_markdown_table(s: str) -> list[CharRoleSpan]:
    spans: list[CharRoleSpan] = []
    lines = s.splitlines(keepends=True)
    offsets: list[int] = []
    acc = 0
    for ln in lines:
        offsets.append(acc)
        acc += len(ln)

    # Find table region: first table row followed by separator.
    table_start_idx: int | None = None
    for i, ln in enumerate(lines[:-1]):
        if _is_table_row(ln) and _is_separator_line(lines[i + 1]):
            table_start_idx = i
            break

    if table_start_idx is None:
        # No table — label everything PROSE / WS.
        i = 0
        while i < len(s):
            if s[i].isspace():
                start = i
                while i < len(s) and s[i].isspace():
                    i += 1
                spans.append(CharRoleSpan(start, i, Role.WS))
            else:
                start = i
                while i < len(s) and not s[i].isspace():
                    i += 1
                spans.append(CharRoleSpan(start, i, Role.PROSE))
        return spans

    # Label pre-table region as PROSE.
    pre_end = offsets[table_start_idx]
    if pre_end > 0:
        spans.append(CharRoleSpan(0, pre_end, Role.PROSE))

    # Label table itself.
    end_idx = table_start_idx
    while end_idx < len(lines) and _is_table_row(lines[end_idx]):
        end_idx += 1
    table_end_offset = offsets[end_idx] if end_idx < len(lines) else len(s)

    for row_idx in range(table_start_idx, end_idx):
        row_line = lines[row_idx]
        row_offset = offsets[row_idx]
        is_header = (row_idx == table_start_idx)
        is_separator = (row_idx == table_start_idx + 1)

        # Walk the line; pipes are DELIM, separator content is DELIM,
        # header content is HEADER, body content is VALUE.
        i = 0
        while i < len(row_line):
            c = row_line[i]
            if c == "|":
                spans.append(CharRoleSpan(
                    row_offset + i, row_offset + i + 1, Role.DELIM
                ))
                i += 1
            elif c.isspace():
                start = i
                while (
                    i < len(row_line)
                    and row_line[i].isspace()
                    and row_line[i] != "\n"
                ):
                    i += 1
                if start < i:
                    spans.append(CharRoleSpan(
                        row_offset + start, row_offset + i, Role.WS
                    ))
                if i < len(row_line) and row_line[i] == "\n":
                    spans.append(CharRoleSpan(
                        row_offset + i, row_offset + i + 1, Role.WS
                    ))
                    i += 1
            else:
                start = i
                while (
                    i < len(row_line)
                    and row_line[i] != "|"
                    and not row_line[i].isspace()
                ):
                    i += 1
                if is_separator:
                    role = Role.DELIM
                elif is_header:
                    role = Role.HEADER
                else:
                    role = Role.VALUE
                spans.append(
                    CharRoleSpan(row_offset + start, row_offset + i, role)
                )

    # Label post-table region as PROSE.
    if table_end_offset < len(s):
        spans.append(CharRoleSpan(table_end_offset, len(s), Role.PROSE))

    return spans

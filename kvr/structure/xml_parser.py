"""Char-level role labeling for XML inputs.

Simple non-validating scanner: anything matching `<...>` is a tag (KEY),
content between tags is VALUE. On any structural mismatch we fall back to
PROSE for the remaining suffix.
"""
from kvr.structure.roles import CharRoleSpan, Role


def label_xml(s: str) -> list[CharRoleSpan]:
    spans: list[CharRoleSpan] = []
    i = 0
    n = len(s)
    depth = 0
    tag_stack: list[str] = []

    def push(start: int, end: int, role: Role, d: int = 0) -> None:
        if start < end:
            spans.append(CharRoleSpan(start, end, role, d))

    try:
        while i < n:
            if s[i] == "<":
                tag_start = i
                while i < n and s[i] != ">":
                    i += 1
                if i >= n:
                    raise ValueError("Unterminated tag")
                i += 1
                tag = s[tag_start:i]
                is_close = tag.startswith("</")
                is_self_close = tag.endswith("/>")

                if is_close:
                    push(tag_start, i, Role.KEY, max(depth - 1, 0))
                    depth = max(depth - 1, 0)
                    if tag_stack:
                        tag_stack.pop()
                else:
                    push(tag_start, i, Role.KEY, depth)
                    if not is_self_close:
                        tag_stack.append(tag)
                        depth += 1
            else:
                content_start = i
                while i < n and s[i] != "<":
                    i += 1
                push(content_start, i, Role.VALUE, depth)
    except (ValueError, IndexError):
        if i < n:
            push(i, n, Role.PROSE)

    if tag_stack:
        # Unclosed tags at EOF: signal lenient parse to the dispatcher by
        # appending a zero-width PROSE span directly (bypassing push's
        # start < end guard). The span acts as a flag; downstream code that
        # skips zero-width spans will ignore it harmlessly.
        spans.append(CharRoleSpan(n, n, Role.PROSE))

    return spans

"""Char-level role labeling for JSON inputs.

Walks the input string character by character with a small stack-based parser
that tracks (a) which "slot" we're in (key vs value of an object), (b) nesting
depth, and (c) whether we're inside a string. Outputs non-overlapping spans
that cover the entire input. Truncation tolerant: on parse failure, the
remaining suffix is labeled PROSE.
"""
from kvr.structure.roles import CharRoleSpan, Role


def _is_ws(c: str) -> bool:
    return c in " \t\n\r"


def label_json(s: str) -> list[CharRoleSpan]:
    spans: list[CharRoleSpan] = []
    n = len(s)
    i = 0
    # Parser state: stack of "context", context ∈ {"OBJECT", "ARRAY"}.
    # For OBJECT, slot ∈ {"KEY", "VALUE"}: whether next string is a key.
    stack: list[tuple[str, str]] = []  # (context, slot)
    depth = -1  # incremented on { or [

    def push_span(start: int, end: int, role: Role, d: int = 0) -> None:
        if start < end:
            spans.append(
                CharRoleSpan(start=start, end=end, role=role, depth=d)
            )

    try:
        while i < n:
            c = s[i]
            if _is_ws(c):
                ws_start = i
                while i < n and _is_ws(s[i]):
                    i += 1
                push_span(ws_start, i, Role.WS)
                continue

            if c == "{":
                push_span(i, i + 1, Role.DELIM, max(depth, 0))
                depth += 1
                stack.append(("OBJECT", "KEY"))
                i += 1
                continue

            if c == "}":
                push_span(i, i + 1, Role.DELIM, max(depth, 0))
                depth -= 1
                if stack:
                    stack.pop()
                i += 1
                continue

            if c == "[":
                push_span(i, i + 1, Role.DELIM, max(depth, 0))
                depth += 1
                stack.append(("ARRAY", "VALUE"))
                i += 1
                continue

            if c == "]":
                push_span(i, i + 1, Role.DELIM, max(depth, 0))
                depth -= 1
                if stack:
                    stack.pop()
                i += 1
                continue

            if c == ":":
                push_span(i, i + 1, Role.DELIM, max(depth, 0))
                if stack and stack[-1][0] == "OBJECT":
                    stack[-1] = ("OBJECT", "VALUE")
                i += 1
                continue

            if c == ",":
                push_span(i, i + 1, Role.DELIM, max(depth, 0))
                if stack and stack[-1][0] == "OBJECT":
                    stack[-1] = ("OBJECT", "KEY")
                i += 1
                continue

            if c == '"':
                # Scan string contents (handles backslash escapes).
                str_start = i
                i += 1
                while i < n and s[i] != '"':
                    if s[i] == "\\" and i + 1 < n:
                        i += 2
                    else:
                        i += 1
                if i >= n:
                    # Unterminated string: label remainder as PROSE.
                    push_span(str_start, n, Role.PROSE)
                    i = n
                    break
                i += 1  # consume closing quote
                str_end = i

                if stack and stack[-1] == ("OBJECT", "KEY"):
                    push_span(str_start, str_end, Role.KEY, depth)
                else:
                    push_span(str_start, str_end, Role.VALUE, depth)
                continue

            # Number / true / false / null
            tok_start = i
            while i < n and s[i] not in '{}[]:," \t\n\r':
                i += 1
            push_span(tok_start, i, Role.VALUE, depth)
    except (ValueError, IndexError):
        # Truncated / malformed → label remainder as PROSE.
        if i < n:
            push_span(i, n, Role.PROSE)

    return spans

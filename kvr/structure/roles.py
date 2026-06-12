"""Token role taxonomy for schema-guided KV-cache work."""
from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    KEY = "KEY"       # JSON keys, XML tags
    HEADER = "HEADER"  # Markdown table header cells
    VALUE = "VALUE"   # JSON values, XML content, table body cells
    DELIM = "DELIM"   # Structural delimiters that aren't keys (commas, braces, pipes)
    WS = "WS"         # Whitespace
    PROSE = "PROSE"   # Unrecognized / free-form prose


# Higher = higher precedence when a token spans multiple roles.
_PRECEDENCE = {
    Role.KEY: 5,
    Role.HEADER: 4,
    Role.VALUE: 3,
    Role.DELIM: 2,
    Role.WS: 1,
    Role.PROSE: 0,
}


def role_precedence(role: Role) -> int:
    return _PRECEDENCE[role]


def set_role_precedence(role: Role, value: int) -> None:
    """Override the precedence of `role`. Used by ablation drivers that need
    to test how labeler precedence affects downstream eviction policies (e.g.
    flipping DELIM above KEY so merged tokens containing both label as DELIM).
    """
    _PRECEDENCE[role] = value


@dataclass(frozen=True)
class CharRoleSpan:
    """A run of characters in the input string with a single role."""
    start: int       # inclusive
    end: int         # exclusive
    role: Role
    depth: int = 0   # for nested structures: 0 = top-level, increasing inward


@dataclass(frozen=True)
class TokenRole:
    """A single token with its resolved role + depth."""
    token_id: int
    role: Role
    depth: int = 0

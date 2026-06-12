from kvr.structure.roles import Role, CharRoleSpan, TokenRole, role_precedence


def test_role_enum_has_all_required_values():
    assert Role.KEY
    assert Role.HEADER
    assert Role.VALUE
    assert Role.DELIM
    assert Role.WS
    assert Role.PROSE


def test_precedence_orders_key_above_header_above_value():
    # If a token straddles spans, the higher-precedence role wins.
    assert role_precedence(Role.KEY) > role_precedence(Role.HEADER)
    assert role_precedence(Role.HEADER) > role_precedence(Role.VALUE)
    assert role_precedence(Role.VALUE) > role_precedence(Role.DELIM)
    assert role_precedence(Role.DELIM) > role_precedence(Role.WS)
    assert role_precedence(Role.WS) >= role_precedence(Role.PROSE)


def test_char_role_span_is_valid():
    span = CharRoleSpan(start=0, end=10, role=Role.KEY, depth=2)
    assert span.start == 0
    assert span.end == 10
    assert span.role == Role.KEY
    assert span.depth == 2


def test_token_role_carries_role_and_depth():
    tr = TokenRole(token_id=5, role=Role.KEY, depth=2)
    assert tr.token_id == 5
    assert tr.role == Role.KEY
    assert tr.depth == 2

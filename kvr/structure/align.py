"""Map char-level role spans onto token IDs using the tokenizer's offset_mapping.

For each token spanning char range [a, b):
  - Find all CharRoleSpans that overlap [a, b).
  - Resolve role by highest precedence among the overlapping spans.
  - Resolve depth by the max depth among the overlapping spans whose role won.
"""
from kvr.structure.roles import CharRoleSpan, Role, TokenRole, role_precedence


def align_char_spans_to_tokens(
    s: str,
    spans: list[CharRoleSpan],
    tokenizer,
) -> list[TokenRole]:
    encoding = tokenizer(s, return_offsets_mapping=True, add_special_tokens=False)
    offsets: list[tuple[int, int]] = encoding["offset_mapping"]
    token_ids: list[int] = encoding["input_ids"]

    # Sort spans by start for faster lookup.
    spans_sorted = sorted(spans, key=lambda sp: sp.start)

    result: list[TokenRole] = []
    for token_idx, (a, b) in enumerate(offsets):
        if a == b:
            # Zero-width tokens (rare, e.g. some SentencePiece tokenizers): treat as PROSE.
            result.append(TokenRole(token_id=token_ids[token_idx], role=Role.PROSE, depth=0))
            continue

        # Find overlapping spans.
        winner_role = Role.PROSE
        winner_prec = -1
        winner_depth = 0
        for sp in spans_sorted:
            if sp.end <= a:
                continue
            if sp.start >= b:
                break
            prec = role_precedence(sp.role)
            if prec > winner_prec:
                winner_prec = prec
                winner_role = sp.role
                winner_depth = sp.depth
            elif prec == winner_prec:
                winner_depth = max(winner_depth, sp.depth)
        result.append(TokenRole(token_id=token_ids[token_idx], role=winner_role, depth=winner_depth))

    return result

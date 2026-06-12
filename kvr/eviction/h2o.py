"""Reference H2O eviction simulator (pilot baseline).

We need to answer: given the attention weights actually emitted by the model
on a long structured context, which tokens would H2O keep at budget B?

The "real" H2O (in the original paper) operates online during decoding by
maintaining a per-head accumulated attention score and evicting the lowest-
scoring entries when the cache grows past budget. For the pilot we don't
need the online behavior — we just need: given a recorded attention matrix
over the prompt's prefill, which positions get the top-B accumulated mass.
This collapses to a top-k over summed columns.
"""
import torch


def simulate_h2o_eviction(
    attn_weights: torch.Tensor,
    budget: int,
) -> list[list[torch.Tensor]]:
    """Simulate H2O eviction.

    Args:
        attn_weights: shape (n_layers, n_heads, n_query, n_kv). Float tensor of
            post-softmax attention probabilities. Typically obtained by
            attaching a forward hook to each LlamaAttention layer.
        budget: number of kv positions to keep per (layer, head).

    Returns:
        keep[layer][head] = 1-D LongTensor of kept kv-position indices,
        length min(budget, n_kv), sorted ascending.
    """
    n_layers, n_heads, _, n_kv = attn_weights.shape
    keep: list[list[torch.Tensor]] = []
    # Accumulate attention mass per kv-position across query tokens.
    mass = attn_weights.sum(dim=2)  # (n_layers, n_heads, n_kv)
    eff_budget = min(budget, n_kv)
    for layer in range(n_layers):
        per_head: list[torch.Tensor] = []
        for head in range(n_heads):
            scores = mass[layer, head]  # (n_kv,)
            _, top_idx = torch.topk(scores, k=eff_budget, largest=True, sorted=False)
            per_head.append(torch.sort(top_idx)[0])
        keep.append(per_head)
    return keep

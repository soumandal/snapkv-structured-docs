import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvr.instrumentation.attention_hook import AttentionRecorder


@pytest.mark.slow
@pytest.mark.gpu
def test_recorder_captures_attention_mass_for_short_prompt():
    # TinyLlama is a cheap shape-correctness stand-in for Llama-family models.
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    ).to("cuda")
    model.eval()

    prompt = '{"a": 1}'
    enc = tok(prompt, return_tensors="pt").to("cuda")

    recorder = AttentionRecorder(model, q_chunk=4)  # tiny chunk to exercise the loop
    recorder.start()
    with torch.inference_mode():
        model(**enc, return_dict=True)
    mass = recorder.stop()

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_tokens = enc["input_ids"].shape[1]

    assert mass.shape == (n_layers, n_heads, n_tokens)
    # Causal attention: each query row sums to 1 in fp32. Summed across
    # n_tokens queries → total accumulated mass per (layer, head) = n_tokens.
    total_per_head = mass.sum(dim=-1)
    expected = torch.full_like(total_per_head, float(n_tokens))
    assert torch.allclose(total_per_head, expected, atol=1e-2)


def test_reset_is_idempotent_and_restores_forwards():
    """Pure-CPU test — reset() unpatches even before start() and after stop()."""
    class FakeAttn:
        def forward(self, *a, **kw): return None

    class FakeLayer:
        def __init__(self): self.self_attn = FakeAttn()

    class FakeBase:
        def __init__(self): self.layers = [FakeLayer(), FakeLayer()]

    class FakeModel:
        def __init__(self): self.model = FakeBase()

    m = FakeModel()
    attns = [l.self_attn for l in m.model.layers]

    rec = AttentionRecorder(m)
    rec.reset()  # before start — no-op, must not error
    assert "forward" not in vars(attns[0])

    rec.start()
    assert "forward" in vars(attns[0])
    assert "forward" in vars(attns[1])

    rec.reset()
    assert "forward" not in vars(attns[0])
    assert "forward" not in vars(attns[1])

    rec.reset()  # second call — no-op

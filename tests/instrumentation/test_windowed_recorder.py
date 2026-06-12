"""Sanity test for WindowedAttentionRecorder.

When window_size >= q_len, windowed mass must equal full-prefix mass
(the AttentionRecorder's output), up to fp tolerance. Uses TinyLlama
to keep the test runnable without the 8B model.
"""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.mark.gpu
def test_full_window_matches_unwindowed():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kvr.instrumentation.attention_hook import AttentionRecorder
    from kvr.instrumentation.windowed_recorder import WindowedAttentionRecorder

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()

    text = "The quick brown fox jumps over the lazy dog. " * 20
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    q_len = enc["input_ids"].shape[-1]

    full_rec = AttentionRecorder(model, q_chunk=64)
    full_rec.start()
    with torch.inference_mode():
        model(**enc)
    full_mass = full_rec.stop()

    win_rec = WindowedAttentionRecorder(model, window_size=q_len, q_chunk=64)
    win_rec.start()
    with torch.inference_mode():
        model(**enc)
    win_mass = win_rec.stop()

    # Full prefix == full window — agreement to fp tolerance.
    diff = (full_mass - win_mass).abs().max().item()
    assert diff < 1e-3, f"max abs diff {diff}"


@pytest.mark.gpu
def test_small_window_is_subset_of_full():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kvr.instrumentation.attention_hook import AttentionRecorder
    from kvr.instrumentation.windowed_recorder import WindowedAttentionRecorder

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    text = "The quick brown fox jumps over the lazy dog. " * 20
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    q_len = enc["input_ids"].shape[-1]

    full = AttentionRecorder(model, q_chunk=64); full.start()
    with torch.inference_mode(): model(**enc)
    full_mass = full.stop()

    win = WindowedAttentionRecorder(model, window_size=max(1, q_len // 4), q_chunk=64)
    win.start()
    with torch.inference_mode(): model(**enc)
    win_mass = win.stop()

    # Window contribution must be elementwise <= full contribution.
    assert (win_mass <= full_mass + 1e-3).all().item()

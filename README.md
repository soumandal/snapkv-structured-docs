# kvr — Schema-Guided KV-Cache Sparsification

## Setup

    uv venv && source .venv/bin/activate
    uv pip install -e ".[dev]"

Always run the pilot/eviction scripts from this project `.venv` — **not** the
machine's `azureml_*` conda envs, which lack `polars`/`kvr` and carry a broken
`botocore`/`urllib3` combo that fails the `transformers` import. To launch
without activating, put the repo root on the path:

    PYTHONPATH="$PWD" .venv/bin/python scripts/02_run_pilot.py --model phi --use-chat-template --ctx-buckets 16000

(`.venv-qwen3` is a separate venv on transformers 4.57.6, used only for the
`qwen3` model, which needs ≥4.51.)

## Testing

    pytest                          # all tests
    pytest -m "not slow and not gpu"  # local laptop subset

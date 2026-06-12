#!/usr/bin/env bash
# Re-run the combined-method headline cell (synthetic_json ctx16k, n=50, no-key)
# for Llama + Mistral on the unified SDPA-decode path, to refresh the locked
# numbers that were produced under the old manual-fp16 decode. Matches the
# originals exactly except the decode path: no chat template, default
# max-new-tokens=32, runtime role labeling (== cached for no-chat-template).
# Each model: 5-budget headline run, then a full-budget (B=1.0) oracle for C9.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export PYTHONPATH="$PWD"
PY=.venv/bin/python

run() {  # $1=model  $2=outcsv  $3...=extra args
  local model="$1"; local out="$2"; shift 2
  echo "=== $(date -u +%H:%M:%S) | $model -> $out | extra: $* ==="
  $PY scripts/13_snapkv_role_conditional.py \
    --model "$model" --source-filter synthetic_json --ctx-tokens 16000 \
    --max-prompts 50 --policy no-key --out-path "$out" "$@" \
    2>&1 | grep -vE "it/s\]|warnings.warn|UserWarning|Starting from v4.46|Loading checkpoint"
}

rm -f results/plan2/ac_combined_llama_sdpa.csv results/plan2/ac_combined_mistral_sdpa.csv

run llama   results/plan2/ac_combined_llama_sdpa.csv
run llama   results/plan2/ac_combined_llama_sdpa.csv   --budgets 1.0 --skip-snapkv-only
run mistral results/plan2/ac_combined_mistral_sdpa.csv
run mistral results/plan2/ac_combined_mistral_sdpa.csv --budgets 1.0 --skip-snapkv-only

echo "=== $(date -u +%H:%M:%S) | ALL DONE ==="

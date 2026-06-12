#!/usr/bin/env bash
# Chain: wait for the Qwen pilot recorder (scripts/02) to finish, then run the
# C2 counterfactual eviction cell (scripts/05) on Qwen2.5 synthetic_json ctx16k.
# Mirrors the scripts/13 Qwen Tier-1 invocation (chat template + 64 decode toks).
set -euo pipefail
cd "$(dirname "$0")/.."

REC_PID="${1:?usage: _qwen_cf_chain.sh <recorder_pid>}"
RUN_ROOT=/mnt/kvr/dumps/2026-WW23-pilot-qwen-ct
BUCKET="$RUN_ROOT/ctx_16000"
OUT=results/plan2/e_counterfactual_qwen.csv

echo "=== $(date +%H:%M:%S) | waiting on recorder PID $REC_PID ==="
while kill -0 "$REC_PID" 2>/dev/null; do sleep 15; done

N=$(ls "$BUCKET"/*.parquet 2>/dev/null | wc -l)
echo "=== $(date +%H:%M:%S) | recorder done; $N parquet dumps in $BUCKET ==="
if [ "$N" -lt 50 ]; then
  echo "ERROR: need >=50 dumps for the n=50 cell, found $N. Aborting scripts/05." >&2
  exit 1
fi

echo "=== $(date +%H:%M:%S) | launching scripts/05 (C2 counterfactual) -> $OUT ==="
PYTHONPATH="$PWD" .venv/bin/python scripts/05_counterfactual_eviction.py \
    --model qwen --use-chat-template \
    --source-filter synthetic_json --ctx-tokens 16000 \
    --max-prompts 50 --max-new-tokens 64 \
    --run-root "$RUN_ROOT" \
    --out-path "$OUT"

echo "=== $(date +%H:%M:%S) | ALL DONE -> $OUT ==="

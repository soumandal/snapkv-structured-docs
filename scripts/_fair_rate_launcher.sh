#!/usr/bin/env bash
# TODO #23 — Chen-style fair-eviction baseline (α_r = n_r / n).
# Re-run on the 4 synthetic (corpus, ctx) cells. Rows append to ac_combined.csv.
set -e
TS=$1
if [[ -z "$TS" ]]; then
  echo "usage: $0 <timestamp>" >&2; exit 2
fi
MASTER=logs/fair_rate_master_${TS}.log
PY=.venv/bin/python
SCRIPT=scripts/13_snapkv_role_conditional.py
COMMON="--max-prompts 50 --skip-snapkv-only --policy fair-rate"

run() {
  local ctx=$1 src=$2 short=$3
  local logf=logs/fair_rate_${short}_${TS}.log
  echo "[$(date -u +%H:%M:%S) start] ctx=$ctx src=$src" >> "$MASTER"
  $PY $SCRIPT --ctx-tokens $ctx --source-filter $src $COMMON > "$logf" 2>&1
  echo "[$(date -u +%H:%M:%S)  done] ctx=$ctx src=$src" >> "$MASTER"
}

echo "fair-rate sweep launched ${TS}" > "$MASTER"
date -u >> "$MASTER"

run 8000  synthetic_json          json_ctx8k
run 8000  synthetic_json_compact  compact_ctx8k
run 8000  synthetic_xml           xml_ctx8k
run 16000 synthetic_json          json_ctx16k

echo "[$(date -u +%H:%M:%S) all done]" >> "$MASTER"

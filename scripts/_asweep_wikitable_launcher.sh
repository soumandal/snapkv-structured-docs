#!/usr/bin/env bash
# α-sweep over α_KEY ∈ {0.02, 0.05, 0.10, 0.20} on wikitable_long ctx 8k.
# Existing α=0 (`no-key`) row already in CSV.
# Total: 4 runs, ~1.5 h wall.
set -e
TS=$1
if [[ -z "$TS" ]]; then
  echo "usage: $0 <timestamp>" >&2; exit 2
fi
MASTER=logs/asweep_wt_master_${TS}.log
PY=.venv/bin/python
SCRIPT=scripts/13_snapkv_role_conditional.py
COMMON="--max-prompts 50 --skip-snapkv-only"

run() {
  local ctx=$1 src=$2 pol=$3 short=$4
  local logf=logs/asweep_${short}_${TS}.log
  echo "[$(date -u +%H:%M:%S) start] ctx=$ctx src=$src pol=$pol" >> "$MASTER"
  $PY $SCRIPT --ctx-tokens $ctx --source-filter $src --policy $pol $COMMON > "$logf" 2>&1
  echo "[$(date -u +%H:%M:%S)  done] ctx=$ctx src=$src pol=$pol" >> "$MASTER"
}

echo "α-sweep (wikitable_long) launched ${TS}" > "$MASTER"
date -u >> "$MASTER"

run 8000 wikitable_long no-key-soft02 wt_ctx8k_a02
run 8000 wikitable_long no-key-soft    wt_ctx8k_a05
run 8000 wikitable_long no-key-soft10 wt_ctx8k_a10
run 8000 wikitable_long no-key-soft20 wt_ctx8k_a20

echo "[$(date -u +%H:%M:%S) all done]" >> "$MASTER"

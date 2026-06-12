#!/usr/bin/env bash
# α-sweep over α_KEY ∈ {0.02, 0.10, 0.20} on:
#   - indented synthetic_json ctx 8k + 16k
#   - synthetic_json_compact ctx 8k
#   - synthetic_xml ctx 8k
# Existing α=0 (`no-key`) and α=0.05 (`no-key-soft`) rows already in CSV.
# Total: 12 runs, ~5.5 h wall.
set -e
TS=$1
if [[ -z "$TS" ]]; then
  echo "usage: $0 <timestamp>" >&2; exit 2
fi
MASTER=logs/asweep_master_${TS}.log
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

echo "α-sweep launched ${TS}" > "$MASTER"
date -u >> "$MASTER"

# ctx 8k — indented JSON (3 α values)
run 8000  synthetic_json          no-key-soft02 json_ctx8k_a02
run 8000  synthetic_json          no-key-soft10 json_ctx8k_a10
run 8000  synthetic_json          no-key-soft20 json_ctx8k_a20

# ctx 8k — compact JSON (3 α values)
run 8000  synthetic_json_compact  no-key-soft02 compact_ctx8k_a02
run 8000  synthetic_json_compact  no-key-soft10 compact_ctx8k_a10
run 8000  synthetic_json_compact  no-key-soft20 compact_ctx8k_a20

# ctx 8k — XML (3 α values)
run 8000  synthetic_xml           no-key-soft02 xml_ctx8k_a02
run 8000  synthetic_xml           no-key-soft10 xml_ctx8k_a10
run 8000  synthetic_xml           no-key-soft20 xml_ctx8k_a20

# ctx 16k — indented JSON (3 α values) — most expensive, last
run 16000 synthetic_json          no-key-soft02 json_ctx16k_a02
run 16000 synthetic_json          no-key-soft10 json_ctx16k_a10
run 16000 synthetic_json          no-key-soft20 json_ctx16k_a20

echo "[$(date -u +%H:%M:%S) all done]" >> "$MASTER"

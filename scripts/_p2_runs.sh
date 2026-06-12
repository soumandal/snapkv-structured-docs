#!/usr/bin/env bash
# P2 optional / reviewer-defense experiments (§17 of plan2_findings.md).
# Two GPU runs that are already fully supported by scripts/13:
#   (1) ctx=32k snapkv-no-key (Llama) -> cross-ctx scaling figure. Tests whether
#       the low-budget gap-closure (63% at 16k) saturates or keeps climbing at 32k.
#       ctx_32000 Llama dumps (n=200) already exist under WW20-pilot for role labels.
#   (2) Task #19 delim-priority: flip role_precedence so DELIM > KEY, then run
#       α=0 (policy no-key) on the two corpora that collapse to 0.000 under the
#       default precedence (synthetic_xml, synthetic_json_compact). If α=0 now
#       works on these, the bimodal-α story collapses to a single α=0 claim.
#       --delim-priority forces runtime labeling (ignores cached parquet roles),
#       so ctx 8000 matches the prior xml/compact cells (§9 line 333).
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export PYTHONPATH="$PWD"
PY=.venv/bin/python
SCRUB='it/s\]|warnings.warn|UserWarning|Starting from v4.46|Loading checkpoint|FutureWarning'

echo "############ $(date -u +%H:%M:%S) | P2 RUNS START ############"

# ---------- (1) ctx=32k snapkv-no-key (Llama) ----------
echo "===== $(date -u +%H:%M:%S) | ctx32k llama no-key ====="
$PY scripts/13_snapkv_role_conditional.py \
  --model llama --source-filter synthetic_json --ctx-tokens 32000 \
  --max-prompts 50 --policy no-key \
  --out-path results/plan2/ac_combined_ctx32k.csv \
  2>&1 | grep -vE "$SCRUB"
echo "===== $(date -u +%H:%M:%S) | ctx32k DONE (rc=${PIPESTATUS[0]}) ====="

# ---------- (2) Task #19 delim-priority on the collapse corpora ----------
for SRC in synthetic_xml synthetic_json_compact; do
  echo "===== $(date -u +%H:%M:%S) | delimprio $SRC (alpha=0) ====="
  $PY scripts/13_snapkv_role_conditional.py \
    --model llama --source-filter "$SRC" --ctx-tokens 8000 \
    --max-prompts 50 --policy no-key --delim-priority \
    --out-path results/plan2/ac_delimprio_${SRC}.csv \
    2>&1 | grep -vE "$SCRUB"
  echo "===== $(date -u +%H:%M:%S) | delimprio $SRC DONE (rc=${PIPESTATUS[0]}) ====="
done

echo "############ $(date -u +%H:%M:%S) | ALL P2 RUNS DONE ############"

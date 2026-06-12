#!/usr/bin/env bash
# P1 experiments that materially complete the paper (§17 of plan2_findings.md):
#   (1) Pool-kernel ablation (C7 curve) — Llama, snapkv-only across pool ∈ {3,5,7,11,15}.
#       kernel=7 must reproduce the headline snapkv-only ctx16k cell
#       (0.20/0.34/0.70/0.80/0.88) since that is SnapKV's default pool.
#   (2) Phi-3 eviction cell — C2 (scripts/05) + C8/C9 (scripts/13) + B=1.0 oracle.
#       Phi uses the chat template and is verbose like Qwen, so --max-new-tokens 64
#       (the resume block in §15.3 predates the Qwen truncation lesson). scripts/05
#       must take --run-root explicitly: dumps are WW22 but the current week is WW23.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export PYTHONPATH="$PWD"
PY=.venv/bin/python
SCRUB='it/s\]|warnings.warn|UserWarning|Starting from v4.46|Loading checkpoint|FutureWarning'

echo "############ $(date -u +%H:%M:%S) | P1 RUNS START ############"

# ---------- (1) Pool-kernel ablation (Llama C7 curve) ----------
for K in 7 5 3 11 15; do
  echo "===== $(date -u +%H:%M:%S) | poolsweep llama kernel=$K ====="
  $PY scripts/13_snapkv_role_conditional.py \
    --model llama --source-filter synthetic_json --ctx-tokens 16000 \
    --max-prompts 50 --policy no-key --only-snapkv-only --pool-kernel "$K" \
    --out-path results/plan2/ac_poolsweep_llama.csv \
    2>&1 | grep -vE "$SCRUB"
done
echo "===== $(date -u +%H:%M:%S) | poolsweep DONE ====="

# ---------- (2) Phi-3 eviction cell ----------
echo "===== $(date -u +%H:%M:%S) | phi C2 (scripts/05) ====="
$PY scripts/05_counterfactual_eviction.py \
  --model phi --use-chat-template --ctx-tokens 16000 --max-prompts 50 \
  --source-filter synthetic_json --max-new-tokens 64 \
  --run-root /mnt/kvr/dumps/2026-WW22-pilot-phi-ct \
  --out-path results/plan2/e_counterfactual_phi.csv \
  2>&1 | grep -vE "$SCRUB"

echo "===== $(date -u +%H:%M:%S) | phi C8/C9 (scripts/13) ====="
$PY scripts/13_snapkv_role_conditional.py \
  --model phi --use-chat-template --ctx-tokens 16000 --max-prompts 50 \
  --source-filter synthetic_json --policy no-key --max-new-tokens 64 \
  --out-path results/plan2/ac_combined_phi.csv \
  2>&1 | grep -vE "$SCRUB"

echo "===== $(date -u +%H:%M:%S) | phi B=1.0 oracle (scripts/13) ====="
$PY scripts/13_snapkv_role_conditional.py \
  --model phi --use-chat-template --ctx-tokens 16000 --max-prompts 50 \
  --source-filter synthetic_json --policy no-key --max-new-tokens 64 \
  --budgets 1.0 --skip-snapkv-only \
  --out-path results/plan2/ac_combined_phi.csv \
  2>&1 | grep -vE "$SCRUB"

echo "############ $(date -u +%H:%M:%S) | ALL P1 RUNS DONE ############"

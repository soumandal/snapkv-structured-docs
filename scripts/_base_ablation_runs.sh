#!/usr/bin/env bash
# Base-model (Llama-3.1-8B, non-instruct) ablation — instruct-tuning-artifact defense.
# Runs C2 (counterfactual) then C8/C9/C7 (combined headline) sequentially on the
# shared A100. Pilot dumps + Q-analysis (C1) already done by a prior session.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="$PWD"
# Home / is full (other model caches). Use the /mnt/cache HF store where the
# prior session downloaded the complete base model; /tmp is on the same disk.
export HF_HOME=/mnt/cache/hf
PY=.venv/bin/python
RUN_ROOT=/mnt/kvr/dumps/2026-WW23-pilot-llama-base

echo "===== [1/2] counterfactual (C2) $(date -u) ====="
# Start fresh — prior run died at ~5/50, partial CSV removed.
rm -f results/plan2/e_counterfactual_llama-base.csv
$PY scripts/05_counterfactual_eviction.py \
  --model llama-base \
  --run-root "$RUN_ROOT" \
  --ctx-tokens 16000 \
  --max-prompts 50 \
  --source-filter synthetic_json \
  --out-path results/plan2/e_counterfactual_llama-base.csv

echo "===== [2/2] combined headline (C8/C9/C7) $(date -u) ====="
rm -f results/plan2/ac_combined_llama-base.csv
$PY scripts/13_snapkv_role_conditional.py \
  --model llama-base \
  --policy no-key \
  --run-root "$RUN_ROOT" \
  --ctx-tokens 16000 \
  --max-prompts 50 \
  --source-filter synthetic_json \
  --out-path results/plan2/ac_combined_llama-base.csv

echo "===== ALL DONE $(date -u) ====="

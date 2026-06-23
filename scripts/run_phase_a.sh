#!/bin/bash
# Phase A: baseline (reader=none) vs reader (reader=vertical) at the full d10 budget.
# Identical config except --reader, so nanochat auto-derives an identical token budget /
# batch / schedule (get_scaling_params counts only transformer_matrices + lm_head).
#
# Run from repo root with CUDA_VISIBLE_DEVICES set to the free GPUs, e.g.:
#   CUDA_VISIBLE_DEVICES=2,3 NPROC=2 nohup bash scripts/run_phase_a.sh > phase_a.log 2>&1 &
#
# window-pattern=L: A100 has no FA3, and SDPA can't do sliding windows efficiently, so use
# full-context attention for both runs (held constant -> comparison stays clean).
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export OMP_NUM_THREADS=1
NPROC=${NPROC:-2}
COMMON="--depth=10 --window-pattern=L --eval-every=100 --eval-tokens=2097152 --core-metric-every=-1 --sample-every=-1"

run() {
  local R=$1 TAG=$2
  echo "===================================================================="
  echo "===== $(date) START reader=$R tag=$TAG ====="
  echo "===================================================================="
  uv run torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- $COMMON --reader="$R" --model-tag="$TAG"
  echo "===== $(date) END reader=$R tag=$TAG ====="
}

run none d10_baseline
run vertical d10_reader
echo "PHASE_A_DONE_OK"

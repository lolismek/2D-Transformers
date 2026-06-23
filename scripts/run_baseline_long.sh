#!/bin/bash
# Compute-matched baseline: stock nanochat d10 (reader=none) trained LONGER so its TOTAL training
# FLOPs equal the d10_wide640_full WideReader run.
#
# WideReader costs 2.122x the baseline's fwd+bwd FLOPs/token, and the wide run is 1605 steps, so:
#     1605 * 2.122 = 3406 steps   (--num-iterations 3406)
# matches the wide run's total compute. This run is the iso-FLOP (compute-axis) control: wide@1605
# vs baseline@3406 consume the same FLOPs, so comparing their final bpb answers "does the depth
# reader beat plain top-state reading at EQUAL COMPUTE?".
#
# Everything else matches scripts/run_phase_a.sh EXACTLY (depth, window-pattern=L full attention,
# eval cadence, default device-batch-size, seed) => the ONLY difference vs the old d10_baseline is
# the longer schedule (warmdown stretched over 3406 steps; anneals to the floor at the new endpoint).
#
#   CUDA_VISIBLE_DEVICES=2,3 NPROC=2 nohup bash scripts/run_baseline_long.sh > baseline_long.log 2>&1 &
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export OMP_NUM_THREADS=1
NPROC=${NPROC:-2}
STEPS=${STEPS:-3406}
TAG=${TAG:-d10_baseline_long}

# Identical to Phase A's baseline arm; only --num-iterations is overridden (default d10 is 1605).
COMMON="--depth=10 --window-pattern=L --eval-every=100 --eval-tokens=2097152 --core-metric-every=-1 --sample-every=-1"
BASE="--reader=none --num-iterations=$STEPS"

echo "===================================================================="
echo "===== $(date) START reader=none tag=$TAG steps=$STEPS (iso-FLOP vs d10_wide640_full) ====="
echo "===================================================================="
uv run torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- $COMMON $BASE --model-tag="$TAG"
echo "===== $(date) END reader=none tag=$TAG ====="
echo "BASELINE_LONG_DONE_OK"

#!/bin/bash
# Iso-total-FLOP WideReader experiment: a full-width depth reader (d_V = n_embd = 640, NO up/down)
# trained at the SAME total training FLOPs as the d10 baseline.
#
# WideReader's bidirectional blocks run at 640^2 over all 11 rungs, so the reader's TRUE fwd+bwd
# cost is 6 * 9.83M params * 11 rungs ~= 6.49e8 FLOPs/token, on top of the baseline's 5.78e8 =>
# 2.12x/token. To match the baseline's total training FLOPs we cut the step budget:
#     1605 / 2.12 = 756 steps   (--num-iterations 756)
# NOTE: estimate_flops() undercounts the reader ~11x (it is blind to the rung axis), so the budget
# is set by hand here, NOT via --target-flops. The reader therefore sees ~47% of the tokens (~395M
# vs 841M): this is the iso-compute question "wide depth-reader (fewer H tokens) vs plain baseline".
#
# Everything else matches scripts/run_phase_a.sh EXACTLY (depth, window-pattern=L full attention,
# eval cadence, batch/seed defaults) => a clean A/B against d10_baseline.
#
# Full run (GPUs 2,3 were free):
#   CUDA_VISIBLE_DEVICES=2,3 NPROC=2 nohup bash scripts/run_wide640.sh > wide640.log 2>&1 &
# Timing/stability pilot (throwaway tag, 20 steps):
#   CUDA_VISIBLE_DEVICES=2,3 NPROC=2 STEPS=20 TAG=d10_wide_pilot bash scripts/run_wide640.sh
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export OMP_NUM_THREADS=1
NPROC=${NPROC:-2}
STEPS=${STEPS:-756}
TAG=${TAG:-d10_wide640}
LAYERS=${LAYERS:-2}   # number of bidirectional reader blocks over the depth axis (V's height).
                      # LAYERS=2 is the original WideReader; bump for the taller-V capacity sweep.
DBS=${DBS:-16}   # device microbatch; smaller than baseline's 32 because the full-width reader's
                 # (B*T, 11, 640) activations are ~11x an H layer's. grad-accum auto-grows to hold
                 # --total-batch-size fixed => same effective batch + same 756-step budget, just
                 # smaller microbatches (mathematically equivalent; not a comparison confound).

# Match Phase A; only the reader (wide, full-width; n_heads = n_embd//128 = 5 heads at d10, == H's
# n_head), the step budget, and the memory-driven microbatch size differ. --reader-dim=640 is passed
# only so the logged/checkpoint config is honest (WideReader ignores it and uses n_embd).
COMMON="--depth=10 --window-pattern=L --eval-every=100 --eval-tokens=2097152 --core-metric-every=-1 --sample-every=-1"
WIDE="--reader=wide --reader-dim=640 --reader-layers=$LAYERS --num-iterations=$STEPS --device-batch-size=$DBS"

echo "===================================================================="
echo "===== $(date) START reader=wide tag=$TAG steps=$STEPS (iso-FLOP) ====="
echo "===================================================================="
uv run torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- $COMMON $WIDE --model-tag="$TAG"
echo "===== $(date) END reader=wide tag=$TAG ====="
echo "WIDE640_DONE_OK"

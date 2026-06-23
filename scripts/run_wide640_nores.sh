#!/bin/bash
# Residual-free H + WideReader experiment.
# Same full-width depth reader (d_V = n_embd = 640, no up/down) as run_wide640.sh, but the H backbone
# is made RESIDUAL-FREE (--h-residual=none): both in-block skips (attn, mlp) AND the per-layer x0
# injection are removed, so each depth rung is a pure transform of the previous one instead of a
# partial sum of one telescoping series.
#
# Thesis under test: the residual stream's free additive depth-aggregation is what makes the depth
# reader redundant (h_L already holds the running sum). Remove it and V's depth-reading finally has
# genuinely distinct rungs to combine -- does V's marginal value grow?
#
# V (the reader) KEEPS its own residual connections -- only H is residual-free.
#
# Equal-DATA budget: 1605 steps == the same 841M tokens / schedule as wide@1605 (0.898 bpb) and the
# stock baseline@1605 (0.877 bpb), so all three val-bpb curves overlay directly. The matched
# residual-free baseline (reader=none --h-residual=none) is a separate run, decided after seeing this.
#
# Scope to FREE A100s (check `nvidia-smi` first; only use GPUs that are idle).
# Full run:
#   CUDA_VISIBLE_DEVICES=<free> NPROC=2 nohup bash scripts/run_wide640_nores.sh > wide640_nores.log 2>&1 &
# Stability/timing pilot (throwaway tag, 20 steps -- confirms the residual-free trunk trains):
#   CUDA_VISIBLE_DEVICES=<free> NPROC=2 STEPS=20 TAG=d10_wide_nores_pilot bash scripts/run_wide640_nores.sh
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export OMP_NUM_THREADS=1
NPROC=${NPROC:-2}
STEPS=${STEPS:-1605}
TAG=${TAG:-d10_wide640_nores}
DBS=${DBS:-16}   # full-width reader activations (B*T, 11, 640) ~11x an H layer => smaller microbatch;
                 # grad-accum auto-grows to hold --total-batch-size fixed (mathematically equivalent).

# Matches run_wide640.sh EXACTLY except --h-residual=none (residual-free H trunk).
COMMON="--depth=10 --window-pattern=L --eval-every=100 --eval-tokens=2097152 --core-metric-every=-1 --sample-every=-1"
WIDE="--reader=wide --reader-dim=640 --h-residual=none --num-iterations=$STEPS --device-batch-size=$DBS"

echo "===================================================================="
echo "===== $(date) START reader=wide h_residual=none tag=$TAG steps=$STEPS ====="
echo "===================================================================="
uv run torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- $COMMON $WIDE --model-tag="$TAG"
echo "===== $(date) END reader=wide h_residual=none tag=$TAG ====="
echo "WIDE640_NORES_DONE_OK"

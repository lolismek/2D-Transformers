#!/bin/bash
# Download climbmix shards + train the tokenizer for the nanochat 10+2 experiments.
# Run from the repo root. All artifacts land in $NANOCHAT_BASE_DIR (home; default ~/.cache/nanochat).
# Mirrors the speedrun overlap trick: download enough shards for the tokenizer, then train the
# tokenizer while the remaining shards download in the background.
#
#   N_TOK   shards needed before tokenizer training (~2B chars => ~8 shards)
#   N_TRAIN total train shards to end up with (d10 budget ~0.85B tokens => ~15 shards; 24 = margin)
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"
N_TOK=${N_TOK:-8}
N_TRAIN=${N_TRAIN:-24}

echo "[setup_data] base_dir=$NANOCHAT_BASE_DIR  N_TOK=$N_TOK  N_TRAIN=$N_TRAIN"
echo "[setup_data] $(date) downloading $N_TOK shards for the tokenizer..."
uv run python -m nanochat.dataset -n "$N_TOK" -w 8

echo "[setup_data] $(date) backgrounding download of remaining shards (-> $N_TRAIN)..."
uv run python -m nanochat.dataset -n "$N_TRAIN" -w 8 > extra_shards.log 2>&1 &
DLPID=$!

echo "[setup_data] $(date) training tokenizer (vocab 32768, ~2B chars)..."
uv run python -m scripts.tok_train
uv run python -m scripts.tok_eval
echo "[setup_data] $(date) TOKENIZER_DONE_OK"

wait "$DLPID"
echo "[setup_data] $(date) ALL_SHARDS_DONE_OK"

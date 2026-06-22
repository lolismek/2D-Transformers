# 2D-Transformers

A second transformer that attends over the **depth (layer) axis** of a GPT.

A normal transformer combines its per-layer hidden states by the fixed additive **residual
stream** and reads out the top layer — there is no attention *between* layers. This project
replaces that fixed depth-combination with a learned, bidirectional, content-based one: for each
token, the stack of per-layer states `[h_0, h_1, …, h_L]` is treated as a sequence along depth and
a shared "vertical" transformer attends over it. See [`PRD.md`](PRD.md) for the full motivation.

## Layout

```
reference/nanogpt/   PRISTINE upstream karpathy/nanoGPT — reference only, never edited
src/                 our training harness (owned)
  model.py             base GPT; depth-combination is pluggable (the `arch`)
  arch/                ONE module per architecture, selected by config `arch=...`
    __init__.py          registry + build_combiner()
    baseline.py          vanilla top-layer readout (bit-for-bit GPT-2)
    vertical.py          the vertical (over-layers) transformer  ← V1
  train.py             arch-agnostic training loop + optional freeze-base
  sample.py            text generation
  configurator.py      nanoGPT's config mechanism
configs/             one file per experiment (e.g. vertical_frozen.py)
analysis/            inspection / visualization (inspect_vertical.py, viz_vertical.py, …)
data/openwebtext/    data prep (prepare.py / prepare_stream.py); .bin lands here (gitignored)
runs/                checkpoints + figures (gitignored)
```

**Adding a new architecture** is one file + one config: drop a combiner in `src/arch/`, register
it in `src/arch/__init__.py`, and point a config at it with `arch = '<name>'`. No edits to
`model.py` / `train.py`. A combiner maps the per-layer stack `H = (B, T, S, d)` → readout
`(B, T, d)`; see `src/arch/vertical.py` for the interface (`STATE_KEY`, `needs_stack`, `forward`,
optional `init_identity`).

## Running (from the repo root)

```bash
# 1. data: tokenize OpenWebText -> data/openwebtext/{train,val}.bin
python data/openwebtext/prepare.py

# 2. train V1: vertical transformer on a frozen GPT-2 124M
python src/train.py configs/vertical_frozen.py
# checkpoints -> runs/vertical_frozen/ckpt.pt

# 3. analyze: samples + depth-attention, and heatmap figures
python analysis/inspect_vertical.py
python analysis/viz_vertical.py        # -> runs/vertical_frozen/viz/*.png

# init-parity sanity check (a fresh vertical model must equal vanilla GPT-2)
python analysis/parity_check.py
```

Compute runs on `kingcrab` (4×V100) in the `2dtf` conda env; the workspace there mirrors this
repo at `~/2d-Transformers`.

## V1 result (frozen GPT-2 124M + vertical, OpenWebText, 5000 iters)

Val loss **3.1046 → 2.9918** (~10.7% lower perplexity) training only ~4.7M params. The depth
attention is non-degenerate but dominated by the **top layer (~31%) + embedding (~23%)**, with
heads specializing by depth — i.e. a real but first/last-dominated, fairly static-looking
combination. Next planned controls: a static per-layer weighting and an embedding-skip-only
baseline, to isolate how much of the gain needs content-dependent attention.

## Reference

`reference/nanogpt/` is an unmodified vendored copy of [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
(MIT). All of our code lives under `src/`, `configs/`, `analysis/`, and `data/`.

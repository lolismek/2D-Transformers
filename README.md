# 2D Transformers — experiments on depth-axis ("vertical") reading

Research code for a running series of experiments on **vertical transformers**: treating the
**depth (layer) axis** of a language model as a sequence in its own right, and asking what a model
can learn by *reading across* that axis instead of predicting from only the top layer's hidden state.

This repo is **not** a single thesis with a fixed direction. It's an experiment log. The north star
is steady — *understand what the depth ladder `[x0, h_1, …, h_L]` carries, and whether reading it
buys anything* — but each experiment attacks that from whatever angle is sharpest next, and is
measured on whichever axis the question calls for: sometimes **parameters**, sometimes **FLOPs**,
sometimes **data**, sometimes **architecture**. Results below are reported as we find them, including
the (many) negatives.

The backbone is **[karpathy/nanochat](https://github.com/karpathy/nanochat)** (MIT, commit
`dc54a1a`), vendored under `nanochat/`. We add a pluggable depth-reader and a few **gated** edits;
with `--reader=none --h-residual=full` the training path is byte-identical to stock nanochat, so the
baseline is a faithful control. All runs are depth-10 (`d=640`, 10 layers, 5 heads) unless noted.

## The recurring setup — "10+2"

A stock nanochat depth-10 backbone **H** produces the 11-rung residual ladder `[x0, h_1, …, h_10]`;
a small bidirectional **reader V** over those rungs produces the readout — **no `h_10` skip, no gate,
no identity init**, so V has to *earn* its keep against the top-state baseline. Variations on V's
width and depth, the compute/data budget, and even H's residual structure are what the experiments
below dial.

> **Guiding question.** Does reading the whole depth ladder beat reading just the top rung `h_10` —
> and if not, *why* doesn't it?

## Experiment log

| # | experiment | axis | question | headline (val bpb) | verdict |
|---|---|---|---|---|---|
| 1 | [Phase A — `d_V=128` reader](#1-phase-a--d_v128-reader-vs-baseline) | **params** | does a ~free (+1.1%) depth-reader beat top-state? | base **0.877** vs reader 0.933 (**+0.056**) | no |
| 2 | [Readout-bottleneck probes](#2-readout-bottleneck-probes-no-retrain) | **rank** | is the 128-dim cap the reason? | PCA-128 +0.454 → trained-128 +0.167 → reader-128 +0.056 | cap is real but doesn't fully explain it |
| 3 | [WideReader `d_V=640` iso-FLOP](#3-widereader-d_v640--iso-flop) | **FLOPs** | does removing the cap rescue it, at equal compute? | base **0.877** vs wide 0.928 (**+0.051**) | no — cap largely exonerated |
| 4 | [WideReader — full budget](#4-widereader--full-budget-equal-data-vs-equal-compute) | **data + FLOPs** | clean per-token vs per-FLOP read | equal-data **+0.021**; equal-compute **+0.053** | modestly worse/token, clearly worse/FLOP |
| 5 | [Residual-free WideReader](#5-residual-free-widereader) | **architecture** | do H's residuals "steal V's job"? | base **0.877** vs no-resid 0.928 (**+0.051**) | no — removing them makes V *worse* |
| 6 | [Taller V — reader depth](#6-taller-v--reader-depth) | **reader depth** | does a *taller* V help where a wider one didn't? | base **0.877** vs wide@4 **0.877** (ties; −0.021 vs wide@2) | yes — closes the gap; genuine-vs-collapse open |

**Detailed writeup with all probes & figures:**
[`experiments/nanochat_10p2_reader.md`](experiments/nanochat_10p2_reader.md). Per-experiment notes
follow.

### 1. Phase A — `d_V=128` reader vs baseline
*Axis: parameters (equal data, 841M tokens, 1605 steps).* The reader is `d_V=128`, 2 bidirectional
blocks over the 11 rungs, a learned query-pool, **+558,592 params (+1.14%)**. It loses by **+0.056**
bpb. But its depth-attention is **non-degenerate** — V almost entirely ignores the top rung
(`h_10` mass ≈ 0.0003) and reads the middle (rungs 8/6/9). So V *does* read the ladder; it just
loses while doing so. Leading suspect: the 128-dim `down 640→128 → up 128→640` bottleneck.
→ `figs/phase_a_val_bpb.png`.

### 2. Readout-bottleneck probes (no retrain)
*Axis: rank (diagnostics on the frozen baseline checkpoint, no training).* Two probes ask whether
128 dims is the cause. **PCA truncation** of the baseline's 640-dim readout to 128 costs **+0.454**
(but participation-ratio is only 5.9 — variance ≠ usefulness, so this is a loose upper bound). A
**trained 128-dim linear cap** on the frozen top state costs **+0.167** — the floor for a *linear*
128-cap. The ladder: PCA-128 **+0.454** → trained-linear-128 **+0.167** → reader-128 **+0.056** →
full-640 **0**. The co-adapted reader already *beats* a frozen linear cap, so it's not simply
bottleneck-limited — the probe implicates but can't pin the cap. Decisive test → experiment 3.

### 3. WideReader `d_V=640` — iso-FLOP
*Axis: FLOPs (equal total compute).* Remove the cap entirely: `WideReader` reads at native width
640, 5 heads / head_dim 128 (= H exactly), no `down`/`up`. It costs **2.122×/token**, so at equal
total FLOPs it trains 756 steps (~395M tokens, ~47% of baseline's). Result **0.928** = **+0.051** vs
baseline. Removing the cap closed only ~0.005 of the original 0.056 gap → the **128-dim bottleneck is
largely exonerated**; depth-reading-as-readout is itself the weaker bet at d10, not the width.
→ `figs/wide640_isoflop.png`.

### 4. WideReader — full budget (equal data vs equal compute)
*Axes: data and FLOPs.* The iso-FLOP test starved the reader of tokens, so two full runs separate
the questions cleanly: the WideReader at the **full 1605-step budget** (same 841M tokens, anneal-
matched), and a **compute-matched baseline** at 3406 steps (= 1605 × 2.122, equal total FLOPs).

| axis | baseline | WideReader `d_V=640` | Δ |
|---|---|---|---|
| **equal data** — 841M tok, 1605 steps | **0.877** | 0.898 | **+0.021** |
| **equal compute** — 1.03e18 FLOPs | **0.845** (3406 steps, 1.79B tok) | 0.898 (1605 steps) | **+0.053** |

Per *token* the reader is only modestly behind (**+0.021** — full-budget training pulled it
0.928 → 0.898, the iso-FLOP number was token-starved). Per *FLOP* it loses by more (**+0.053**): its
2.12× overhead is better spent on 2× the tokens, and data scales well here (0.877 → 0.845). The
reader's readout is *nearly* as good per token, but never wins. → `figs/flops_compare.png`,
`figs/tokens_compare.png`.

### 5. Residual-free WideReader
*Axis: architecture (equal data, 1605 steps).* A different hypothesis: maybe H's **residual
connections** are a "free additive aggregator" that makes `h_10` already hold the whole sum
(`h_L = x0 + Σ deltas`), leaving the offline reader V redundant. So remove them — drop both in-block
skips and the x0 injection, keeping the learned per-layer gain `resid_lambdas` and V's own residuals
(`--h-residual=none`; needs a fan-in `c_proj` init or the skip-less trunk is dead at init). If the
thesis held, V should now have a real job. It doesn't: residual-free WideReader is **0.928** —
**+0.051 vs baseline, +0.030 worse than the WideReader *with* residuals**.

**Removing H's residuals makes the depth-reader worse, not better.** V is an *offline* reader over
cached rungs and can't substitute for the residual stream's *online* accumulation — so a residual-
free H computes worse rungs *and* V reads worse rungs. → `figs/wide640_nores_compare.png`.

### 6. Taller V — reader depth
*Axis: reader depth (equal data, 1605 steps).* Width didn't help (exp 3/4), so test the *other*
capacity axis: make V **taller** — 4 bidirectional blocks over the ladder instead of 2, everything
else fixed (full width `d_V=640`, residuals normal). It closes the **entire** +0.021 gap: wide@4 =
**0.877**, landing essentially *on* baseline (Δ +0.0001), −0.021 below wide@2. So the reader was
**depth-capacity-limited, not width-limited** — a genuine surprise that revises exp 3/4's "capacity
exonerated" (that was width only). → `figs/wide640_layers_compare.png`.

**Open — not yet resolved.** "Ties baseline" has two readings: **(A)** 4 layers let V extract a
readout from the ladder as good as the top state (then more depth might *beat* it), or **(B)** the
extra capacity let V collapse to mimicking `h_10` (and since baseline *is* `h_10`, that also lands on
baseline — depth-reading still does nothing). The exact match fits either; `inspect_reader` on the L4
checkpoint (query-pool mass on rung 10 = collapse; spread = genuine) plus a rung-ablation
disambiguate. And note the cost: even at parity wide@4 burns ~3.2×/token, so to be *worth it*
depth-reading must beat, not tie — the open `reader_layers=6` run tests that.

## What we've learned about vertical transformers (so far)

- **The depth ladder is genuinely structured.** With 2 reader blocks V doesn't collapse onto `h_10`;
  it spreads attention over the middle rungs. There *is* something to read on the depth axis.
- **Reaching the top state took reader *depth*, not width.** A wider V barely moved (exp 3/4,
  +0.021); a *taller* V (exp 6, 4 layers) closed the whole gap and **tied** baseline. So V was
  depth-capacity-limited — but note it *ties*, it has not *beaten* the top state, and at ~3.2×/token
  it loses per-FLOP. Whether the tie is genuine ladder-reading or V learning to mimic `h_10` is the
  current open question.
- **The unifying mechanism (still the best account of the negatives):** nanochat's residual stream
  makes every rung a partial sum of one telescoping series, so `h_10` already holds the whole sum —
  re-aggregating the ladder offline is largely redundant. Exp 5 (residual-free) tested it head-on:
  removing the residuals made V *worse*, confirming the online accumulation does the work.
- **Open directions:** settle the exp-6 tie (genuine vs collapse) and push reader depth further
  (`reader_layers=6` — does it *beat* baseline?); more H depth (more, less-redundant rungs to read);
  V as an *addition* to the top-state path rather than a *replacement*.
- **MLM on the ladder (idea, not run).** Attacks the redundancy mechanism head-on. *Bidirectional H*
  (BERT-style masked-token objective) would make the top layer task-specialized and *shed* info that
  middle rungs keep (the classic "best layer is in the middle"), so for the first time the ladder is
  non-redundant and V has a real job — could *beat* the top state, not just tie. Caveats: drops bpb-
  comparability + the generative stack, and is partly circular (the objective, not V, removes the
  redundancy). **Pre-test cheaply first:** per-layer probe (`frozen_probe.py`) on a bidirectional H —
  do middle rungs beat the top? If not, MLM buys V nothing. Note the *mask-a-rung* variant (denoise a
  dropped rung from its neighbors) is a dead end: the residual stream makes it near-trivial (copy the
  neighbor) and only reinforces the redundancy we're fighting.

## Repository layout

| path | owner | what |
|---|---|---|
| `nanochat/` | vendored | nanochat backbone (MIT, Karpathy). Upstream docs: [`nanochat/README.md`](nanochat/README.md). |
| `nanochat/readers/` | **ours** | depth-reader plugin — `base.py` (interface), `vertical.py` (`d_V=128`), `wide.py` (`d_V=640`), registry in `__init__.py`. |
| `nanochat/h_variants.py` | **ours** | H-backbone variants — `ResidualFreeBlock` (experiment 5). |
| `nanochat/gpt.py`, `nanochat/optim.py` | vendored **+ our edits** | gated edits (below). |
| `scripts/` | mixed | nanochat entrypoints (`base_train.py` **+ our CLI**, `base_eval.py`, `tok_*`, `chat_*`) **and** our experiment scripts (below). |
| `experiments/` | **ours** | writeups, figures (`figs/`), and data (CSV/JSON). |
| `dev/`, `tasks/`, `tests/`, `pyproject.toml`, `uv.lock` | vendored | nanochat. |

The reader is a **plugin into nanochat's reader extension point**: `gpt.py` does
`from nanochat.readers import build_reader`, so adding a depth-reader is one module + one registry
line. H variants (like residual-free) are gated on `config.h_residual` and live in `h_variants.py`.

### Our edits to nanochat core (all gated — baseline stays stock nanochat)
- **`nanochat/gpt.py`**: when `config.reader != 'none'`, stash the per-block ladder and let the
  reader own the readout; when `config.h_residual == 'none'`, build `ResidualFreeBlock`, drop the x0
  injection, fan-in-init `c_proj`, and exclude `x0_lambdas` from the optimizer. Both default off →
  the stock top-state path is byte-identical. Plus reader param-counting + Muon/AdamW wiring.
- **`nanochat/optim.py`**: two `world_size > 1` reader-optimizer fixes and a `torch._dynamo`
  recompile-limit bump for the reader's extra weight shapes.
- **`scripts/base_train.py`**: the `--reader / --reader-dim / --reader-layers / --reader-heads /
  --reader-mlp-mult` and `--h-residual` CLI, threaded into `GPTConfig`.

### Our experiment scripts (in `scripts/`)
| script | purpose |
|---|---|
| `run_phase_a.sh` | exp 1 — d10 baseline vs `d_V=128` reader at full budget |
| `run_wide640.sh` | exp 3/4/6 — `d_V=640` WideReader (iso-FLOP 756, `STEPS=1605` full budget, `LAYERS=N` reader depth) |
| `run_baseline_long.sh` | exp 4 — compute-matched baseline (3406 steps) |
| `run_wide640_nores.sh` | exp 5 — residual-free WideReader (`--h-residual=none`, 1605 steps) |
| `setup_data.sh` | download data shards + train the tokenizer |
| `inspect_reader.py` | depth-attention diagnostic (per-rung query-pool weights) |
| `svd_readout_probe.py` · `frozen_probe.py` | exp 2 — PCA rank probe / trained 128-dim cap on the frozen baseline |
| `check_reader.py` · `check_reader_dist.py` · `check_wide_reader.py` · `check_residual_free.py` | data-free integration / distributed / shape / residual-free-init checks |
| `plot_phase_a.py` · `plot_wide_compare.py` · `plot_compute_data.py` · `plot_nores_compare.py` · `plot_layers_compare.py` | figures |
| `modal_train.py` | Modal (serverless GPU) launcher for WideReader runs — `MODAL_GPUS`, `--h-residual`, `--reader-layers`, `--dbs` (exp 5/6) |

## Setup & reproduce

Uses nanochat's toolchain (uv + Rust for the tokenizer). On a CUDA box:
```bash
uv sync
bash scripts/setup_data.sh                                      # data shards + tokenizer
CUDA_VISIBLE_DEVICES=2,3 NPROC=2 bash scripts/run_phase_a.sh     # exp 1: baseline vs d_V=128 reader
CUDA_VISIBLE_DEVICES=2,3 NPROC=2 bash scripts/run_wide640.sh     # exp 3: d_V=640 WideReader, iso-FLOP
CUDA_VISIBLE_DEVICES=2,3 NPROC=2 STEPS=1605 TAG=d10_wide640_full bash scripts/run_wide640.sh   # exp 4
CUDA_VISIBLE_DEVICES=2,3 NPROC=2 bash scripts/run_baseline_long.sh                              # exp 4
CUDA_VISIBLE_DEVICES=2,3 NPROC=2 bash scripts/run_wide640_nores.sh                              # exp 5
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 LAYERS=4 STEPS=1605 TAG=d10_wide640_L4_full bash scripts/run_wide640.sh  # exp 6
```
Experiments 5–6 were run on **Modal** (serverless 2–4× A100) via `scripts/modal_train.py` instead of
a local box, e.g. exp 6:
```bash
MODAL_GPUS=4 modal run scripts/modal_train.py --action train \
    --steps 1605 --reader-layers 4 --h-residual full --dbs 8 --tag d10_wide640_L4_full
```
Exact flags, the rank/frozen probes, and checkpoint paths are in
[`experiments/nanochat_10p2_reader.md`](experiments/nanochat_10p2_reader.md). Runs were on 2–4×
A100-40GB (local or Modal); `--window-pattern=L` (full-context attention) is used because A100 has no
FA3 kernel. (Modal's bare `A100` pool is 40/80GB-heterogeneous — `reader_layers=4` needs `--dbs 8` to
fit 40GB; it's grad-accum-compensated, so the result is unchanged.)

## Credits & license

Backbone: [karpathy/nanochat](https://github.com/karpathy/nanochat) @ `dc54a1a`, MIT © Andrej
Karpathy (see [`LICENSE`](LICENSE)). Our additions are released under the same MIT license.

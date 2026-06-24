# nanochat 10+2 — vertical reader: full experiment log

Does a small bidirectional **reader** over the depth axis (the nanochat 10+2 "vertical"
architecture) beat reading only the top-layer state? Across six experiments — on the parameter,
rank, FLOP, data, architecture, and reader-depth axes — the short answer is *not quite*: with enough
reader **depth** it reaches **parity** with the top state, but it never clearly beats it and pays
more to get there. This log records why. It runs Phase A (baseline vs `d_V=128` reader), the probes
asking whether the **128-dim bottleneck** is the cause, the `d_V=640` **WideReader** (iso-FLOP and
full-budget) that removes the bottleneck, the **residual-free** test that asks whether H's residual
stream was making the reader redundant, and the **taller-V** sweep that finds the deficit was a
reader-*depth* limit.

Related code: `nanochat/readers/vertical.py` (the `d_V=128` reader), `nanochat/readers/wide.py`
(the full-width `d_V=640` reader), `nanochat/h_variants.py` (`ResidualFreeBlock`),
`scripts/inspect_reader.py` (depth-attention diagnostic), `scripts/svd_readout_probe.py` (PCA
truncation), `scripts/frozen_probe.py` (trained 128-dim cap).

## Setup

- **Backbone H**: stock nanochat depth-10 (d=640, 10 layers, 5 heads), trained from scratch.
- **Reader V ("vertical")**: `d_V=128`, 2 bidirectional blocks over 11 rungs `[x0, h_1..h_10]`,
  2 heads, ReLU² MLP, learned query-pool. **V owns the readout** — no `h_10` skip, no gate, no
  identity init. **+558,592 params (+1.14% of transformer matrices).**
- Both runs share data / token budget (~841M tokens, 1605 iters) / seed; the reader is excluded
  from the scaling-param count → a clean A/B differing only in `--reader`. tigerfish, 2×A100
  (GPUs 2,3).

## Phase A — baseline vs reader (val bpb)

| model | val bpb | Δ |
|---|---|---|
| baseline (stock top-state readout) | **0.8770** | — |
| reader (`d_V=128`, owns readout) | **0.9326** | **+0.0556 (worse)** |

A clean, stable negative across the whole run. Per the pre-registered rule → no head/depth
ablations.

The depth-attention is **non-degenerate** (`inspect_reader.py`): V almost entirely ignores the top
rung (h_10 mass ≈ 0.0003) and concentrates on the middle (rung 8: 0.37, 6: 0.30, 9: 0.18; norm.
entropy 0.50). So V genuinely depth-reads — it just loses while doing so. Leading suspect: the
**128-dim bottleneck** (`down 640→128 → up 128→640` rank-caps the readout vs the baseline's full
640-dim top state).

## The question

Is the reader worse *because of* the 128-dim reduction in V? Two no-retrain probes on the
**baseline** checkpoint. Both hook the `lm_head` input (= the post-final-norm readout,
`gpt.py:504→508`) and reuse the stock `evaluate_bpb`, so the units match Phase A exactly.

## Probe 1 — PCA truncation (`svd_readout_probe.py`)

Stack the baseline's 640-dim readout over val tokens → centered covariance → PCA. For each `k`,
project onto the top-`k` **variance** subspace, reconstruct, and run the real lm_head + softcap +
CE. (`k=640` = exact reconstruction = correctness check.) Raw output:
`experiments/svd_probe_d10_baseline.json`.

| k | bpb | Δ vs full | cum. variance |
|---|---|---|---|
| 128 | 1.320 | **+0.454** | 0.73 |
| 256 | 1.044 | +0.178 | 0.83 |
| 384 | 0.928 | +0.062 | 0.90 |
| 512 | 0.882 | +0.017 | 0.96 |
| 640 | 0.866 | 0 (exact ✓) | 1.00 |

Participation ratio (variance effective-rank) = **5.9**. (Baseline bpb on this ~1M-token slice is
0.866; the official full-val number is 0.877. All `k` are compared on identical tokens, so the Δ's
are exact.)

**Variance ≠ usefulness.** The top-128 variance dims hold 73% of the variance, yet discarding the
rest costs +0.45 bpb; one direction alone holds 41% of the variance but is predictively cheap; the
head reads a high-dimensional, low-variance code (no plateau before 640). So PCA is a
*variance*-aligned (wrong-objective) cap — a loose upper bound that overshoots 8×. It **implicates
but cannot prove** the bottleneck.

## Probe 2 — trained 128-dim cap (`frozen_probe.py`)

Freeze the entire baseline (H + lm_head). Splice a trainable bottleneck before the frozen lm_head:
`readout → down 640→128 → up 128→640 → [frozen] lm_head`. Warm-start `down`/`up` from the PCA
solution (so step 0 reproduces +0.454), then train **only** `down`/`up` (164k params) on the
model's own loss via live forwards — which pairs features↔targets internally and uses the exact
head, so it is correct by construction and can only improve on the warm start.

| readout | bpb | Δ vs full |
|---|---|---|
| full 640-dim (baseline) | 0.866 | 0 |
| **trained 128-dim linear cap (frozen)** | **1.032** | **+0.167** |
| PCA 128-dim cap (variance) | 1.320 | +0.454 |

**The ladder:** PCA-128 **+0.454** → trained-linear-128 **+0.167** → reader-128 **+0.056** →
full-640 **0**. Each smarter use of the 128 dims shrinks the cost. Only PCA→trained is
single-variable (the subspace objective) and recovers 0.29 bpb — confirming variance≠prediction;
**+0.167 is the floor for a frozen *linear* 128-cap on the top state.**

## Verdict

- The 128-dim cap is a **real, material cost** (+0.167 trained-linear on frozen features — not
  free) ⇒ the bottleneck is **implicated, not refuted**. (Had this returned ~0 it would have
  *refuted* the bottleneck and saved a retrain.)
- But the co-adapted reader (+0.056) already **beats** the frozen cap (+0.167): its co-adaptation +
  ladder-reading + nonlinearity buy back ~0.11 over a plain 128-cap. So the reader is *not* simply
  bottleneck-limited, and the probe **cannot pin** the residual 0.056 to *width* vs the *readout
  mechanism*.
- **Decisive next test: the `d_V=640` retrain** (remove the cap entirely) — **done; see the next
  section.** Verdict: removing the bottleneck did *not* recover the deficit.

## d_V=640 WideReader — the decisive iso-FLOP test (`nanochat/readers/wide.py`)

Remove the 128-dim cap entirely and ask the bottleneck question at **equal total compute**. New
module `WideReader` (the old `VerticalReader` is untouched — every experiment kept): reads the ladder
at **native width 640**, 2 bidirectional depth-blocks, **5 heads / head_dim 128 (= H exactly)**,
learned query-pool, owns the readout. At `d_V = n_embd` the Phase-A `down`/`up` projections are
redundant (down is absorbable into block-0's input projections; up into the shared final-norm +
lm_head), so **both are dropped**. **9,838,080 params.**

The depth-blocks now run at `640²` over all 11 rungs → true fwd+bwd cost **648.8M FLOPs/token** on
top of the baseline's 578M = **2.122×/token**. To hold *total* training FLOPs = baseline, cut the
budget: `1605 / 2.122 ≈ 756 steps` (`--num-iterations 756`). `estimate_flops()` undercounts the
reader ~11× (blind to the rung axis), so the budget is set by hand, not via `--target-flops`.
`--device-batch-size=16` (the full-width `(B*T,11,640)` activations are ~11× an H layer's and OOM at
the default); grad-accum auto-grows 4→8 to hold the effective batch, so it is math-equivalent. The
catch baked into iso-FLOP: the wide run sees **~47% of the tokens** (~395M vs 841M).

| model | val bpb | steps | total FLOPs | tokens |
|---|---|---|---|---|
| baseline (top-state readout) | **0.877** | 1605 | 1.00 F | 841M |
| reader `d_V=128` (Phase A) | 0.933 | 1605 | 1.06 F | 841M |
| **WideReader `d_V=640`, iso-FLOP** | **0.928** | 756 | 1.00 F | ~395M |

**Verdict: widening did not rescue the reader.** Full width, H-matched heads, no bottleneck, no
down/up — and at equal compute it still loses to plain baseline by **+0.051** (0.928 vs 0.877).
Removing the 128-dim cap closed only ~0.005 of the original 0.056 gap, so the **bottleneck is largely
exonerated**: the depth-reading-as-readout approach is itself the weaker bet at d10, not the width.
(Secondary, weak: WideReader edged the bottlenecked reader 0.928 vs 0.933 *despite half the tokens* —
a faint hint that removing the cap helps per-token efficiency, nowhere near enough to matter.)
Caveat: this iso-FLOP test bundles the ~47% token cut. The cleaner "width at *fixed tokens*" run
(`d_V=640 × 1605 steps`) plus a compute-matched baseline were run next — **see the full-budget
section below**, which both refines (the per-token gap is smaller than this section implied) and
confirms (the reader still loses) this verdict. Plot `experiments/figs/wide640_isoflop.png` (val bpb
vs cumulative compute); data `experiments/figs/wide640_val.csv`.

## Full budget — equal data vs equal compute (`scripts/run_baseline_long.sh`, `scripts/plot_compute_data.py`)

The iso-FLOP test above is honest about *compute* but conflates two questions and starves the reader
of tokens (756 steps, ~395M). Two full runs separate them cleanly:

- **WideReader at the full budget** — same `wide.py` module, `--num-iterations 1605` (tag
  `d10_wide640_full`), so it sees the **same 841M tokens on the same schedule** as the baseline →
  directly overlay-able (anneal-matched at every step).
- **Compute-matched baseline** — stock `reader=none`, `--num-iterations 3406` (= 1605 × 2.122, tag
  `d10_baseline_long`), so its **total FLOPs equal the wide run's** (1.032e18, confirmed to 4 sig
  figs: baseline F/token = 5.78e8 logged, wide = +6.49e8 reader = 2.122×).

| axis | baseline | WideReader `d_V=640` | Δ |
|---|---|---|---|
| **equal data** — 841M tok, 1605 steps, anneal-matched | **0.877** | 0.898 | **+0.021** |
| **equal compute** — 1.032e18 FLOPs | **0.845** (3406 steps, 1.79B tok) | 0.898 (1605 steps) | **+0.053** |

**Verdict (refines, does not flip, the iso-FLOP section):**
- **Per token the reader is only modestly behind (+0.021), not +0.05.** Training to the full budget
  pulled the wide reader 0.928 → **0.898**: it was still descending at the 756-step cutoff, so the
  iso-FLOP number overstated the deficit. Removing the bottleneck *and* giving equal tokens closes
  most of the original 0.056 gap — the reader's readout is nearly as good per token, but never wins.
- **Per FLOP it loses by more (+0.053).** The reader's 2.122×/token overhead is better spent training
  the baseline on 2× the tokens, and data scales well here (baseline 0.877 → **0.845**). The old
  "iso-FLOP +0.051" was coincidentally close to this true equal-compute +0.053 (both sides dropped
  ~0.03 with 2× their tokens) but muddled the per-token question.
- **Bottom line:** depth-reading-as-readout is a dead end at d10 — not a capacity/bottleneck problem
  (it is exonerated), just that the top state `h_10` already exposes what the ladder carries.

Figures: `experiments/figs/flops_compare.png` (equal compute, FLOPs axis) and
`experiments/figs/tokens_compare.png` (equal data, tokens axis). Data:
`experiments/figs/{wide640_full_val.csv, baseline_long_val.csv}`.

## Residual-free WideReader — does H's residual stream make V redundant? (`nanochat/h_variants.py`)

Every result above lands on the same mechanism: nanochat's residual stream computes
`x_{i+1} = resid_λ·x_i + x0_λ·x0 + attn_i + mlp_i`, so each rung is a partial sum of one telescoping
series and the **top state `h_10` already holds the whole sum** `x0 + Σ deltas`. If that's why the
offline reader keeps losing — it's re-aggregating something `h_10` already has — then **removing the
residuals should give V a real job.** This experiment tests that thesis directly.

`--h-residual=none` strips H's free additive aggregator so each rung is a *pure deep transform* of
the previous one, not a partial sum (`ResidualFreeBlock`: `x = attn(norm(x)); x = mlp(norm(x))`, plus
the x0 injection dropped in the trunk loop). Kept: the learned per-layer gain `resid_lambdas` (it
scales the signal, it is not a skip), and **V's own residuals** (only H is made residual-free). Two
init/optimizer fixes are required and were verified data-free (`scripts/check_residual_free.py`):
with the skips gone, nanochat's zero-init `c_proj` would make the whole trunk emit 0 at init (and
relu²'(0)=0 kills the gradient), so `c_proj` is initialized at **fan-in scale** instead; and the now-
unused `x0_lambdas` is **excluded from the optimizer** (else its None grad crashes the distributed
all-reduce at `world_size>1` — a bug the Modal pilot caught). This is a **single arm** — one
residual-free WideReader run at the full 1605-step budget (`scripts/run_wide640_nores.sh`), compared
against the same-schedule, fully-annealed baseline and WideReader (so all three are equal-data,
per-token). No matched residual-free baseline control was run.

| readout (same 1605-step schedule = equal data) | val bpb | Δ vs baseline | Δ vs WideReader |
|---|---|---|---|
| baseline — top-state `h_10` | **0.877** | — | — |
| WideReader `d_V=640`, H **keeps** residuals | 0.898 | +0.021 | — |
| **WideReader `d_V=640`, H residual-free** | **0.928** | **+0.051** | **+0.030 (worse)** |

**Verdict: the thesis is wrong — removing H's residuals makes the reader worse, not better.** The
gap is monotone across the whole back half (every eval checkpoint from step 100), and both wide runs
are fully annealed, so it's a clean signal, not a slow-start artifact. The interpretation: V is an
**offline** reader over *cached* rungs and cannot substitute for the residual stream's **online**
accumulation. Take the residuals away and a residual-free H computes *worse* rungs **and** V reads
*worse* rungs — strictly worse on both counts. So the residual stream isn't "stealing V's job"; it's
doing a job (online depth-aggregation) that an offline reader fundamentally can't replace. (Aside:
the 0.928 here coincidentally equals the old iso-FLOP wide640@756 — unrelated; different token
budget.) Plot `experiments/figs/wide640_nores_compare.png` (3 curves, same schedule); data
`experiments/figs/wide640_nores_val.csv`.

## Taller V — was the reader depth-limited? (`reader_layers=4`)

Width was exonerated (the `d_V=640` WideReader didn't rescue the reader), which left the *other*
capacity axis untested: reader **depth**. This experiment makes V **taller** — 4 bidirectional blocks
over the ladder instead of 2 — holding everything else fixed (full width `d_V=640`, residuals normal,
same 1605-step / 841M-token schedule, so it's equal-data and directly overlay-able on baseline and
wide@2). Only `--reader-layers` changes (2→4). Run on Modal, 4× A100, via the generalized
`scripts/modal_train.py`.

| readout (same 1605-step schedule = equal data) | val bpb | Δ vs baseline | Δ vs wide@2 |
|---|---|---|---|
| baseline — top-state `h_10` | **0.8770** | — | — |
| WideReader `d_V=640`, **L=2** | 0.8980 | +0.0210 | — |
| **WideReader `d_V=640`, L=4** (taller V) | **0.8771** | **+0.0001** | **−0.0209** |

**Result: a taller V closes the entire +0.021 gap and ties baseline.** Going 2→4 reader layers lands
wide@4 essentially *on* the top-state baseline (Δ +0.0001), so the reader was **depth-capacity-limited,
not width-limited** — which revises the earlier "capacity exonerated" reading (that was width only).
The curve is monotone (one transient eval blip at step 400 = 1.44 that recovered next eval). Peak
memory 23.7 GB at `--dbs 8`; 53.6 min on 4× A100-40GB.

**Open — the result is a *tie*, and ties are ambiguous here.** Two readings fit the exact match:
- **(A) genuine parity** — 4 layers let V extract a readout from the ladder as good as the top state.
  Then more depth (`reader_layers=6`) might push *below* baseline = a real win.
- **(B) collapse to `h_10`** — the extra capacity let V learn to mostly ignore the ladder and
  reconstruct the top rung; since baseline *is* `h_10`, that also lands exactly on baseline, and
  depth-reading still buys nothing.

`inspect_reader` on the L4 checkpoint is the cheap first cut (query-pool mass concentrated on rung 10
⇒ collapse (B); spread over the middle rungs, as the L2 reader did ⇒ genuine (A)) — though the blocks
mix across rungs *before* the pool, so a rung-ablation at eval is the real clincher. And the cost
matters: even at parity wide@4 burns ~3.2×/token, so per-FLOP it is still a loss — to be *worth it*
depth-reading must beat, not tie. Plot `experiments/figs/wide640_layers_compare.png`; data
`experiments/figs/wide640_L4_val.csv`.

## Reproduce

```bash
# on tigerfish, GPU 2 (single GPU is fine for both probes)
cd ~/2d-Transformers-nc
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m scripts.svd_readout_probe --model-tag d10_baseline
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m scripts.frozen_probe     --model-tag d10_baseline

# depth-attention diagnostic on a reader checkpoint
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m scripts.inspect_reader --model-tag d10_reader

# d_V=640 WideReader, iso-total-FLOP (756 steps), GPUs 2,3
CUDA_VISIBLE_DEVICES=2,3 NPROC=2 nohup bash scripts/run_wide640.sh > wide640.log 2>&1 &

# full-budget WideReader (1605 steps) + compute-matched baseline (3406 steps), then both plots
CUDA_VISIBLE_DEVICES=2,3 NPROC=2 STEPS=1605 TAG=d10_wide640_full nohup bash scripts/run_wide640.sh > wide640_full.log 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 NPROC=2 nohup bash scripts/run_baseline_long.sh > baseline_long.log 2>&1 &
python scripts/plot_compute_data.py   # -> experiments/figs/{flops_compare,tokens_compare}.png

# residual-free WideReader (1605 steps): data-free init check first, then the run + plot
python -m scripts.check_residual_free
CUDA_VISIBLE_DEVICES=2,3 NPROC=2 nohup bash scripts/run_wide640_nores.sh > wide640_nores.log 2>&1 &
python scripts/plot_nores_compare.py  # -> experiments/figs/wide640_nores_compare.png

# taller V (exp 6): reader_layers=4, full budget, on Modal (4x A100); --dbs 8 to fit A100-40GB
MODAL_GPUS=4 modal run scripts/modal_train.py --action train --steps 1605 \
    --reader-layers 4 --h-residual full --dbs 8 --tag d10_wide640_L4_full
python scripts/plot_layers_compare.py  # -> experiments/figs/wide640_layers_compare.png
```

Checkpoints (tigerfish): `~/.cache/nanochat/base_checkpoints/d10_{baseline,reader,wide640,wide640_full,baseline_long,wide640_nores}/`;
exp 6 (`d10_wide640_L4_full`) lives on the Modal `nanochat-cache` volume.

# nanochat 10+2 — vertical reader: Phase A + readout-bottleneck probes

Does a small bidirectional **reader** over the depth axis (the nanochat 10+2 "vertical"
architecture) beat reading only the top-layer state, at near-zero added cost? Phase A says no.
This log records Phase A and the follow-up probes asking whether the reader's **128-dim
bottleneck** is the cause.

Related code: `nanochat/readers/vertical.py` (the reader), `scripts/inspect_reader.py`
(depth-attention diagnostic), `scripts/svd_readout_probe.py` (PCA truncation),
`scripts/frozen_probe.py` (trained 128-dim cap).

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
- **Decisive next test: the `d_V=640` retrain** (remove the cap entirely; ~1.5 hr, per-step cost
  grows ~quadratically in `d_V` while the token budget stays fixed). Optional midpoint `d_V=320`
  for monotonicity.

## Reproduce

```bash
# on tigerfish, GPU 2 (single GPU is fine for both probes)
cd ~/2d-Transformers-nc
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m scripts.svd_readout_probe --model-tag d10_baseline
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m scripts.frozen_probe     --model-tag d10_baseline

# depth-attention diagnostic on the reader checkpoint
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m scripts.inspect_reader --model-tag d10_reader
```

Checkpoints (tigerfish): `~/.cache/nanochat/base_checkpoints/d10_{baseline,reader}/model_001605.pt`.

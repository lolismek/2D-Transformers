# Plan — Gated depth-reader + per-FLOP V-slope at d10 (matched baselines)

## Context

Six experiments showed the depth-reader V (reading the residual ladder `[x0, h_1..h_L]`) never
*beats* the top-state baseline; at best a taller V *ties* it (d10, `reader_layers=4`, 0.877) while
costing ~3.2×/token. Every prior arm made V **own** the readout (no `h_L` skip, no gate). We pivot to:
**does a *gated* reader (an additive correction to `h_L`) of any height beat the plain baseline you
could have trained for the same compute?**

First experiment (this plan) — the **V-slope at d10**, with each reader matched to a trained,
equal-FLOPs baseline:
1. **Gate V** — `readout = base + g·V(ladder)`, `g` scalar init ≈0, so `g→0` is exactly baseline.
2. **Sweep reader depth at d10**: `reader_layers ∈ {2, 4, 8, 10}` (4 gated runs).
3. **Matched baselines**: train `d12, d14, d16` (reuse d10 = 0.877) → a 4-point loss-vs-FLOPs
   frontier that brackets all four reader cells. Verdict = is each reader point **below** that curve.
4. **Fix the reader FLOP undercount first** (blocker) so the frontier's x-axis is honest.

H-slope (varying backbone depth) is a **deferred extension** once this arm shows signal.

## Design decisions (resolved)

- **Gate: scalar `g ∈ R`, init `g₀ = 1e-3`** (not exact-0). With `readout = base + g·V` the V-weight
  gradient is `∝ g`; tiny `g₀` gives V gradient from step 0 (Muon normalizes → ~full-rate learning)
  while init is baseline to ~4 decimals. `g` gets its **own AdamW group** (lr≈0.2, wd=0) and is
  **logged every eval** — "did V turn on?" becomes observed.
- **Init identity: match baseline (keep backout).** `base = h_L − λ·h_mid`; gate adds `g·V`. At `g=0`
  the gated model is **exactly** the baseline.
- **Budget: ratio-12 on H+V for readers, ratio-12 on H for baselines.** A reader is a bigger model,
  so it earns proportionally more tokens; it's then compared to a *deeper* equal-compute baseline.
- **Compute: Modal A100, up to 4 GPUs**, via extended `scripts/modal_train.py`.

## Reader FLOPs & the run set (validated — `scripts/reader_flops.py`)

Corrected per-token reader cost: **`reader_fpt = 6 · (12·V·n_embd²) · (H+1)`** (each reader layer =
one backbone-layer's matmul re-run over all H+1 rungs). Validated vs three repo anchors (d10 fpt
`5.78e8`, reader-L2 params `9,830,400`, reader fpt `648.8e6` → `2.122×`).

**Reader cells** (d10, n_embd=640; gated; ratio-12 on H+V via `--target-param-data-ratio = 12·(70.12M
+ reader_matrix)/70.12M`):

| run | reader_matrix | fpt × base | tokens | total FLOPs | dbs | time |
|---|---|---|---|---|---|---|
| d10 × L2 | 9.83M | 2.12× | 0.96e9 | 1.18e18 | 16 | 0.7 h |
| d10 × L4 | 19.66M | 3.24× | 1.08e9 | 2.02e18 | 8 | 1.4 h |
| d10 × L8 | 39.32M | 5.49× | 1.31e9 | 4.16e18 | 8 | 2.3 h |
| d10 × L10 | 49.15M | 6.61× | 1.43e9 | 5.46e18 | 8 | 3.1 h |

**Baselines** (`reader=none`, ratio-12) — the matched-FLOPs frontier:

| run | total FLOPs | role | time |
|---|---|---|---|
| d10 | 0.49e18 | **reuse** existing 0.877 (low anchor) | 0 |
| d12 | 1.17e18 | ≈ exact match for d10×L2 | 0.7 h |
| d14 | 2.55e18 | brackets d10×L4 (with d12) | 1.4 h |
| d16 | 5.11e18 | brackets d10×L8; ≈ match for d10×L10 | 2.9 h |

- **Matching:** the curve through {d10, d12, d14, d16} brackets all four reader cells (1.18–5.46e18);
  d10×L2↔d12 and d10×L10↔d16 are near-exact, d10×L4/L8 interpolate between adjacent rungs. No extrap.
- **Total to run ≈ 21.7e18 FLOPs ≈ ~12 h** on one 4×A100 box (~50 A100-hr, @~40% MFU from the exp-6
  anchor; ±~50%). Under a day.
- **d10 reuse caveat:** the 0.877 was trained on tigerfish, not Modal. For a clean same-environment
  frontier, optionally retrain d10 baseline on Modal (+0.3 h, trivial); else trust 0.877 (same
  data/tokenizer/seed/numerics, hash-verified).

## Change 1 — additive scalar gate

Files: `nanochat/readers/base.py`, `wide.py`, `vertical.py`, `nanochat/gpt.py`, `scripts/base_train.py`.

- **`base.py` (BaseReader):** `_init_gate(self, config)` reads `config.reader_gate ∈
  {none,scalar,channel}` → sets `self.gate` to `None` / `Parameter(zeros(1))` /
  `Parameter(zeros(n_embd))`; `combine(self, base, r)` → **`r if getattr(self,'gate',None) is None
  else base + self.gate.to(r.dtype)*r`** (ungated ⇒ reader OWNS the readout, i.e. returns `r`, the
  prior behavior — NOT `base`); `gate_parameters()` → `[]` or `[self.gate]`. Do **not** declare a
  class attr `gate = None`: nn.Module's `register_parameter` rejects assigning a Parameter to a name
  that already exists. All access is `getattr(self,"gate",None)`-guarded.
- **`wide.py` / `vertical.py`:** call `self._init_gate(config)` in `__init__`; in `init_weights()`
  add `if self.gate is not None: torch.nn.init.constant_(self.gate, 1e-3)` (mandatory — re-init'd
  after `to_empty`).
- **`gpt.py` forward (~535-542):** replace the reader branch with
  ```
  r = self.reader.readout(ladder)
  base = ladder[-1]
  if x_backout is not None and not residual_free:
      base = base - self.backout_lambda.to(base.dtype) * x_backout
  x = self.reader.combine(base, r)        # base + g·r ; g=0 ⇒ exactly baseline
  ```
- **`gpt.py` setup_optimizer (430-432, 451-465):** fire the backout exclusion **only for ungated
  readers** (`reader_gate == "none"`) — the gated path uses backout, so `backout_lambda` keeps a grad.
  Append a gate group: `if self.reader and self.reader.gate_parameters():
  param_groups.append(dict(kind='adamw', params=self.reader.gate_parameters(), lr=0.2,
  betas=(0.8,0.95), eps=1e-10, weight_decay=0.0))`. Extend the coverage assert (gpt.py:444).
- **`gpt.py` GPTConfig (~51):** add `reader_gate: str = "none"`.
- **`base_train.py` (56-61, 147-148):** add `--reader-gate`; thread into `GPTConfig`. Log
  `gate=<reader.gate.item()>` each eval.

## Change 2 — reader FLOP accounting fix (the blocker)

File: `nanochat/gpt.py`, `estimate_flops()` (353-379). Reader matrix params are applied once per rung
(`N=n_layer+1`) but currently counted only 6× each. Fix (matches `check_wide_reader.py:58`):
```
reader_total  = sum(p.numel() for p in self.reader.parameters())          if self.reader else 0
reader_matrix = sum(w.numel() for w in self.reader.matrix_parameters())   if self.reader else 0
num_flops_per_token = 6*(nparams - nparams_exclude - reader_total) \
                    + 6 * reader_matrix * (self.config.n_layer + 1) + attn_flops
```
Now `num_flops_per_token` is honest → frontier x-axis + MFU correct. Consumers are display/MFU/
`--target-flops` (`base_train.py:267/356/428`, `chat_sft.py`) — none break.

## Change 3 — launcher (H+V budget) + per-FLOP analysis

**Launcher — extend `scripts/modal_train.py`:**
- Parameterize `train(...)`: add `depth` (hardcoded `--depth=10` at line 114), `reader="wide"`
  (emit `--reader=none` for baselines), `reader_gate="scalar"`. Thread into the flag strings (116-117).
- **H+V budget via the existing ratio knob:** readers pass
  `--target-param-data-ratio = 12·(scaling_base(H)+reader_matrix(H,V))/scaling_base(H)` (so
  `target_tokens = 12·(H+V) params`, `total_batch_size` stays auto-keyed on H — muP-correct).
  Baselines pass `--target-param-data-ratio=12`. Import `scaling_base`/`reader_matrix` from
  `scripts/reader_flops.py` (2 lines). Drop the hardcoded `--num-iterations`.
- New `sweep` entrypoint runs exactly: readers d10×{L2,L4,L8,L10} (`hv_d10_L{V}_g`, dbs per table)
  + baselines d12,d14,d16 (`hv_d{H}_base`, dbs 16). `MODAL_GPUS=4`.
- Each run appends a row to `experiments/figs/hv_sweep_results.csv`: `depth, reader_layers, reader,
  reader_gate, n_embd, num_scaling_params, num_flops_per_token (corrected), num_iterations,
  total_tokens, total_flops, final_val_bpb, min_val_bpb, final_gate, tag, seed`.

**Analysis — new `scripts/plot_hv_sweep.py`** (mirrors `plot_*.py`; uses `reader_flops.py` for each
cell's FLOPs):
1. **Per-FLOP frontier (PRIMARY).** Fit log-log power law through baseline anchors {d10,d12,d14,d16}
   (val_bpb vs total FLOPs); overlay the 4 reader points; annotate each cell's gap = `reader_bpb −
   fitted_baseline(reader_flops)`. Below the curve = a per-FLOP win.
2. **V-slope.** gap vs V (= vs reader FLOPs). Does adding reader depth buy more than a deeper plain model.
3. **Gate trajectory.** `final_gate` (+ per-eval trace) per cell — did the correction turn on.
4. **(Secondary) mechanism.** `inspect_reader` depth-attention on each reader checkpoint (which rungs
   V reads); eval-time, no extra training.

## Coherence & verification

- **Init == baseline (load-bearing):** extend `scripts/check_wide_reader.py` to build
  `reader_gate="scalar"`, assert param coverage (blocks→Muon; depth_pos/q→adamw; gate→gate group),
  assert `combine(base,r)` at `gate=0` returns `base`, and assert `estimate_flops ≈ 6·matmul·(L+1) +
  E_base`. Live check: gate-forced-to-0 reader gives logits identical to a `reader=none` model.
- **DDP gate param:** extend `scripts/check_reader_dist.py` (`reader_gate="scalar"`, `nproc=2`); gate
  always has a grad and 1-elem → shape-agnostic all-reduce → no None-grad crash.
- **FLOP self-check:** `scripts/reader_flops.py` anchors stay green.
- **Pilots:** every cell at `steps=20` first (catches OOM/dbs) before full budget.
- **Comparability:** `--window-pattern=L`, bf16, fixed seed, hash-verified data/tokenizer.

## Verification (end-to-end)

1. `python3 scripts/reader_flops.py` → anchors OK; the run set above reproduces.
2. `python -m scripts.check_wide_reader` → coverage + gate-zero-identity + FLOP asserts pass.
3. `torchrun --nproc_per_node=2 -m scripts.check_reader_dist` (gate variant) → no DDP crash.
4. `modal run scripts/modal_train.py --action train --depth 10 --reader wide --reader-gate scalar
   --reader-layers 4 --steps 20 --tag pilot` → trains, logs `gate=`, writes a CSV row; gate-forced-0
   reader vs `reader=none` give identical step-0 val bpb.
5. `modal run scripts/modal_train.py --action sweep` (MODAL_GPUS=4) → 7 runs populate the CSV (~12 h).
6. `python scripts/plot_hv_sweep.py` → frontier + V-slope + gate figures in `experiments/figs/`.

## Risks / open

- **"Exact match" is exact only for L2↔d12 and L10↔d16**; L4/L8 interpolate within their bracket
  (negligible — the power-law fit is clean over 0.49–5.11e18, all interpolation).
- **`g₀=1e-3` vs 0** is a one-line knob; gate logs pinned near 0 ⇒ a real negative, not an artifact.
- **d10 reuse parity:** retrain d10 baseline on Modal (+0.3 h) if the tigerfish 0.877 isn't trusted.
- **Deferred:** the H-slope (does the gap close with backbone depth) — add d14/d18 reader rows + d18+
  anchors once this V-slope shows the reader is worth pursuing.

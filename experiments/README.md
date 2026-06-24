# Experiments

Research log for **vertical (depth-axis) transformers** on a nanochat depth-10 backbone (the "10+2"
setup). Open-ended — no fixed thesis; each experiment probes the depth ladder from a different angle
(params / FLOPs / data / architecture). See the repo [`README.md`](../README.md) for the architecture
and the scripts that produced these artifacts.

## Main writeup
**[`nanochat_10p2_reader.md`](nanochat_10p2_reader.md)** — the full log: Phase A (baseline vs
`d_V=128` reader), the readout-bottleneck probes (PCA truncation + a trained frozen 128-dim linear
cap), the `d_V=640` **WideReader** (iso-FLOP and full-budget), the **residual-free** WideReader, and
the **taller-V** (reader-depth) sweep.

## Experiments at a glance (val bpb, depth-10)

| # | experiment | axis | result vs baseline 0.877 |
|---|---|---|---|
| 1 | `d_V=128` reader (Phase A) | params | 0.933 (**+0.056**) |
| 2 | readout-bottleneck probes (no retrain) | rank | PCA-128 +0.454 → trained-128 +0.167 → reader-128 +0.056 |
| 3 | WideReader `d_V=640`, iso-FLOP (756 steps) | FLOPs | 0.928 (**+0.051**) |
| 4 | WideReader `d_V=640`, full budget | data + FLOPs | equal-data +0.021 · equal-compute +0.053 (base 0.845 @3406) |
| 5 | residual-free WideReader (1605 steps) | architecture | 0.928 (**+0.051**; +0.030 worse than wide *with* residuals) |
| 6 | taller V — `reader_layers=4` (1605 steps) | reader depth | **0.877 (ties baseline)** — closes the entire +0.021 wide@2 gap |

**So far:** the ladder is non-degenerate to read (V spreads over the middle rungs, ignores `h_10`).
Reading it as the readout was modestly worse than the top state per *token* (+0.021 wide@2) and
clearly worse per *FLOP* (+0.053) — but that deficit is a reader-*depth* limit, not width: a taller
V (exp 6, `reader_layers=4`) closes the whole gap and **ties** baseline (0.877). It still doesn't
*beat* the top state, and at ~3.2×/token it loses per-FLOP; whether the tie is genuine ladder-reading
or V collapsing to mimic `h_10` is open. Mechanism for the negatives: nanochat's residual stream
makes every rung a partial sum, so `h_10` already holds what an offline reader would re-aggregate
(exp 5: removing the residuals makes V *worse*, confirming it).

## Artifacts
- `figs/phase_a_val_bpb.png`, `figs/phase_a_val.csv`, `figs/phase_a_train.csv` — exp 1 loss curves
  (baseline vs `d_V=128` reader), produced by `../scripts/plot_phase_a.py`.
- `svd_probe_d10_baseline.json` — exp 2 raw SVD spectrum + bpb(k) curve from the PCA readout probe.
- `figs/wide640_isoflop.png`, `figs/wide640_val.csv` — exp 3 iso-FLOP 3-way comparison vs cumulative
  compute, produced by `../scripts/plot_wide_compare.py`.
- `figs/flops_compare.png`, `figs/tokens_compare.png`, `figs/wide640_full_val.csv`,
  `figs/baseline_long_val.csv` — exp 4 full-budget WideReader@1605 vs compute-matched baseline@3406
  (FLOPs axis) and vs baseline@1605 (tokens axis), produced by `../scripts/plot_compute_data.py`.
- `figs/wide640_nores_compare.png`, `figs/wide640_nores_val.csv` — exp 5 residual-free WideReader vs
  WideReader vs baseline on the same 1605-step schedule, produced by `../scripts/plot_nores_compare.py`.
- `figs/wide640_layers_compare.png`, `figs/wide640_L4_val.csv` — exp 6 taller-V sweep: baseline vs
  WideReader L=2 vs L=4 on the same 1605-step schedule, produced by `../scripts/plot_layers_compare.py`.

# Experiments

Research log for the depth-axis **vertical reader** on a nanochat depth-10 backbone (the "10+2"
setup). See the repo [`README.md`](../README.md) for the architecture and the scripts that produced
these artifacts.

## Main writeup
**[`nanochat_10p2_reader.md`](nanochat_10p2_reader.md)** — Phase A (baseline vs `d_V=128` reader),
the readout-bottleneck probes (PCA truncation + a trained frozen 128-dim linear cap), and the
decisive `d_V=640` iso-FLOP **WideReader** test.

## Headline (val bpb, depth-10)

Full-width reader (`d_V=640`) tested two ways against the stock top-state baseline:

| comparison | baseline | WideReader `d_V=640` | Δ |
|---|---|---|---|
| **equal data** (841M tok, 1605 steps) | **0.877** | 0.898 | +0.021 |
| **equal compute** (1.03e18 FLOPs) | **0.845** (3406 steps) | 0.898 (1605 steps) | +0.053 |

Depth-reading-as-readout doesn't beat top-state reading: only modestly worse per *token* (+0.021,
once trained to the full budget — the earlier 756-step iso-FLOP test was token-starved and overstated
it), and clearly worse per *FLOP* (+0.053, since the reader's 2.12× overhead is better spent on 2×
more tokens). The 128-dim bottleneck is exonerated — `h_10` already carries what the ladder offers.
(The bottlenecked `d_V=128` reader was +0.056 at equal data.)

## Artifacts
- `figs/phase_a_val_bpb.png`, `figs/phase_a_val.csv`, `figs/phase_a_train.csv` — Phase A loss curves
  (baseline vs `d_V=128` reader), produced by `../scripts/plot_phase_a.py`.
- `figs/wide640_isoflop.png`, `figs/wide640_val.csv` — iso-FLOP 3-way comparison vs cumulative
  compute, produced by `../scripts/plot_wide_compare.py`.
- `figs/flops_compare.png`, `figs/tokens_compare.png`, `figs/wide640_full_val.csv`,
  `figs/baseline_long_val.csv` — full-budget WideReader@1605 vs compute-matched baseline@3406 (FLOPs
  axis) and vs the old baseline@1605 (tokens axis), produced by `../scripts/plot_compute_data.py`.
- `svd_probe_d10_baseline.json` — raw SVD spectrum + bpb(k) curve from the PCA readout probe.

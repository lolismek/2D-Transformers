# Experiments

Research log for the depth-axis **vertical reader** on a nanochat depth-10 backbone (the "10+2"
setup). See the repo [`README.md`](../README.md) for the architecture and the scripts that produced
these artifacts.

## Main writeup
**[`nanochat_10p2_reader.md`](nanochat_10p2_reader.md)** — Phase A (baseline vs `d_V=128` reader),
the readout-bottleneck probes (PCA truncation + a trained frozen 128-dim linear cap), and the
decisive `d_V=640` iso-FLOP **WideReader** test.

## Headline (val bpb, depth-10)
| readout | val bpb | training compute |
|---|---|---|
| baseline (top-state `h_10`) | **0.877** | 1.00× |
| reader `d_V=128` | 0.933 | 1.06× |
| WideReader `d_V=640` (iso-FLOP) | 0.928 | 1.00× |

Depth-reading-as-readout doesn't beat top-state reading at equal compute; widening the reader to
full width doesn't rescue it ⇒ the 128-dim bottleneck is largely exonerated as the cause.

## Artifacts
- `figs/phase_a_val_bpb.png`, `figs/phase_a_val.csv`, `figs/phase_a_train.csv` — Phase A loss curves
  (baseline vs `d_V=128` reader), produced by `../scripts/plot_phase_a.py`.
- `figs/wide640_isoflop.png`, `figs/wide640_val.csv` — iso-FLOP 3-way comparison vs cumulative
  compute, produced by `../scripts/plot_wide_compare.py`.
- `svd_probe_d10_baseline.json` — raw SVD spectrum + bpb(k) curve from the PCA readout probe.

"""Two comparisons for the full-budget WideReader (d_V=640, 1605 steps) vs the baseline.

  (1) flops_compare.png  — wide@1605 vs the COMPUTE-MATCHED baseline@3406, x-axis = training FLOPs.
      Both consume the same total fwd+bwd FLOPs (iso-FLOP), so the endpoint asks:
      at EQUAL COMPUTE, does the depth-reader's readout beat plain top-state reading?

  (2) tokens_compare.png — wide@1605 vs the OLD baseline@1605 (Phase A), x-axis = tokens consumed.
      Both consume 841M tokens on the same 1605-step schedule (anneal-matched), so the endpoint asks:
      at EQUAL DATA, is the reader's readout better PER TOKEN? (the capability question)

FLOP accounting (fwd+bwd / token), matching scripts/run_wide640.sh:
  baseline    F_BASE = 5.78e8        (nanochat's own estimate for reader=none; accurate, no reader)
  WideReader  F_WIDE = F_BASE + 6 * 9,838,080 params * 11 rungs = F_BASE + 6.493e8 = 1.227e9 = 2.122x
  estimate_flops() undercounts the reader ~11x (blind to the rung axis), so F_WIDE is set by hand
  here, consistent with the iso-FLOP step budget 1605*2.122 = 3406.
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "experiments", "figs")

TOK_PER_STEP = 524288          # total_batch_size (= 2^19); identical for baseline & wide
F_BASE = 5.78e8                # baseline fwd+bwd FLOPs / token
F_WIDE = F_BASE + 6.4931e8     # + reader (6 * 9,838,080 * 11 rungs) = 1.227e9 = 2.122x


def load2(name):
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for row in csv.DictReader(f):
            out.append((int(row["step"]), float(row["val_bpb"])))
    return out


def load_phase_a_baseline():
    p = os.path.join(FIG, "phase_a_val.csv")
    out = []
    with open(p) as f:
        for row in csv.DictReader(f):
            if row["baseline_val_bpb"]:
                out.append((int(row["step"]), float(row["baseline_val_bpb"])))
    return out


def tail_zoom(ax, xs_ys, ylim=(0.86, 1.00)):
    axin = ax.inset_axes([0.46, 0.40, 0.50, 0.50])
    xmax = max(x[-1] for x, _ in xs_ys)
    for x, y in xs_ys:
        axin.plot(x, y, "-o", ms=3, lw=1.6,
                  color=ax.lines[xs_ys.index((x, y))].get_color())
    axin.set_xlim(left=0.45 * xmax)
    axin.set_ylim(*ylim)
    axin.grid(True, alpha=0.25)
    axin.tick_params(labelsize=8)
    axin.set_title("tail (zoom)", fontsize=8.5)
    ax.indicate_inset_zoom(axin, edgecolor="0.5", alpha=0.5)


wide = load2("wide640_full_val.csv")          # wide @ 1605
newbase = load2("baseline_long_val.csv")      # compute-matched baseline @ 3406
oldbase = load_phase_a_baseline()             # old baseline @ 1605

# ---- Plot 1: FLOPs axis — wide vs compute-matched baseline ----
if wide and newbase:
    fig, ax = plt.subplots(figsize=(9.2, 5.7), dpi=150)
    bx = [s * TOK_PER_STEP * F_BASE for s, _ in newbase]
    by = [b for _, b in newbase]
    wx = [s * TOK_PER_STEP * F_WIDE for s, _ in wide]
    wy = [b for _, b in wide]
    ax.plot(bx, by, "-o", color="#1f77b4", ms=3.5, lw=1.8,
            label=f"baseline (top-state $h_{{10}}$), {newbase[-1][0]} steps — {newbase[-1][1]:.3f}")
    ax.plot(wx, wy, "-o", color="#d62728", ms=3.5, lw=1.8,
            label=f"WideReader $d_V$=640, 1605 steps — {wide[-1][1]:.3f}")
    ax.set_xlabel("training compute  (fwd+bwd FLOPs)")
    ax.set_ylabel("validation bpb")
    ax.set_title("nanochat 10+2 — iso-FLOP: WideReader vs compute-matched baseline")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left")
    tail_zoom(ax, [(bx, by), (wx, wy)])
    fig.tight_layout()
    out = os.path.join(FIG, "flops_compare.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}")
    print(f"  @iso-FLOP endpoint: baseline {newbase[-1][1]:.3f} | wide {wide[-1][1]:.3f} "
          f"(delta {wide[-1][1] - newbase[-1][1]:+.3f})")
else:
    print("skip flops_compare (need wide640_full_val.csv + baseline_long_val.csv)")

# ---- Plot 2: tokens axis — wide vs old baseline at equal data ----
if wide and oldbase:
    fig, ax = plt.subplots(figsize=(9.2, 5.7), dpi=150)
    bx = [s * TOK_PER_STEP / 1e9 for s, _ in oldbase]
    by = [b for _, b in oldbase]
    wx = [s * TOK_PER_STEP / 1e9 for s, _ in wide]
    wy = [b for _, b in wide]
    ax.plot(bx, by, "-o", color="#1f77b4", ms=3.5, lw=1.8,
            label=f"baseline (top-state $h_{{10}}$) — {oldbase[-1][1]:.3f}")
    ax.plot(wx, wy, "-o", color="#d62728", ms=3.5, lw=1.8,
            label=f"WideReader $d_V$=640 — {wide[-1][1]:.3f}")
    ax.set_xlabel("data consumed  (billion tokens)")
    ax.set_ylabel("validation bpb")
    ax.set_title("nanochat 10+2 — equal data: WideReader vs baseline (same 1605-step schedule)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left")
    tail_zoom(ax, [(bx, by), (wx, wy)])
    fig.tight_layout()
    out = os.path.join(FIG, "tokens_compare.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}")
    print(f"  @equal-tokens endpoint (841M): baseline {oldbase[-1][1]:.3f} | wide {wide[-1][1]:.3f} "
          f"(delta {wide[-1][1] - oldbase[-1][1]:+.3f})")
else:
    print("skip tokens_compare (need wide640_full_val.csv + phase_a_val.csv)")

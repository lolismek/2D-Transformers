"""3-way iso-FLOP comparison: baseline vs d_V=128 reader vs d_V=640 WideReader.

x-axis = cumulative training FLOPs as a fraction of the baseline's TOTAL, so the runs are compared
at equal compute. WideReader costs 2.122x/token, so it reaches 1.0 (baseline's budget) in 756 steps;
baseline reaches 1.0 in 1605 steps; the d_V=128 reader ran 1605 steps so it ends slightly past 1.0.

Reads baseline+reader from experiments/figs/phase_a_val.csv; the WideReader trace is embedded
(from wide640.log) and also written to experiments/figs/wide640_val.csv.
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "experiments", "figs")

base, reader = [], []
with open(os.path.join(FIG, "phase_a_val.csv")) as f:
    for row in csv.DictReader(f):
        s = int(row["step"])
        if row["baseline_val_bpb"]:
            base.append((s, float(row["baseline_val_bpb"])))
        if row["reader_val_bpb"]:
            reader.append((s, float(row["reader_val_bpb"])))

# d_V=640 WideReader (iso-FLOP, 756 steps), from wide640.log
wide = [(0, 3.154605), (100, 1.417282), (200, 1.142844), (300, 1.068976), (400, 1.023801),
        (500, 0.981679), (600, 0.953708), (700, 0.934679), (756, 0.928049)]
with open(os.path.join(FIG, "wide640_val.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step", "val_bpb"])
    w.writerows(wide)

N_BASE = 1605
MULT = {"base": 1.0, "reader": 1.055, "wide": 2.122}   # true fwd+bwd FLOPs/token vs baseline


def xy(data, mult):
    return [(s / N_BASE) * mult for s, _ in data], [b for _, b in data]


fig, ax = plt.subplots(figsize=(9.4, 5.7), dpi=150)
series = [
    (base,   MULT["base"],   "#1f77b4", f"baseline (top-state $h_{{10}}$ readout) — {base[-1][1]:.3f}"),
    (reader, MULT["reader"], "#7f7f7f", f"reader $d_V$=128 (Phase A) — {reader[-1][1]:.3f}"),
    (wide,   MULT["wide"],   "#d62728", f"WideReader $d_V$=640, iso-FLOP — {wide[-1][1]:.3f}"),
]
for data, mult, color, label in series:
    x, y = xy(data, mult)
    ax.plot(x, y, "-o", color=color, ms=4, lw=1.9, label=label)
ax.axvline(1.0, color="0.5", ls="--", lw=1)
ax.text(1.005, 2.7, "baseline's\ncompute budget", fontsize=8, color="0.4", va="top")
ax.set_xlabel("training compute  (× baseline's total FLOPs)")
ax.set_ylabel("validation bpb")
ax.set_title("nanochat 10+2 — iso-FLOP: full-width reader still loses to plain top-state readout")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper right")

axin = ax.inset_axes([0.45, 0.34, 0.51, 0.50])
for data, mult, color in [(base, 1.0, "#1f77b4"), (reader, 1.055, "#7f7f7f"), (wide, 2.122, "#d62728")]:
    x, y = xy(data, mult)
    axin.plot(x, y, "-o", color=color, ms=4, lw=1.9)
axin.set_xlim(0.44, 1.10)
axin.set_ylim(0.86, 1.05)
axin.axvline(1.0, color="0.5", ls="--", lw=1)
axin.grid(True, alpha=0.25)
axin.tick_params(labelsize=8)
axin.set_title("tail (zoom)", fontsize=8.5)
ax.indicate_inset_zoom(axin, edgecolor="0.5", alpha=0.5)

fig.tight_layout()
out = os.path.join(FIG, "wide640_isoflop.png")
fig.savefig(out, bbox_inches="tight")
print(f"saved -> {out}")
print(f"@ 1.0F (equal compute): baseline {base[-1][1]:.3f}  |  WideReader {wide[-1][1]:.3f}  "
      f"(+{wide[-1][1]-base[-1][1]:.3f})  |  reader@1.055F {reader[-1][1]:.3f}")

"""Taller-V sweep: does adding reader DEPTH (more bidirectional blocks over the ladder) help?

All three runs are the SAME 1605-step schedule (841M tokens, fully LR-annealed at 1605), so this is a
clean per-token / equal-DATA comparison -- only the reader's height (reader_layers) differs.

  * baseline                  -- stock nanochat d10, top-state h_10 readout   (phase_a_val.csv)
  * WideReader d_V=640, L=2   -- full-width depth-reader, 2 blocks            (wide640_full_val.csv)
  * WideReader d_V=640, L=4   -- same reader, 4 blocks (taller V)             (wide640_L4_val.csv)

Headline: 2->4 reader layers closes the entire +0.021 wide@2 gap; wide@4 lands ON baseline (0.877).
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "experiments", "figs")


def load(name, col="val_bpb"):
    out = []
    with open(os.path.join(FIG, name)) as f:
        for row in csv.DictReader(f):
            if row.get(col):
                out.append((int(row["step"]), float(row[col])))
    return out


# baseline lives in phase_a_val.csv under a differently-named column
base = []
with open(os.path.join(FIG, "phase_a_val.csv")) as f:
    for row in csv.DictReader(f):
        if row["baseline_val_bpb"]:
            base.append((int(row["step"]), float(row["baseline_val_bpb"])))

wide2 = load("wide640_full_val.csv")
wide4 = load("wide640_L4_val.csv")

series = [
    (base,  "#1f77b4", f"baseline (top-state $h_{{10}}$) — {base[-1][1]:.3f}"),
    (wide2, "#d62728", f"WideReader $d_V$=640, L=2 — {wide2[-1][1]:.3f}"),
    (wide4, "#2ca02c", f"WideReader $d_V$=640, L=4 (taller V) — {wide4[-1][1]:.3f}"),
]

fig, ax = plt.subplots(figsize=(9.4, 5.7), dpi=150)
for data, color, label in series:
    x, y = [s for s, _ in data], [b for _, b in data]
    ax.plot(x, y, "-o", color=color, ms=4, lw=1.9, label=label)
ax.set_xlabel("training step  (same 1605-step schedule = equal data)")
ax.set_ylabel("validation bpb")
ax.set_title("nanochat 10+2 — a taller depth-reader (2→4 layers) closes the gap to baseline")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper right")
ax.set_ylim(0.84, 1.50)

# tail zoom: where the runs separate / converge
axin = ax.inset_axes([0.46, 0.40, 0.50, 0.50])
for data, color, _ in series:
    x, y = [s for s, _ in data], [b for _, b in data]
    axin.plot(x, y, "-o", color=color, ms=4, lw=1.9)
axin.set_xlim(1100, 1650)
axin.set_ylim(0.873, 0.945)
axin.grid(True, alpha=0.25)
axin.tick_params(labelsize=8)
axin.set_title("tail (zoom)", fontsize=8.5)
ax.indicate_inset_zoom(axin, edgecolor="0.5", alpha=0.5)

fig.tight_layout()
out = os.path.join(FIG, "wide640_layers_compare.png")
fig.savefig(out, bbox_inches="tight")
print(f"saved -> {out}")
print(f"@1605  baseline {base[-1][1]:.4f}  |  wide@2 {wide2[-1][1]:.4f} (+{wide2[-1][1]-base[-1][1]:.4f})  "
      f"|  wide@4 {wide4[-1][1]:.4f} ({wide4[-1][1]-base[-1][1]:+.4f} vs base, "
      f"{wide4[-1][1]-wide2[-1][1]:+.4f} vs wide@2)")

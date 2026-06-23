"""Residual-free WideReader vs WideReader vs baseline, all on the SAME 1605-step schedule.

The thesis under test: H's residual connections "steal the job" from the offline depth-reader V, so
removing them (the in-block attn+mlp skips and the x0 injection; V keeps its own residuals) should
free V to do more. All three runs are scheduled-for-1605 and fully LR-annealed at 1605, so this is a
clean per-token (equal-DATA) comparison — only the architecture differs.

  * baseline           — stock nanochat d10, top-state h_10 readout      (phase_a_val.csv)
  * WideReader d_V=640 — full-width depth-reader, H keeps residuals       (wide640_full_val.csv)
  * WideReader d_V=640 residual-free — same reader, H residuals removed   (wide640_nores_val.csv)
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "experiments", "figs")


def load(name, step="step", col="val_bpb"):
    out = []
    with open(os.path.join(FIG, name)) as f:
        for row in csv.DictReader(f):
            if row.get(col):
                out.append((int(row[step]), float(row[col])))
    return out


# baseline lives in phase_a_val.csv under a differently-named column
base = []
with open(os.path.join(FIG, "phase_a_val.csv")) as f:
    for row in csv.DictReader(f):
        if row["baseline_val_bpb"]:
            base.append((int(row["step"]), float(row["baseline_val_bpb"])))

wide = load("wide640_full_val.csv")
nores = load("wide640_nores_val.csv")

series = [
    (base,  "#1f77b4", f"baseline (top-state $h_{{10}}$) — {base[-1][1]:.3f}"),
    (wide,  "#d62728", f"WideReader $d_V$=640, H keeps residuals — {wide[-1][1]:.3f}"),
    (nores, "#2ca02c", f"WideReader $d_V$=640, H residual-free — {nores[-1][1]:.3f}"),
]

fig, ax = plt.subplots(figsize=(9.4, 5.7), dpi=150)
for data, color, label in series:
    x, y = [s for s, _ in data], [b for _, b in data]
    ax.plot(x, y, "-o", color=color, ms=4, lw=1.9, label=label)
ax.set_xlabel("training step  (same 1605-step schedule = equal data)")
ax.set_ylabel("validation bpb")
ax.set_title("nanochat 10+2 — removing H's residuals makes the depth-reader WORSE, not better")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper right")

# tail zoom: where the runs separate
axin = ax.inset_axes([0.46, 0.40, 0.50, 0.50])
for data, color, _ in series:
    x, y = [s for s, _ in data], [b for _, b in data]
    axin.plot(x, y, "-o", color=color, ms=4, lw=1.9)
axin.set_xlim(900, 1650)
axin.set_ylim(0.87, 1.01)
axin.grid(True, alpha=0.25)
axin.tick_params(labelsize=8)
axin.set_title("tail (zoom)", fontsize=8.5)
ax.indicate_inset_zoom(axin, edgecolor="0.5", alpha=0.5)

fig.tight_layout()
out = os.path.join(FIG, "wide640_nores_compare.png")
fig.savefig(out, bbox_inches="tight")
print(f"saved -> {out}")
print(f"@1605  baseline {base[-1][1]:.3f}  |  WideReader {wide[-1][1]:.3f} (+{wide[-1][1]-base[-1][1]:.3f})  "
      f"|  residual-free {nores[-1][1]:.3f} (+{nores[-1][1]-base[-1][1]:.3f} vs base, "
      f"+{nores[-1][1]-wide[-1][1]:.3f} vs WideReader)")

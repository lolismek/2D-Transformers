"""Plot Phase A loss curves: d10 baseline vs vertical reader, in a single graph.

Source = the stdout training logs on tigerfish (no wandb was used):
  ~/2d-Transformers-nc/phase_a.log      (reader=none,     tag=d10_baseline)
  ~/2d-Transformers-nc/reader_rerun.log (reader=vertical, tag=d10_reader)

Series were extracted with (val every 100 steps, train every step):
  grep "Validation bpb" LOG | awk '{print $2","$NF}'                                  # step,val_bpb
  grep -E "^step [0-9]+/[0-9]+" LOG | awk '{split($2,a,"/"); print a[1]","$6}'          # step,train_loss

This script parses the two transfer files (/tmp/val_series.txt, /tmp/train_series.txt),
writes tidy CSVs to experiments/figs/, and renders experiments/figs/phase_a_val_bpb.png.
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "experiments", "figs")
os.makedirs(FIGDIR, exist_ok=True)

BASE = "#1f77b4"   # baseline (blue)
READ = "#d62728"   # reader   (red)


def parse_blocks(path, markers):
    """Return {key: [(step:int, val:float), ...]} split by the given marker substrings."""
    out = {k: [] for k in markers}
    cur = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            hit = next((k for k, m in markers.items() if m in line), None)
            if hit is not None:
                cur = hit
                continue
            if cur is None or line.startswith("step,") or "ROWS=" in line or line.startswith("---"):
                continue
            if "," not in line:
                continue
            s, v = line.split(",")[:2]
            try:
                out[cur].append((int(s), float(v)))
            except ValueError:
                continue
    return out


val = parse_blocks("/tmp/val_series.txt",
                   {"baseline": "VAL baseline", "reader": "VAL reader"})
train = parse_blocks("/tmp/train_series.txt",
                     {"baseline": "===BASELINE===", "reader": "===READER==="})

for name, d in (("val", val), ("train", train)):
    for k in ("baseline", "reader"):
        assert d[k], f"empty {name}/{k} -- check /tmp series files"
    print(f"{name}: baseline={len(d['baseline'])} pts, reader={len(d['reader'])} pts")

# tidy CSVs (wide; the two runs share a step grid in each series)
def dump_csv(path, series, ycol):
    bs, rs = dict(series["baseline"]), dict(series["reader"])
    steps = sorted(set(bs) | set(rs))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", f"baseline_{ycol}", f"reader_{ycol}"])
        for s in steps:
            w.writerow([s, bs.get(s, ""), rs.get(s, "")])
    print(f"wrote {path}")

dump_csv(os.path.join(FIGDIR, "phase_a_val.csv"), val, "val_bpb")
dump_csv(os.path.join(FIGDIR, "phase_a_train.csv"), train, "train_loss")

bx, by = zip(*val["baseline"])
rx, ry = zip(*val["reader"])
b_final, r_final = by[-1], ry[-1]
gap = r_final - b_final

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=150)
ax.plot(bx, by, "-o", color=BASE, ms=4, lw=1.8,
        label=f"baseline (top-state $h_{{10}}$ readout) — {b_final:.3f}")
ax.plot(rx, ry, "-o", color=READ, ms=4, lw=1.8,
        label=f"vertical reader ($d_V$=128, owns readout) — {r_final:.3f}  (+{gap:.3f})")
ax.set_xlabel("training step  (1605 = full d10 budget, ~841M tokens)")
ax.set_ylabel("validation bpb")
ax.set_title("nanochat 10+2 — Phase A validation loss: baseline vs vertical reader")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper right", framealpha=0.95)
ax.annotate(f"identical at init: {by[0]:.3f} bpb\n(same seed / data / budget)",
            xy=(bx[0], by[0]), xytext=(120, 2.55),
            fontsize=8.5, color="0.35",
            arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))

# zoom inset on the separating tail
axin = ax.inset_axes([0.46, 0.34, 0.50, 0.50])
axin.plot(bx, by, "-o", color=BASE, ms=4, lw=1.8)
axin.plot(rx, ry, "-o", color=READ, ms=4, lw=1.8)
axin.set_xlim(380, 1660)
axin.set_ylim(0.855, 1.12)
axin.grid(True, alpha=0.25)
axin.tick_params(labelsize=8)
axin.set_title("tail (zoom)", fontsize=8.5)
axin.axhline(b_final, color=BASE, ls=":", lw=1, alpha=0.7)
axin.annotate(f"+{gap:.4f} bpb\n(reader worse)", xy=(1605, (b_final + r_final) / 2),
              xytext=(1080, 0.99), fontsize=8.2, color="0.2",
              arrowprops=dict(arrowstyle="-[, widthB=1.4", color="0.4", lw=0.9))
ax.indicate_inset_zoom(axin, edgecolor="0.5", alpha=0.5)

fig.tight_layout()
out = os.path.join(FIGDIR, "phase_a_val_bpb.png")
fig.savefig(out, bbox_inches="tight")
print(f"\nsaved -> {out}")
print(f"baseline {b_final:.6f} | reader {r_final:.6f} | gap +{gap:.6f}")

"""Analysis for the gated depth-reader V-slope at d10 (see experiments/gated_reader_plan.md).

Reads experiments/figs/hv_sweep_results.csv (one row per run, written by scripts/modal_train.py) and
produces three views, the first of which is the verdict:

  1. PER-FLOP FRONTIER (primary).  Fit a log-log power law through the plain-baseline anchors
     {d10, d12, d14, d16} (val_bpb vs total training FLOPs) and overlay the four gated reader cells
     d10x{L2,L4,L8,L10}. Each reader's gap = reader_bpb - fitted_baseline(reader_flops): a point
     BELOW the curve (gap < 0) is a per-compute win -- the reader beats the deeper plain model you
     could have trained for the same FLOPs. This is the question the whole experiment exists to answer.

  2. V-SLOPE.  gap vs reader depth V. Does adding reader layers buy more than a deeper plain trunk?

  3. GATE TRAJECTORY.  final gate magnitude per cell -- did the additive correction actually turn on
     (|g| well above its 1e-3 init) or stay ~off (a real negative result, not an artifact)?

The d10 baseline (0.877, reused from the tigerfish Phase-A run) is injected as the low-FLOP anchor
unless a depth-10 reader=none row is present in the CSV (i.e. you retrained it on Modal).

    python scripts/plot_hv_sweep.py
"""
import os
import csv
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.reader_flops import base_point

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "experiments", "figs")
CSV = os.path.join(FIG, "hv_sweep_results.csv")

D10_REUSE_BPB = 0.877            # tigerfish Phase-A baseline@1605 (same data/tokenizer/seed; hash-verified)


def load_rows():
    if not os.path.exists(CSV):
        raise SystemExit(f"no results yet at {CSV} (run: modal run scripts/modal_train.py --action sweep)")
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            def fnum(k):
                v = r.get(k, "")
                return float(v) if v not in ("", "None", None) else None
            rows.append(dict(
                tag=r["tag"], depth=int(r["depth"]), reader=r["reader"],
                reader_layers=int(r["reader_layers"]), reader_gate=r.get("reader_gate", "none"),
                total_flops=fnum("total_flops"), final_val_bpb=fnum("final_val_bpb"),
                min_val_bpb=fnum("min_val_bpb"), final_gate=fnum("final_gate"),
            ))
    return rows


def loglog_fit(flops, bpb):
    """Least-squares line in log-log space: returns predict(C) = exp(c) * C**m (no numpy needed)."""
    xs = [math.log(c) for c in flops]
    ys = [math.log(b) for b in bpb]
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    m = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    c = (sy - m * sx) / n
    return lambda C: math.exp(c) * C ** m


def main():
    rows = load_rows()
    baselines = [r for r in rows if r["reader"] == "none" and r["final_val_bpb"] is not None]
    readers = [r for r in rows if r["reader"] != "none" and r["final_val_bpb"] is not None]

    # Inject the reused d10 baseline as the low-FLOP anchor unless one was retrained on Modal.
    if not any(b["depth"] == 10 for b in baselines):
        baselines.append(dict(tag="d10_reuse", depth=10, reader="none", reader_layers=0,
                              total_flops=base_point(10)["flops"], final_val_bpb=D10_REUSE_BPB,
                              min_val_bpb=D10_REUSE_BPB, final_gate=None))
    baselines.sort(key=lambda r: r["total_flops"])
    readers.sort(key=lambda r: r["reader_layers"])

    have_fit = len(baselines) >= 2
    predict = loglog_fit([b["total_flops"] for b in baselines],
                         [b["final_val_bpb"] for b in baselines]) if have_fit else None

    # Verdict table (stdout) + per-reader gap vs the fitted baseline frontier.
    print(f"{'tag':>14} {'V':>3} {'totFLOPs':>10} {'bpb':>7} {'fit_base':>8} {'gap':>8} {'gate':>8}  verdict")
    for r in readers:
        fb = predict(r["total_flops"]) if predict else float("nan")
        gap = r["final_val_bpb"] - fb
        verdict = "WIN (below)" if gap < 0 else "loss (above)"
        g = "-" if r["final_gate"] is None else f"{r['final_gate']:+.4f}"
        print(f"{r['tag']:>14} {r['reader_layers']:>3} {r['total_flops']:>10.2e} "
              f"{r['final_val_bpb']:>7.3f} {fb:>8.3f} {gap:>+8.3f} {g:>8}  {verdict}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), dpi=150)

    # ---- Panel 1: per-FLOP frontier (PRIMARY) ----
    ax = axes[0]
    bx = [b["total_flops"] for b in baselines]
    by = [b["final_val_bpb"] for b in baselines]
    ax.plot(bx, by, "-o", color="#1f77b4", ms=6, lw=1.8, label="plain baseline frontier")
    for b in baselines:
        ax.annotate(f"d{b['depth']}", (b["total_flops"], b["final_val_bpb"]),
                    textcoords="offset points", xytext=(0, 8), fontsize=8, color="#1f77b4")
    if predict:
        lo, hi = min(bx), max(bx + [r["total_flops"] for r in readers] + [max(bx)])
        cs = [lo * (hi / lo) ** (i / 60) for i in range(61)]
        ax.plot(cs, [predict(c) for c in cs], "--", color="#1f77b4", lw=1.0, alpha=0.6,
                label="baseline power-law fit")
    if readers:
        rx = [r["total_flops"] for r in readers]
        ry = [r["final_val_bpb"] for r in readers]
        ax.plot(rx, ry, "o", color="#d62728", ms=8, label="gated reader d10xLV")
        for r in readers:
            ax.annotate(f"L{r['reader_layers']}", (r["total_flops"], r["final_val_bpb"]),
                        textcoords="offset points", xytext=(6, -10), fontsize=8.5, color="#d62728")
    ax.set_xscale("log")
    ax.set_xlabel("total training compute  (fwd+bwd FLOPs)")
    ax.set_ylabel("validation bpb")
    ax.set_title("Per-FLOP frontier (PRIMARY): reader below the line = a per-compute win")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    # ---- Panel 2: V-slope (gap vs reader depth) ----
    ax = axes[1]
    if readers and predict:
        vs = [r["reader_layers"] for r in readers]
        gaps = [r["final_val_bpb"] - predict(r["total_flops"]) for r in readers]
        ax.plot(vs, gaps, "-o", color="#d62728", ms=7, lw=1.8)
        for v, g in zip(vs, gaps):
            ax.annotate(f"{g:+.3f}", (v, g), textcoords="offset points", xytext=(0, 8), fontsize=8.5)
    ax.axhline(0, color="#1f77b4", lw=1.3, ls="--", label="baseline frontier (gap=0)")
    ax.set_xlabel("reader depth  V (reader_layers)")
    ax.set_ylabel("gap = reader_bpb - fitted_baseline(FLOPs)")
    ax.set_title("V-slope: does deeper V buy more than a deeper plain trunk?")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    # ---- Panel 3: gate trajectory ----
    ax = axes[2]
    gated = [r for r in readers if r["final_gate"] is not None]
    if gated:
        vs = [r["reader_layers"] for r in gated]
        gates = [abs(r["final_gate"]) for r in gated]
        ax.bar([str(v) for v in vs], gates, color="#2ca02c", alpha=0.8)
        for i, g in enumerate(gates):
            ax.annotate(f"{g:.3f}", (i, g), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8.5)
    ax.axhline(1e-3, color="0.4", lw=1.0, ls=":", label="init |g|=1e-3")
    ax.set_xlabel("reader depth  V (reader_layers)")
    ax.set_ylabel("final |gate|")
    ax.set_title("Gate trajectory: did the correction turn on?")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    out = os.path.join(FIG, "hv_sweep.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()

"""Mechanical (no-data, CPU) check of WideReader: shapes, optimizer param coverage,
param count, and the exact iso-total-FLOP step budget vs the d10 baseline.

Constructs at BOTH depth 10 (the real model, n_embd=640) AND depth 12 (the throwaway reference
model base_train builds for scaling-law sizing, n_embd=768) -- the latter is what tripped the
fixed-head-count assert, so it is part of the check now.

    python -m scripts.check_wide_reader
"""
from types import SimpleNamespace
import torch

from nanochat.readers import build_reader


def build_and_check(depth, n_embd):
    cfg = SimpleNamespace(
        reader="wide", n_embd=n_embd, n_layer=depth,
        reader_dim=128, reader_layers=2, reader_heads=5, reader_mlp_mult=4,  # reader_dim/heads ignored by wide
    )
    torch.manual_seed(0)
    reader = build_reader(cfg)
    reader.init_weights()

    # forward shapes
    B, T, D = 2, 4, n_embd
    ladder = [torch.randn(B, T, D) for _ in range(depth + 1)]
    out = reader.readout(ladder)
    w = reader.attn_weights()
    assert out.shape == (B, T, D), out.shape
    assert w.shape == (B, T, depth + 1), w.shape
    assert torch.allclose(w.sum(-1), torch.ones(B, T), atol=1e-5), "pool weights must sum to 1"

    # optimizer param coverage (mirrors the gpt.py setup_optimizer assert)
    all_p, mat_p, adw_p = list(reader.parameters()), reader.matrix_parameters(), reader.adamw_parameters()
    assert len(mat_p) + len(adw_p) == len(all_p), (len(mat_p), len(adw_p), len(all_p))
    assert {id(p) for p in all_p} == {id(p) for p in mat_p} | {id(p) for p in adw_p}, "uncovered param"
    assert len({id(p) for p in mat_p} & {id(p) for p in adw_p}) == 0, "matrix/adamw overlap"
    print(f"d{depth:>2} n_embd={n_embd}: dim={reader.dim} heads={reader.n_heads} "
          f"layers={len(reader.blocks)} | readout{tuple(out.shape)} attn{tuple(w.shape)} | "
          f"coverage {len(mat_p)}+{len(adw_p)}={len(all_p)} OK")
    return reader


# depth 10 = real model; depth 12 = base_train's scaling-law reference (must also construct)
r10 = build_and_check(10, 640)
build_and_check(12, 768)

# --- counts + iso-FLOP step budget (for the real d10 model) ------------------
all_p = list(r10.parameters())
matmul = sum(p.numel() for p in r10.matrix_parameters())   # block weights, applied to all 11 rungs/token
total = sum(p.numel() for p in all_p)
print(f"\nd10 reader params: total={total:,}  matmul(blocks)={matmul:,}  "
      f"adamw(depth_pos+q)={sum(p.numel() for p in r10.adamw_parameters()):,}")

E_BASE = 5.780293e8          # baseline d10 estimate_flops() (accurate), from phase_a.log
N_ITERS_BASE, n_rungs = 1605, 11
reader_true_fpt = 6 * matmul * n_rungs                     # 6=fwd(2)+bwd(4); each weight hit on all 11 rungs
true_fpt_wide = E_BASE + reader_true_fpt
mult = true_fpt_wide / E_BASE
iters = round(N_ITERS_BASE / mult)
print(f"\nTRUE fwd+bwd FLOPs/token: baseline {E_BASE:.3e} + reader(x{n_rungs}) {reader_true_fpt:.3e} "
      f"= {true_fpt_wide:.3e}  ({mult:.3f}x)")
print(f"ISO-TOTAL-FLOP step budget: {N_ITERS_BASE} / {mult:.3f} = {iters}  ->  --num-iterations {iters}")
print(f"(estimate_flops would report only +{6*total/E_BASE*100:.1f}% -- the ~{n_rungs}x undercount)")

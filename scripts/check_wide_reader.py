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
from nanochat.gpt import GPT, GPTConfig


def build_and_check(depth, n_embd, reader_gate="none"):
    cfg = SimpleNamespace(
        reader="wide", n_embd=n_embd, n_layer=depth,
        reader_dim=128, reader_layers=2, reader_heads=5, reader_mlp_mult=4,  # reader_dim/heads ignored by wide
        reader_gate=reader_gate,
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

    # optimizer param coverage (mirrors the gpt.py setup_optimizer assert): matrix + adamw + gate
    all_p = list(reader.parameters())
    mat_p, adw_p, gate_p = reader.matrix_parameters(), reader.adamw_parameters(), reader.gate_parameters()
    assert len(mat_p) + len(adw_p) + len(gate_p) == len(all_p), (len(mat_p), len(adw_p), len(gate_p), len(all_p))
    assert {id(p) for p in all_p} == {id(p) for p in mat_p} | {id(p) for p in adw_p} | {id(p) for p in gate_p}, "uncovered param"
    assert len({id(p) for p in mat_p} & {id(p) for p in adw_p}) == 0, "matrix/adamw overlap"

    # gate semantics (load-bearing): ungated -> combine returns r (reader owns readout); gated ->
    # base + g*r, where g=0 must reproduce the baseline readout `base` bit-for-bit, and g=1 -> base+r.
    rb = torch.randn(B, T, D)
    if reader_gate == "none":
        assert reader.gate is None and len(gate_p) == 0
        assert torch.equal(reader.combine(rb, out), out), "ungated combine must return the reader output r"
    else:
        want_numel = 1 if reader_gate == "scalar" else n_embd
        assert reader.gate is not None and reader.gate.numel() == want_numel, reader.gate.shape
        assert abs(float(reader.gate.detach().flatten()[0]) - 1e-3) < 1e-9, float(reader.gate.detach().flatten()[0])
        with torch.no_grad():
            reader.gate.zero_()
        assert torch.equal(reader.combine(rb, out), rb), "gated combine at g=0 must equal base exactly"
        with torch.no_grad():
            reader.gate.fill_(1.0)
        assert torch.allclose(reader.combine(rb, out), rb + out), "gated combine at g=1 must equal base + r"

    gate_str = "none" if reader.gate is None else f"{reader_gate}{tuple(reader.gate.shape)}"
    print(f"d{depth:>2} n_embd={n_embd}: dim={reader.dim} heads={reader.n_heads} "
          f"layers={len(reader.blocks)} gate={gate_str} | readout{tuple(out.shape)} attn{tuple(w.shape)} | "
          f"coverage {len(mat_p)}+{len(adw_p)}+{len(gate_p)}={len(all_p)} OK")
    return reader


# depth 10 = real model; depth 12 = base_train's scaling-law reference (must also construct)
r10 = build_and_check(10, 640)
build_and_check(12, 768)
# gated variants (the new sweep uses scalar): g=0 reproduces the baseline readout, coverage holds
build_and_check(10, 640, reader_gate="scalar")
build_and_check(10, 640, reader_gate="channel")

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

# --- estimate_flops() now counts the reader per-rung: assert it equals baseline + 6*matmul*(L+1) ----
# (Change 2 -- the FLOP-accounting fix. Build full GPTs on meta; estimate_flops only reads numel/shape.)
def gpt_estimate_flops(reader, reader_gate="none"):
    cfg = GPTConfig(sequence_len=2048, vocab_size=32768, n_layer=10, n_head=5, n_kv_head=5,
                    n_embd=640, window_pattern="L", reader=reader, reader_gate=reader_gate,
                    reader_dim=128, reader_layers=2, reader_heads=5, reader_mlp_mult=4)
    with torch.device("meta"):
        m = GPT(cfg)
    return m.estimate_flops(), m

base_fpt, _ = gpt_estimate_flops("none")
gated_fpt, mg = gpt_estimate_flops("wide", "scalar")
matmul_gpt = sum(w.numel() for w in mg.reader.matrix_parameters())
expected_fpt = base_fpt + 6 * matmul_gpt * n_rungs
assert abs(base_fpt - E_BASE) / E_BASE < 2e-3, (base_fpt, E_BASE)            # baseline matches the phase_a anchor
assert gated_fpt == expected_fpt, (gated_fpt, expected_fpt)                  # reader counted at 6*matmul*(L+1)
assert matmul_gpt == matmul, (matmul_gpt, matmul)                            # same reader matmul as the standalone build
print(f"\nestimate_flops(): baseline {base_fpt:.4e}  reader(gated) {gated_fpt:.4e} "
      f"= baseline + 6*matmul*{n_rungs} ({gated_fpt/base_fpt:.3f}x)  -- per-rung count OK (was a ~{n_rungs}x undercount)")

"""Mechanical (no-data, CPU) check of the residual-free H trunk (h_residual="none").

The central risk: with both in-block skips and the x0 injection removed, nanochat's zero-initialized
output projections would make the whole trunk emit 0 at init (and relu^2'(0)=0 kills the gradient
too), so nothing would train. We init c_proj at fan-in scale instead. This builds the FULL model
(not just the reader) at a tiny size and checks that one forward+backward gives a finite loss and
NON-ZERO gradients deep in the trunk -- i.e. the residual-free trunk is alive at init.

Also checks (a) the WideReader runs on top of the residual-free trunk, (b) the x0 injection really
left the graph (x0_lambdas gets no grad), and (c) the stock baseline (h_residual="full", reader=none)
still builds and trains -- i.e. the gate left the baseline path intact.

    python -m scripts.check_residual_free
"""
import torch
from nanochat.gpt import GPT, GPTConfig


def build(h_residual, reader):
    # tiny config; n_embd=256 so WideReader gets n_embd//128 = 2 depth-heads
    cfg = GPTConfig(
        sequence_len=64, vocab_size=512, n_layer=6, n_head=2, n_kv_head=2, n_embd=256,
        window_pattern="L", reader=reader, reader_dim=256, h_residual=h_residual,
    )
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


def fwd_bwd(model):
    torch.manual_seed(0)
    B, T = 2, 32
    idx = torch.randint(0, 512, (B, T))
    targets = torch.randint(0, 512, (B, T))
    loss = model(idx, targets)
    loss.backward()
    return loss.item()


def gnorm(p):
    return 0.0 if p.grad is None else p.grad.norm().item()


print("=== residual-free trunk + WideReader (h_residual=none, reader=wide) ===")
m = build("none", "wide")
loss = fwd_bwd(m)
deep = m.transformer.h[-1]  # deepest block: gradient here proves the signal reached the top
checks = {
    "attn.c_proj (deep)": gnorm(deep.attn.c_proj.weight),
    "mlp.c_proj (deep)":  gnorm(deep.mlp.c_proj.weight),
    "attn.c_q (deep)":    gnorm(deep.attn.c_q.weight),
}
# The reader owns the readout, but it zero-inits some matrices (the residual-net trick; they are
# zero-grad at exactly step 0 and become active after the first update), so check that the reader
# receives gradient in aggregate rather than at any single parameter.
reader_grad = sum(gnorm(p) for p in m.reader.parameters())
print(f"loss={loss:.4f}  finite={bool(torch.isfinite(torch.tensor(loss)))}")
for k, v in checks.items():
    print(f"  grad |{k}| = {v:.3e}")
print(f"  grad |reader (total)| = {reader_grad:.3e}")
assert torch.isfinite(torch.tensor(loss)), "loss not finite"
for k, v in checks.items():
    assert v > 0, f"DEAD INIT: zero gradient at {k} -- residual-free trunk not alive"
assert reader_grad > 0, "reader received no gradient"
assert m.x0_lambdas.grad is None, "x0_lambdas got a grad, but the x0 injection should be removed"
print("  deep-trunk gradients non-zero (trunk alive)      OK")
print("  reader receives gradient                         OK")
print("  x0_lambdas.grad is None (x0 injection removed)   OK")

# The optimizer must EXCLUDE x0_lambdas in residual-free mode: it gets no grad, so leaving it in an
# AdamW group crashes the distributed all-reduce (DistMuonAdamW) at world_size>1 -- the bug the Modal
# pilot caught. resid_lambdas (the kept per-layer gain) must stay optimized.
opt = m.setup_optimizer(unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0)
in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
assert id(m.x0_lambdas) not in in_opt, "x0_lambdas must be EXCLUDED from the optimizer in residual-free mode"
assert id(m.resid_lambdas) in in_opt, "resid_lambdas should still be optimized (it carries the per-layer gain)"
print("  optimizer excludes x0_lambdas, keeps resid_lambdas   OK")

print("\n=== baseline parity (h_residual=full, reader=none) ===")
mb = build("full", "none")
lb = fwd_bwd(mb)
print(f"loss={lb:.4f}  finite={bool(torch.isfinite(torch.tensor(lb)))}")
assert torch.isfinite(torch.tensor(lb)), "baseline loss not finite"
# Stock nanochat zero-inits c_proj (the skip carries the signal at init), so at exactly step 0 the
# gradient to c_q/c_k/c_v is blocked by c_proj=0 -- only the output projections get gradient until
# c_proj moves off zero. So check the alive-at-init param (mlp.c_proj), not c_q. The residual-free
# arm's non-zero c_proj init is precisely why ITS deep c_q does get gradient above.
assert gnorm(mb.transformer.h[-1].mlp.c_proj.weight) > 0, "baseline output-proj grad zero?!"
assert mb.x0_lambdas.grad is not None, "baseline x0_lambdas should receive a grad (x0 injection active)"
print("  baseline builds + trains (output-proj grad non-zero), x0 injection active   OK")

print("\nALL CHECKS PASSED")

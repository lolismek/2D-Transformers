"""
Init-parity check for the 2D Transformer.

A freshly-built 2D model (vertical=True; zero-init vertical transformer with top-layer readout)
must produce logits IDENTICAL to vanilla GPT-2, because at initialization the vertical transformer
is the exact identity on the per-layer stack and returns the top-layer hidden state h_L.

This validates two things before we ever train:
  (1) from_pretrained loads GPT-2 weights into the 2D model's base (and leaves the vertical
      transformer at its fresh identity init),
  (2) the stack-collection + vertical wiring in GPT.forward preserves the baseline at init.
"""
import torch
from model import GPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)
print(f"device: {device}")

print("\n=== loading vanilla GPT-2 (124M) ===")
m_base = GPT.from_pretrained('gpt2').eval().to(device)

print("\n=== loading 2D GPT-2 (vertical=True, zero-init identity) ===")
m_2d = GPT.from_pretrained('gpt2', override_args=dict(
    vertical=True, n_vertical_layer=1, n_vertical_head=12, vertical_mlp_ratio=2)).eval().to(device)

# identical inputs and targets for both models
B, T = 4, 128
idx = torch.randint(0, 50257, (B, T), device=device)
targets = torch.randint(0, 50257, (B, T), device=device)

with torch.no_grad():
    logits_base, loss_base = m_base(idx, targets)
    logits_2d,   loss_2d   = m_2d(idx, targets)

max_abs = (logits_base - logits_2d).abs().max().item()
dloss = abs(loss_base.item() - loss_2d.item())

def nparams(m):
    return sum(p.numel() for p in m.parameters())
vert = sum(p.numel() for n, p in m_2d.named_parameters() if n.startswith('vertical.'))

print("\n=== results ===")
print(f"logits shape       : {tuple(logits_2d.shape)}")
print(f"max |delta logits| : {max_abs:.3e}")
print(f"loss base/2d/delta : {loss_base.item():.6f} / {loss_2d.item():.6f} / {dloss:.3e}")
print(f"base params        : {nparams(m_base)/1e6:.2f}M")
print(f"2d   params        : {nparams(m_2d)/1e6:.2f}M  (vertical adds {vert/1e6:.2f}M)")

ok = (max_abs < 1e-3) and (dloss < 1e-4)
print("\nPARITY:", "OK" if ok else "FAILED")
assert ok, "init-parity FAILED: 2D model at init does not match vanilla GPT-2"

"""Standalone integration check for a depth-reader (no data / tokenizer needed).

Mirrors base_train's meta -> to_empty -> init_weights flow at depth 10, then for both
reader='none' (baseline) and reader='vertical':
  - verifies num_scaling_params() and setup_optimizer() asserts pass (reader params wired)
  - runs a forward + backward + one optimizer step on random tokens (no NaNs)
  - checks the token-budget scaling params (transformer_matrices + lm_head) are IDENTICAL
    between baseline and reader (so both auto-get the same training horizon)
  - checks reader query-pool attention sums to 1 over the rungs

Run on a *free* GPU, e.g.:  CUDA_VISIBLE_DEVICES=3 uv run python -m scripts.check_reader
"""
import os
import torch
from nanochat.gpt import GPT, GPTConfig

DEVICE = os.environ.get("CHECK_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def build(reader, **kw):
    cfg = GPTConfig(sequence_len=512, vocab_size=32768, n_layer=10, n_head=5, n_kv_head=5,
                    n_embd=640, window_pattern="L", reader=reader, **kw)
    with torch.device("meta"):
        m = GPT(cfg)
    m.to_empty(device=DEVICE)
    m.init_weights()
    return m


def run(reader, **kw):
    torch.manual_seed(0)
    m = build(reader, **kw)
    pc = m.num_scaling_params()                      # asserts total == sum(params)
    opt = m.setup_optimizer()                         # asserts reader params are covered
    x = torch.randint(0, 32768, (2, 512), device=DEVICE)
    y = torch.randint(0, 32768, (2, 512), device=DEVICE)
    loss = m(x, y)
    loss.backward()
    gnorm = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in m.parameters() if p.grad is not None))
    opt.step()                                        # exercise Muon (reader matrices) + AdamW (depth_pos/q)
    w = m.reader.attn_weights() if m.reader is not None else None
    wsum = w.sum(-1).mean().item() if w is not None else None
    wshape = tuple(w.shape) if w is not None else None
    return pc, loss.item(), gnorm.item(), wshape, wsum


print(f"device={DEVICE}")
base_pc, base_loss, base_g, _, _ = run("none")
r_pc, r_loss, r_g, wshape, wsum = run("vertical")

scaling_base = base_pc["transformer_matrices"] + base_pc["lm_head"]
scaling_r = r_pc["transformer_matrices"] + r_pc["lm_head"]
print(f"baseline : total={base_pc['total']:,}  reader=0  loss={base_loss:.3f}  gnorm={base_g:.2f}")
print(f"vertical : total={r_pc['total']:,}  reader={r_pc['reader']:,}  loss={r_loss:.3f}  gnorm={r_g:.2f}")
print(f"reader adds {r_pc['reader']:,} params "
      f"(+{100 * r_pc['reader'] / base_pc['transformer_matrices']:.2f}% of transformer matrices)")
print(f"scaling params (transformer_matrices+lm_head): baseline={scaling_base:,} vertical={scaling_r:,} "
      f"-> {'EQUAL (same token budget)' if scaling_base == scaling_r else 'DIFFER!!'}")
print(f"reader attn shape={wshape}  sum-over-rungs(mean)={wsum:.4f} (want ~1.0)")

assert scaling_base == scaling_r, "token-budget scaling params differ between baseline and reader!"
assert wshape == (2, 512, 11), f"unexpected reader attn shape {wshape}"
assert abs(wsum - 1.0) < 1e-2, "reader attention does not sum to 1"
for nm, v in [("base_loss", base_loss), ("r_loss", r_loss), ("base_g", base_g), ("r_g", r_g)]:
    assert v == v and abs(v) < 1e4, f"{nm} looks wrong: {v}"
print("OK: integration sane.")

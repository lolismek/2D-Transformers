"""Distributed (world_size > 1) integration check for a depth-reader.

The single-GPU scripts/check_reader.py never touches DistMuonAdamW's collectives, so it
missed two world_size>1-only bugs: (1) backout_lambda has no grad in reader mode -> None fed
to dist.all_reduce; (2) the reader's (n_rungs, dV) depth-pos embeds have a leading dim not
divisible by world_size -> reduce_scatter assert. This runs a few real optimizer steps under
torchrun for both reader='none' and 'vertical' to exercise that path.

Run on the free GPUs, e.g.:
    CUDA_VISIBLE_DEVICES=2,3 uv run torchrun --standalone --nproc_per_node=2 -m scripts.check_reader_dist
"""
import torch
from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type
from nanochat.gpt import GPT, GPTConfig

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
print0(f"world_size={ddp_world_size} device={device}")

# (reader, reader_gate): 'none' baseline; 'vertical' ungated (backout excluded); 'wide'+'scalar'
# gated -> exercises the gate param's all-reduce AND the gated backout path (backout_lambda keeps a grad).
for reader, reader_gate in (("none", "none"), ("vertical", "none"), ("wide", "scalar")):
    torch.manual_seed(0)
    cfg = GPTConfig(sequence_len=256, vocab_size=32768, n_layer=10, n_head=5, n_kv_head=5,
                    n_embd=640, window_pattern="L", reader=reader, reader_gate=reader_gate)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=device)
    model.init_weights()
    opt = model.setup_optimizer()                       # asserts reader params wired (incl. gate group)
    x = torch.randint(0, 32768, (4, 256), device=device)
    y = torch.randint(0, 32768, (4, 256), device=device)
    loss = None
    for step in range(3):                               # >=2 steps: exercises AdamW state init + reuse
        opt.zero_grad(set_to_none=True)
        loss = model(x, y)
        loss.backward()
        opt.step()                                      # the distributed reduce/compute/gather path
    gate_str = "-" if getattr(model.reader, "gate", None) is None else f"{model.reader.gate.mean().item():.4f}"
    print0(f"reader={reader:>8} gate={reader_gate:>6}: 3 distributed optimizer steps OK | "
           f"final loss={loss.item():.3f} | gate={gate_str}")

compute_cleanup()
print0("DIST_CHECK_OK")

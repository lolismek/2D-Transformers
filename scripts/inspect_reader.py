"""Inspect the VerticalReader's query-pool attention over the depth rungs.

Loads a trained base checkpoint, runs a few val batches, and reports how the single
learned query distributes attention over the N = n_layer+1 rungs [x0, h_1..h_L]:

  - mean weight per rung (averaged over tokens)         -> which rungs V reads
  - top-rung (h_L) mass                                 -> ~1.0 => V collapsed to baseline
  - argmax-rung histogram                               -> which rung dominates per token
  - normalized entropy                                  -> spread vs peaked
  - per-rung std across tokens                          -> content-dependent vs static

Usage (single GPU is fine):
    python -m scripts.inspect_reader --model-tag d10_reader --device-batch-size 8 --num-batches 8
"""
import argparse
import torch

from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import load_model
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit


def main():
    parser = argparse.ArgumentParser(description="Inspect VerticalReader depth attention")
    parser.add_argument("--model-tag", type=str, required=True, help="checkpoint tag, e.g. d10_reader")
    parser.add_argument("--step", type=int, default=None, help="step to load (default = last)")
    parser.add_argument("--device-batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--device-type", type=str, default="")
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
    model.eval()
    if getattr(model, "reader", None) is None:
        print0(f"Model '{args.model_tag}' has no reader (reader='none'); nothing to inspect.")
        compute_cleanup()
        return
    sequence_len = meta["model_config"]["sequence_len"]
    n_rungs = model.reader.n_rungs

    loader = tokenizing_distributed_data_loader_bos_bestfit(
        get_tokenizer(), args.device_batch_size, sequence_len, "val", device=device)

    ws = []
    with torch.no_grad():
        for _ in range(args.num_batches):
            x, y = next(loader)                # these loaders are generators (cf. base_train)
            _ = model(x)                       # forward populates reader._last_w as a side effect
            w = model.reader.attn_weights()    # (B, T, N)
            ws.append(w.reshape(-1, n_rungs).float().cpu())
    W = torch.cat(ws, dim=0)                   # (num_tokens, N)

    mean_w = W.mean(dim=0)                      # (N,)
    std_w = W.std(dim=0)                        # (N,)
    top_mass = mean_w[-1].item()               # weight on h_L (the rung baseline reads)
    argmax_hist = torch.bincount(W.argmax(dim=1), minlength=n_rungs).float()
    argmax_hist = argmax_hist / argmax_hist.sum()
    ent = -(W.clamp_min(1e-9).log() * W).sum(dim=1).mean().item()
    norm_ent = ent / torch.log(torch.tensor(float(n_rungs))).item()

    print0(f"\nVerticalReader depth attention | model={args.model_tag} step={meta['step']} | tokens={W.size(0):,}")
    print0(f"rungs: 0=x0(embed) .. {n_rungs-1}=h_L(top)")
    print0(f"{'rung':>5} {'mean_w':>9} {'std_w':>9} {'argmax%':>9}")
    for r in range(n_rungs):
        print0(f"{r:>5} {mean_w[r].item():>9.4f} {std_w[r].item():>9.4f} {100*argmax_hist[r].item():>8.1f}%")
    print0(f"\ntop-rung (h_L) mean mass : {top_mass:.4f}   (~1.0 => collapsed to baseline readout)")
    print0(f"normalized entropy        : {norm_ent:.4f}   (1.0 => uniform over rungs, 0 => one-hot)")
    print0(f"off-top mass              : {1.0 - top_mass:.4f}   (fraction of attention NOT on h_L)")

    compute_cleanup()


if __name__ == "__main__":
    main()

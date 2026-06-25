"""Cleanest collapse probe: does the depth-reader V need anything besides the top rung h_L?

Eval-time intervention on a trained reader checkpoint. We overwrite EVERY rung of the
residual ladder [x0, h_1..h_L] with the top rung h_L, so the reader is handed only h_L's
*content* (its own learned depth-pos structure is left intact), then re-measure val bpb:

  * full ladder         -> sanity; should reproduce the trained val bpb (~0.877 for L4)
  * ablated (all = h_L) -> the test.  bpb ~ unchanged => V only needed h_L (collapse/identity);
                                      bpb clearly worse => V genuinely used the lower rungs.

Both passes consume the SAME val tokens (fresh, identical loaders), so the per-token loss
difference is paired and the delta is far more precise than either absolute number.

Usage (single GPU):
    NANOCHAT_BASE_DIR=/cache python -m scripts.ablate_reader --model-tag d10_wide640_L4_full \
        --device-batch-size 16 --eval-tokens 2097152
"""
import argparse
import torch

from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import load_model
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb


def main():
    p = argparse.ArgumentParser(description="Top-rung ablation probe for depth-readers")
    p.add_argument("--model-tag", type=str, required=True, help="checkpoint tag, e.g. d10_wide640_L4_full")
    p.add_argument("--step", type=int, default=None, help="step to load (default = last)")
    p.add_argument("--device-batch-size", type=int, default=16)
    p.add_argument("--eval-tokens", type=int, default=2097152,
                   help="val tokens per pass (default matches the training-time eval budget)")
    p.add_argument("--device-type", type=str, default="")
    args = p.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
    model.eval()
    reader = getattr(model, "reader", None)
    if reader is None or not getattr(reader, "needs_ladder", False):
        print0(f"Model '{args.model_tag}' has no ladder-reader; nothing to ablate.")
        compute_cleanup()
        return

    sequence_len = meta["model_config"]["sequence_len"]
    token_bytes = get_token_bytes(device=device)
    tok = get_tokenizer()

    tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size
    steps = max(1, args.eval_tokens // tokens_per_step)
    eff_tokens = steps * tokens_per_step

    def fresh_loader():
        # generator over the val split; deterministic order => both passes see identical tokens
        return tokenizing_distributed_data_loader_bos_bestfit(
            tok, args.device_batch_size, sequence_len, "val", device=device)

    # --- pass 1: full ladder (sanity; should reproduce the trained val bpb) ---
    bpb_full = evaluate_bpb(model, fresh_loader(), steps, token_bytes)
    # capture the genuine per-rung query-pool weights from this (un-ablated) pass, for color
    w = reader.attn_weights()
    mean_w = w.reshape(-1, reader.n_rungs).float().mean(dim=0).cpu() if w is not None else None

    # --- install the ablation: hand the reader a ladder whose every rung is the top rung h_L ---
    orig_readout = reader.readout
    def ablated_readout(ladder):
        top = ladder[-1]                              # h_L, the exact state the baseline reads out
        return orig_readout([top] * len(ladder))      # destroy rungs 0..L-1, keep only h_L's content
    reader.readout = ablated_readout

    # --- pass 2: ablated ladder (the test), SAME val tokens ---
    bpb_abl = evaluate_bpb(model, fresh_loader(), steps, token_bytes)

    reader.readout = orig_readout  # restore (be a good citizen)

    print0("\n" + "=" * 66)
    print0(f"Top-rung ablation | model={args.model_tag} step={meta['step']} | "
           f"{eff_tokens:,} val tokens/pass (dbs={args.device_batch_size}, world={ddp_world_size})")
    print0(f"rungs: 0=x0(embed) .. {reader.n_rungs-1}=h_L(top); ablation sets EVERY rung := h_L")
    print0("-" * 66)
    print0(f"  full ladder    val bpb : {bpb_full:.6f}")
    print0(f"  ablated (=h_L) val bpb : {bpb_abl:.6f}")
    print0(f"  delta (ablated - full) : {bpb_abl - bpb_full:+.6f}")
    print0("-" * 66)
    if mean_w is not None:
        top = mean_w[-1].item()
        print0(f"  [secondary] full-pass pool mass on h_L: {top:.4f}  (off-top {1.0 - top:.4f})")
    print0("read: delta ~ 0  => V only needs h_L (collapse / effective identity)")
    print0("      delta >> 0 => V genuinely reads the lower rungs (the ladder carries signal h_L lacks)")
    print0("=" * 66)

    compute_cleanup()


if __name__ == "__main__":
    main()

"""Frozen-feature probe: the predictively-OPTIMAL 128-dim linear cap on the baseline readout.

The PCA probe (scripts/svd_readout_probe.py) cut the readout to its top-128 VARIANCE directions
and cost +0.454 bpb -- a loose upper bound, because variance != usefulness. This finds the
top-128 *prediction* directions instead, by gradient descent, holding everything else frozen:

  baseline readout x (lm_head input, n_embd) --> down (n_embd->k) --> up (k->n_embd) --> [FROZEN] lm_head

The whole backbone H and lm_head are frozen; only `down`/`up` train (~164k params). We WARM-START
them from the PCA solution, so step 0 reproduces the PCA-128 bpb (~1.31). Training then minimizes
the model's OWN cross-entropy on live forwards (loss = model(x, y) with the bottleneck hooked into
the lm_head input) -- this pairs features<->targets internally and uses the exact head, so it is
correct by construction and can only improve on the 1.31 warm-start.

Decision (cost = bpb_probe - bpb_baseline_on_same_slice):
  ~ +0.056 (the reader's deficit)  => a trained 128-cap on the top state costs what the reader lost
                                       => the bottleneck explains it; depth-reading ~neutral.
  ~ 0                              => 128 dims is plenty => reader's deficit is readout/mixing, not width.
  in between                      => both contribute.
Caveat: reads only h_L (not the ladder) and uses FROZEN features (the real reader co-adapts H),
so this is a tighter-than-PCA upper bound; a cheap result refutes the bottleneck, an expensive
one still leaves the d_V=640 retrain as the final word.

    CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m scripts.frozen_probe --model-tag d10_baseline
"""
import argparse
import torch
import torch.nn as nn

from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import load_model
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb


def main():
    p = argparse.ArgumentParser(description="Frozen-feature trained 128-dim bottleneck probe")
    p.add_argument("--model-tag", type=str, default="d10_baseline")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--k", type=int, default=128, help="bottleneck width")
    p.add_argument("--device-batch-size", type=int, default=16)
    p.add_argument("--basis-steps", type=int, default=24, help="batches used to estimate the PCA warm-start basis")
    p.add_argument("--train-steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--eval-steps", type=int, default=32, help="val batches for each bpb measurement")
    p.add_argument("--device-type", type=str, default="")
    args = p.parse_args()

    torch.manual_seed(0)
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)                          # freeze the entire baseline
    D = meta["model_config"]["n_embd"]
    k = args.k
    token_bytes = get_token_bytes(device=device)
    print0(f"model={args.model_tag} step={meta['step']} n_embd={D} k={k}")

    def loader(split):
        return tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, args.device_batch_size, meta["model_config"]["sequence_len"], split, device=device)

    # -------------------------------------------------------------------------
    # Estimate the PCA warm-start basis from the readout covariance (features only).
    # -------------------------------------------------------------------------
    feats = []

    def capture_hook(module, inputs):
        feats.append(inputs[0].reshape(-1, D).detach().float())
        return None

    h = model.lm_head.register_forward_pre_hook(capture_hook)
    ltr = loader("train")
    with torch.no_grad():
        for _ in range(args.basis_steps):
            x, y = next(ltr)
            model(x, y, loss_reduction="none")
    h.remove()
    Xf = torch.cat(feats, dim=0)                            # (Ntok, D)
    mean = Xf.mean(dim=0)
    Xc = Xf - mean
    cov = (Xc.t() @ Xc) / Xf.shape[0]
    _, U = torch.linalg.eigh(0.5 * (cov + cov.t()).double())
    Uk = U.flip(1)[:, :k].float()                          # top-k principal directions (D,k)
    del feats, Xf, Xc, cov
    print0(f"PCA warm-start basis from {args.basis_steps * args.device_batch_size * meta['model_config']['sequence_len']:,} tokens")

    # -------------------------------------------------------------------------
    # Bottleneck (affine rank-k), warm-started so up(down(x)) = PCA-k reconstruction.
    # -------------------------------------------------------------------------
    down = nn.Linear(D, k, bias=True).to(device).float()
    up = nn.Linear(k, D, bias=True).to(device).float()
    with torch.no_grad():
        down.weight.copy_(Uk.t());  down.bias.copy_(-(Uk.t() @ mean))
        up.weight.copy_(Uk);        up.bias.copy_(mean)

    def probe_hook(module, inputs):
        x = inputs[0]
        return (up(down(x.detach().float())).to(x.dtype),)   # backbone frozen via detach; grad -> down/up

    def probe_bpb():
        hh = model.lm_head.register_forward_pre_hook(probe_hook)
        b = evaluate_bpb(model, loader("val"), args.eval_steps, token_bytes)
        hh.remove()
        return b

    bpb_base = evaluate_bpb(model, loader("val"), args.eval_steps, token_bytes)   # no hook, same slice
    bpb_step0 = probe_bpb()
    print0(f"\nbaseline bpb (val slice, no bottleneck) = {bpb_base:.6f}")
    print0(f"probe bpb @ step 0 (PCA warm-start)      = {bpb_step0:.6f}   (sanity: ~PCA-128 1.31)")

    # -------------------------------------------------------------------------
    # Train ONLY down/up on live forwards via the model's own loss (correct pairing + exact head).
    # -------------------------------------------------------------------------
    opt = torch.optim.Adam(list(down.parameters()) + list(up.parameters()), lr=args.lr)
    n_params = sum(prm.numel() for prm in list(down.parameters()) + list(up.parameters()))
    hh = model.lm_head.register_forward_pre_hook(probe_hook)
    ltr = loader("train")
    x, y = next(ltr)
    with torch.no_grad():
        l0 = model(x, y)
    print0(f"\nwarm-start live train loss = {l0.item():.4f} nats   (sanity: a few nats, NOT ~11.9)")
    print0(f"training bottleneck ({n_params:,} params)...")
    for step in range(1, args.train_steps + 1):
        x, y = next(ltr)
        loss = model(x, y)                                 # mean CE, exact head, correct pairing
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == 1:
            print0(f"  step {step:>4}/{args.train_steps}  train_ce={loss.item():.4f} nats")
    hh.remove()

    bpb_probe = probe_bpb()

    # -------------------------------------------------------------------------
    # Verdict.
    # -------------------------------------------------------------------------
    cost = bpb_probe - bpb_base
    print0("\n" + "=" * 64)
    print0(f"baseline (full {D}-dim readout) : {bpb_base:.6f}")
    print0(f"trained {k}-dim bottleneck      : {bpb_probe:.6f}")
    print0(f"COST of a trained {k}-dim cap   : {cost:+.6f} bpb")
    print0("-" * 64)
    print0(f"  reader's measured deficit     : +0.056   (joint-trained {k}-dim reader over the ladder)")
    print0(f"  PCA (variance) {k}-dim cap     : +0.454   (untrained, wrong objective)")
    print0("=" * 64)
    if cost < 0.015:
        print0("READ: a trained 128-cap is ~free => bottleneck does NOT explain the reader's loss")
        print0("      => the deficit is the readout/depth-mixing, not width. (refutes bottleneck)")
    elif cost > 0.045:
        print0("READ: a trained 128-cap costs ~the reader's deficit => bottleneck is the prime suspect")
        print0("      => confirm decisively with the d_V=640 retrain.")
    else:
        print0("READ: partial => width explains some of the deficit; readout/mixing the rest.")

    compute_cleanup()


if __name__ == "__main__":
    main()

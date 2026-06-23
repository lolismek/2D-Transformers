"""Rank probe of the model's readout (the lm_head input) — no retraining.

Tests the "128-dim bottleneck" hypothesis on an existing base checkpoint. Operates on the
post-final-norm readout vector x (the exact input to lm_head, n_embd-dim; gpt.py:504->508):

  1. Singular spectrum: stack x over val tokens, eigendecompose its (centered) covariance ->
     how many dims the readout VARIES along (participation ratio, cumulative energy).
     NOTE: for a *reader* checkpoint this is tautological (rank <= d_V by construction, since the
     readout is up_proj(r_V)); it's the *baseline* spectrum that's informative.

  2. Truncation -> bpb curve (the causal part): project x onto its top-k PCA subspace,
     reconstruct, and run the REAL lm_head + softcap + cross-entropy via the stock evaluate_bpb,
     for a sweep of k. bpb(k) measures how many dims the head actually NEEDS:
       - bpb(n_embd) reconstructs x exactly  -> must reproduce the model's known bpb (CORRECTNESS CHECK).
       - bpb(128) vs bpb(n_embd) = the predictive cost of a 128-dim cap. ~0.056 (the reader's
         measured deficit) => the bottleneck is the prime suspect; ~0 => width is cheap and the
         reader loses for other reasons (readout/depth-mixing).

  Variance (spectrum) != usefulness (bpb curve): a direction can carry variance yet barely move the
  logits, or vice versa. PCA picks the max-VARIANCE k-subspace, not the max-PREDICTIVE one, so
  bpb(128) here is an UPPER BOUND on an optimally-trained 128-dim linear cap. We report both and
  lead with the bpb curve.

Single GPU is fine:
    CUDA_VISIBLE_DEVICES=2 uv run python -m scripts.svd_readout_probe --model-tag d10_baseline
"""
import os
import json
import argparse
import torch

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import load_model
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb


def parse_ks(s):
    return [int(k) for k in s.split(",") if k.strip() != ""]


def main():
    p = argparse.ArgumentParser(description="SVD / truncation-bpb rank probe of the readout")
    p.add_argument("--model-tag", type=str, default="d10_baseline")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--device-batch-size", type=int, default=16)
    p.add_argument("--basis-steps", type=int, default=8, help="batches used to estimate the covariance basis")
    p.add_argument("--eval-steps", type=int, default=32, help="batches used per k for the bpb sweep")
    p.add_argument("--ks", type=str, default="1,8,16,32,64,96,128,160,192,256,384,512,640")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--device-type", type=str, default="")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
    model.eval()
    sequence_len = meta["model_config"]["sequence_len"]
    D = meta["model_config"]["n_embd"]
    token_bytes = get_token_bytes(device=device)
    has_reader = getattr(model, "reader", None) is not None
    print0(f"model={args.model_tag} step={meta['step']} n_embd={D} seq={sequence_len} reader={has_reader}")

    def fresh_loader():
        # val loader is deterministic from the start, so recreating it gives every k the same tokens
        return tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, args.device_batch_size, sequence_len, args.split, device=device)

    # -------------------------------------------------------------------------
    # Pass A: estimate the (centered) covariance of the post-norm readout x.
    # -------------------------------------------------------------------------
    N = torch.zeros((), dtype=torch.float64, device=device)
    sum_x = torch.zeros(D, dtype=torch.float64, device=device)
    sum_xxT = torch.zeros(D, D, dtype=torch.float64, device=device)

    def capture_hook(module, inputs):
        x = inputs[0].reshape(-1, D).double()   # (B*T, D) post-norm readout = lm_head input
        N.add_(x.shape[0])
        sum_x.add_(x.sum(dim=0))
        sum_xxT.add_(x.t() @ x)
        return None                              # observe only

    h = model.lm_head.register_forward_pre_hook(capture_hook)
    loader = fresh_loader()
    with torch.no_grad():
        for _ in range(args.basis_steps):
            x, y = next(loader)
            model(x, y, loss_reduction="none")
    h.remove()

    n = int(N.item())
    mean = sum_x / N                              # (D,)
    cov = sum_xxT / N - torch.outer(mean, mean)   # centered covariance (D,D)
    cov = 0.5 * (cov + cov.t())                   # symmetrize away roundoff
    evals, U = torch.linalg.eigh(cov)             # ascending eigh of a symmetric PSD matrix
    evals = evals.flip(0).clamp_min(0.0)          # descending; clamp tiny negative roundoff
    U = U.flip(1)                                 # columns = principal directions (descending)
    mean_f = mean.float()
    U_f = U.float()

    total_var = evals.sum().item()
    cum = (torch.cumsum(evals, dim=0) / max(total_var, 1e-30))
    part_ratio = (evals.sum() ** 2 / (evals ** 2).sum()).item()  # participation ratio = effective rank

    print0(f"\n[spectrum] tokens={n:,}  total_var={total_var:.4g}  participation_ratio(eff_rank)={part_ratio:.1f}")
    print0(f"{'k':>5} {'cum_energy':>11}")
    for k in [1, 8, 16, 32, 64, 128, 192, 256, 384, 512, D]:
        if 1 <= k <= D:
            print0(f"{k:>5} {cum[k - 1].item():>11.4f}")

    # -------------------------------------------------------------------------
    # Pass B: truncation -> bpb sweep. Replace x with its top-k PCA reconstruction.
    # -------------------------------------------------------------------------
    ks = [k for k in parse_ks(args.ks) if 0 <= k <= D]
    ks = sorted(set(ks + [D]))                    # always include the k=D identity correctness check
    state = {"k": None, "P": None}                # P = Uk Uk^T (projection onto top-k subspace)

    def trunc_hook(module, inputs):
        x = inputs[0]
        x2 = x.reshape(-1, D).float()
        if state["k"] == 0:
            recon = mean_f.expand(x2.shape[0], D)
        else:
            recon = mean_f + (x2 - mean_f) @ state["P"]
        return (recon.to(x.dtype).reshape(x.shape),)

    results = {}
    for k in ks:
        if k == 0:
            state["k"], state["P"] = 0, None
        else:
            Uk = U_f[:, :k]
            state["k"], state["P"] = k, Uk @ Uk.t()
        h = model.lm_head.register_forward_pre_hook(trunc_hook)
        bpb = evaluate_bpb(model, fresh_loader(), args.eval_steps, token_bytes)
        h.remove()
        results[k] = bpb
        print0(f"[bpb] k={k:>4}  bpb={bpb:.6f}")

    bpb_full = results[D]
    print0(f"\n{'k':>5} {'bpb':>10} {'d_vs_full':>12} {'cum_energy':>11}")
    for k in ks:
        ce = cum[k - 1].item() if k >= 1 else 0.0
        print0(f"{k:>5} {results[k]:>10.6f} {results[k] - bpb_full:>+12.6f} {ce:>11.4f}")

    if 128 in results:
        print0(f"\nHEADLINE: bpb(128) - bpb(full={D}) = {results[128] - bpb_full:+.6f}   "
               f"(reader's measured deficit was +0.056)")
        print0(f"          variance energy beyond 128 dims = {1.0 - cum[127].item():.4f}   "
               f"(geometric only; contrast with the predictive cost above)")

    if ddp_rank == 0:
        out = args.out or os.path.join(get_base_dir(), f"svd_probe_{args.model_tag}.json")
        with open(out, "w") as f:
            json.dump({
                "model_tag": args.model_tag, "step": meta["step"], "n_embd": D,
                "tokens_basis": n, "eval_steps": args.eval_steps, "has_reader": has_reader,
                "participation_ratio": part_ratio,
                "spectrum": evals.tolist(), "cum_energy": cum.tolist(),
                "bpb_by_k": {str(k): results[k] for k in ks}, "bpb_full": bpb_full,
            }, f)
        print0(f"\nsaved -> {out}")

    compute_cleanup()


if __name__ == "__main__":
    main()

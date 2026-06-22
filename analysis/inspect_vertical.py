"""
Inspect the trained vertical (over-layers) transformer:
  1) generate a few text samples (sanity that the refined top-layer readout still decodes well)
  2) extract the DEPTH attention -- in particular the readout row: which of the S = n_layer+1
     per-layer states the top (readout) position attends to. This is the key interpretability
     test: a non-degenerate spread => genuine cross-depth combination; a near one-hot on the top
     layer => the vertical transformer collapsed onto the baseline and the gain was just params.

Run from the repo root (kingcrab, 2dtf env):
    ~/miniforge3/envs/2dtf/bin/python analysis/inspect_vertical.py
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import tiktoken

# make src/ importable regardless of the current working directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from model import GPTConfig, GPT, make_config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_path = 'runs/vertical_frozen/ckpt.pt'
data_dir = 'data/openwebtext'
torch.manual_seed(1337)

# ---------------------------------------------------------------- load model
ckpt = torch.load(ckpt_path, map_location=device)
conf = make_config(ckpt['model_args'])
model = GPT(conf)
sd = ckpt['model']
for k in list(sd.keys()):                       # strip a torch.compile prefix if present
    if k.startswith('_orig_mod.'):
        sd[k[len('_orig_mod.'):]] = sd.pop(k)
model.load_state_dict(sd)
model.eval().to(device)
bvl = ckpt.get('best_val_loss')
bvl = float(bvl) if bvl is not None else float('nan')
print(f"loaded checkpoint: step {ckpt.get('iter_num')}, best val loss {bvl:.4f}")
assert conf.vertical, "this checkpoint has no vertical transformer"

enc = tiktoken.get_encoding('gpt2')
S = conf.n_layer + 1                            # number of depth positions (13)
labels = ['h0(emb)'] + [f'h{i}' for i in range(1, S - 1)] + [f'h{S-1}(top)']

# ---------------------------------------------------------------- 1) samples
print("\n" + "=" * 72 + "\nSAMPLES  (temperature 0.8, top_k 200)\n" + "=" * 72)
prompts = [
    "The meaning of life is",
    "In a shocking finding, scientists discovered that",
    "Breaking news from Washington:",
]
for p in prompts:
    ids = torch.tensor([enc.encode_ordinary(p)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=120, temperature=0.8, top_k=200)
    print(f"\n--- {p!r} ---\n{enc.decode(out[0].tolist())}")

# ---------------------------------------------------------------- 2) depth attention
# Real batch from val.bin, manual forward to collect the per-layer stack H, then run the
# vertical transformer with save_attn on to capture its (S x S) attention.
val = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
B, T = 16, 256
ix = torch.randint(len(val) - T, (B,))
x = torch.stack([torch.from_numpy(val[i:i + T].astype(np.int64)) for i in ix]).to(device)

attn = model.vertical.blocks[0].attn
attn.save_attn = True
with torch.no_grad():
    pos = torch.arange(T, device=device)
    h = model.transformer.drop(model.transformer.wte(x) + model.transformer.wpe(pos))
    hs = [h]
    for blk in model.transformer.h:
        h = blk(h); hs.append(h)
    H = torch.stack(hs, dim=2)                  # (B, T, S, d)  per-layer stack
    y = model.vertical(H)                       # (B, T, d) refined readout; fills attn.attn_weights
raw_top = H[:, :, -1, :]                        # un-refined top layer = what vanilla GPT-2 reads

A = attn.attn_weights.float()                   # (B*T, n_head, S, S)
A_mean = A.mean(dim=(0, 1))                     # (S, S) mean over tokens and heads
readout = A_mean[-1]                            # (S,) the readout (top) position's attention over layers

# how much the vertical transformer actually moved the readout away from plain GPT-2's top layer
cos = F.cosine_similarity(y, raw_top, dim=-1).mean().item()
relshift = ((y - raw_top).norm(dim=-1) / raw_top.norm(dim=-1)).mean().item()

print("\n" + "=" * 72 + "\nDEPTH ATTENTION  -- readout (top) position over the 13 layer states\n" + "=" * 72)
mx = readout.max().item()
for lab, w in zip(labels, readout.tolist()):
    print(f"  {lab:>9}  {w*100:5.1f}%  {'#' * int(round(40 * w / mx))}")
ent = -(readout * (readout + 1e-12).log()).sum().item() / np.log(S)
off_top = 1.0 - readout[-1].item()
print(f"\n  off-top mass (attention NOT on the top layer): {off_top*100:.1f}%")
print(f"  argmax layer: {labels[int(readout.argmax())]}   entropy(normalized 0..1): {ent:.3f}")
print(f"  readout vs plain top layer:  cosine {cos:.3f}   relative shift {relshift*100:.1f}%")

# per-head: which layer each head's readout most attends to (specialization across depth)
rh = A[:, :, -1, :].mean(dim=0)                 # (n_head, S)
print("\n  per-head readout argmax (head: layer @ weight):")
for hd in range(rh.size(0)):
    j = int(rh[hd].argmax())
    print(f"    head {hd:2d}: {labels[j]:>9} @ {rh[hd, j]*100:4.1f}%")

# full mean depth-attention map (rows = query layer, cols = attended layer)
print("\n" + "=" * 72 + "\nFULL MEAN DEPTH-ATTENTION MAP  (row attends over cols)\n" + "=" * 72)
print("        " + "".join(f"{i:>5}" for i in range(S)))
for i in range(S):
    print(f"  {labels[i]:>9} " + "".join(f"{A_mean[i, j].item()*100:5.1f}" for j in range(S)))

"""
Visualize the vertical (over-layers) transformer's DEPTH attention.

For each example sentence produces a PNG with two panels:
  (left)  per-token readout attention: rows = tokens (the words), cols = the 13
          depth positions h0(emb)..h12(top); value = how much that token's readout
          (top) position attends to each layer-state. "Which layers feed the
          prediction at this word."
  (right) the 13x13 vertical attention matrix (mean over the sentence's tokens and
          heads): row = query layer position, col = attended layer position.

Also writes a per-head map (12 heads x 13 layers) over a val batch, showing how
different heads specialize on different depths.

Run from the repo root (kingcrab, 2dtf env):
    ~/miniforge3/envs/2dtf/bin/python analysis/viz_vertical.py
"""
import os
import sys
import numpy as np
import torch
import tiktoken
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# make src/ importable regardless of the current working directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from model import GPTConfig, GPT, make_config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_path = 'runs/vertical_frozen/ckpt.pt'
viz_dir = 'runs/vertical_frozen/viz'
os.makedirs(viz_dir, exist_ok=True)
torch.manual_seed(1337)

# ---------------------------------------------------------------- load model
ckpt = torch.load(ckpt_path, map_location=device)
conf = make_config(ckpt['model_args'])
model = GPT(conf)
sd = ckpt['model']
for k in list(sd.keys()):
    if k.startswith('_orig_mod.'):
        sd[k[len('_orig_mod.'):]] = sd.pop(k)
model.load_state_dict(sd)
model.eval().to(device)

enc = tiktoken.get_encoding('gpt2')
S = conf.n_layer + 1
col_labels = ['emb'] + [f'{i}' for i in range(1, S - 1)] + [f'{S-1}\n(top)']
attn = model.vertical.blocks[0].attn
attn.save_attn = True


def tok_label(t):
    s = enc.decode([t]).replace('\n', '\\n')
    if s.startswith(' '):
        s = '·' + s[1:]          # show a leading space as a middle dot
    return s if s else '␀'


def depth_attention(ids):
    """Return (A) attention (T, n_head, S, S) and token strings for a token id list."""
    x = torch.tensor([ids], device=device)
    T = len(ids)
    with torch.no_grad():
        pos = torch.arange(T, device=device)
        h = model.transformer.drop(model.transformer.wte(x) + model.transformer.wpe(pos))
        hs = [h]
        for blk in model.transformer.h:
            h = blk(h); hs.append(h)
        H = torch.stack(hs, dim=2)
        _ = model.vertical(H)                       # fills attn.attn_weights
    A = attn.attn_weights.float().view(T, -1, S, S).cpu().numpy()
    return A, [tok_label(t) for t in ids]


sentences = [
    "The cat sat on the mat.",
    "Paris is the capital of France.",
    "She quickly realized that the answer was completely wrong.",
]

for si, sent in enumerate(sentences):
    ids = enc.encode_ordinary(sent)
    A, toks = depth_attention(ids)
    T = len(ids)
    readout = A[:, :, -1, :].mean(axis=1)           # (T, S) per-token readout row
    M = A.mean(axis=(0, 1))                          # (S, S) mean matrix

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(15, max(4.2, 0.55 * T + 1.5)),
        gridspec_kw={'width_ratios': [1.25, 1]})

    im0 = ax0.imshow(readout, aspect='auto', cmap='magma', vmin=0)
    ax0.set_xticks(range(S)); ax0.set_xticklabels(col_labels, fontsize=8)
    ax0.set_yticks(range(T)); ax0.set_yticklabels(toks, fontsize=9)
    ax0.set_xlabel('depth position attended  (h0=embedding … h12=top)')
    ax0.set_ylabel('token  (prediction made here)')
    ax0.set_title(f'Readout attention per token\n"{sent}"', fontsize=10)
    for i in range(T):
        for j in range(S):
            v = readout[i, j]
            if v >= 0.06:
                ax0.text(j, i, f'{v*100:.0f}', ha='center', va='center',
                         color='white' if v < readout.max() * 0.6 else 'black', fontsize=6)
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label='attention')

    im1 = ax1.imshow(M, cmap='magma', vmin=0)
    ax1.set_xticks(range(S)); ax1.set_xticklabels(col_labels, fontsize=8)
    ax1.set_yticks(range(S)); ax1.set_yticklabels(col_labels, fontsize=8)
    ax1.set_xlabel('key: layer attended')
    ax1.set_ylabel('query: layer position')
    ax1.set_title('Vertical attention matrix (S×S)\nmean over tokens & heads', fontsize=10)
    for i in range(S):
        for j in range(S):
            v = M[i, j]
            if v >= 0.06:
                ax1.text(j, i, f'{v*100:.0f}', ha='center', va='center',
                         color='white' if v < M.max() * 0.6 else 'black', fontsize=5.5)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='attention')

    plt.tight_layout()
    out = f'{viz_dir}/vertical_sent{si}.png'
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig)
    print('wrote', out)

# ---------------------------------------------------------------- per-head specialization
val = np.memmap('data/openwebtext/val.bin', dtype=np.uint16, mode='r')
B, T = 32, 128
ix = torch.randint(len(val) - T, (B,))
x = torch.stack([torch.from_numpy(val[i:i + T].astype(np.int64)) for i in ix]).to(device)
with torch.no_grad():
    pos = torch.arange(T, device=device)
    h = model.transformer.drop(model.transformer.wte(x) + model.transformer.wpe(pos))
    hs = [h]
    for blk in model.transformer.h:
        h = blk(h); hs.append(h)
    _ = model.vertical(torch.stack(hs, dim=2))
A = attn.attn_weights.float().view(B * T, -1, S, S)
per_head = A[:, :, -1, :].mean(dim=0).cpu().numpy()     # (n_head, S)

fig, ax = plt.subplots(figsize=(11, 6))
im = ax.imshow(per_head, aspect='auto', cmap='magma', vmin=0)
ax.set_xticks(range(S)); ax.set_xticklabels(col_labels, fontsize=8)
ax.set_yticks(range(per_head.shape[0])); ax.set_yticklabels([f'head {i}' for i in range(per_head.shape[0])], fontsize=8)
ax.set_xlabel('depth position attended  (h0=embedding … h12=top)')
ax.set_title('Per-head readout attention over depth (mean over a val batch)\n'
             'different heads read different layers', fontsize=11)
for i in range(per_head.shape[0]):
    for j in range(S):
        v = per_head[i, j]
        if v >= 0.05:
            ax.text(j, i, f'{v*100:.0f}', ha='center', va='center',
                    color='white' if v < per_head.max() * 0.6 else 'black', fontsize=6)
fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, label='attention')
plt.tight_layout()
plt.savefig(f'{viz_dir}/vertical_per_head.png', dpi=130, bbox_inches='tight'); plt.close(fig)
print(f'wrote {viz_dir}/vertical_per_head.png')

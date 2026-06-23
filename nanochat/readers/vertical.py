"""VerticalReader: a small bidirectional transformer over the depth (layer) axis.

Per token position, independently:
  ladder [x0, h_1..h_L]  --RMSNorm per rung-->  down-proj (n_embd -> dV)  (+ depth pos emb)
    --> reader_layers bidirectional blocks over the N = L+1 rungs (dV, multi-head, ReLU^2 MLP)
    --> query pool: one learned query attends over the N output rungs -> r_V in R^dV
    --> up-proj (dV -> n_embd)
GPT then applies its shared final norm + lm_head, so V *owns* the readout: no h_L skip,
no gate, no identity init. The 128-dim dV bottleneck throttles V on input and output (see
the spec): widening only up_proj cannot recover rank that down_proj already discarded.

Reuses nanochat's `Linear` (master weights fp32, matmul in activation dtype) and `norm`
(parameter-free RMSNorm) so reader numerics match the backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import Linear, norm
from nanochat.readers.base import BaseReader

# CUDA caps grid dimensions at 65535. SDPA folds the (B*T) batch onto a grid axis, so the
# reader's depth-attention must tile that batch to keep every kernel launch under the cap
# (B*T = 32*2048 = 65536 at the d10 default trips "invalid configuration argument").
# 8192 stays safe for reader_heads up to 8; the rung count S is tiny, so extra launches are free.
_SDPA_BATCH_CHUNK = 8192


class VBlock(nn.Module):
    """Pre-norm bidirectional transformer block over the rung axis, width dV."""

    def __init__(self, dim, n_heads, mlp_mult):
        super().__init__()
        assert dim % n_heads == 0, f"reader_dim {dim} not divisible by reader_heads {n_heads}"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.c_q = Linear(dim, dim, bias=False)
        self.c_k = Linear(dim, dim, bias=False)
        self.c_v = Linear(dim, dim, bias=False)
        self.c_proj = Linear(dim, dim, bias=False)
        self.c_fc = Linear(dim, mlp_mult * dim, bias=False)
        self.mlp_proj = Linear(mlp_mult * dim, dim, bias=False)

    def _attn(self, z):
        M, S, D = z.shape  # M = folded batch (B*T), S = rungs (=n_layer+1), D = dim
        q = self.c_q(z).view(M, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(z).view(M, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(z).view(M, S, self.n_heads, self.head_dim).transpose(1, 2)
        # Bidirectional (non-causal) attention over the S rungs (tiny: S = n_layer+1). SDPA
        # maps the folded batch M = B*T onto a CUDA grid axis capped at 65535, so a single
        # launch with M > 65535 dies ("invalid configuration argument"). Tile M under the cap;
        # attention is independent per batch row, so chunking is exactly equivalent.
        if M <= _SDPA_BATCH_CHUNK:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            y = torch.cat([
                F.scaled_dot_product_attention(
                    q[i:i + _SDPA_BATCH_CHUNK], k[i:i + _SDPA_BATCH_CHUNK],
                    v[i:i + _SDPA_BATCH_CHUNK], is_causal=False)
                for i in range(0, M, _SDPA_BATCH_CHUNK)
            ], dim=0)
        y = y.transpose(1, 2).contiguous().view(M, S, D)
        return self.c_proj(y)

    def _mlp(self, z):
        return self.mlp_proj(F.relu(self.c_fc(z)).square())

    def forward(self, z):
        z = z + self._attn(norm(z))
        z = z + self._mlp(norm(z))
        return z


class VerticalReader(BaseReader):
    needs_ladder = True

    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.dim = config.reader_dim
        self.n_rungs = config.n_layer + 1
        self.n_heads = config.reader_heads
        self.mlp_mult = config.reader_mlp_mult
        self.down = Linear(self.n_embd, self.dim, bias=False)              # 640 -> dV, shared across rungs
        self.depth_pos = nn.Parameter(torch.zeros(self.n_rungs, self.dim)) # real init in init_weights()
        self.blocks = nn.ModuleList([
            VBlock(self.dim, self.n_heads, self.mlp_mult) for _ in range(config.reader_layers)
        ])
        self.q = nn.Parameter(torch.zeros(self.dim))                       # query-pool query; real init below
        self.up = Linear(self.dim, self.n_embd, bias=False)               # dV -> 640
        self._last_w = None  # cache of query-pool attention (B,T,N) for diagnostics

    @torch.no_grad()
    def init_weights(self):
        s_in = 3 ** 0.5 * self.n_embd ** -0.5   # match nanochat matrix init (uniform, fan-in n_embd)
        s_v = 3 ** 0.5 * self.dim ** -0.5       # fan-in dV
        torch.nn.init.uniform_(self.down.weight, -s_in, s_in)
        torch.nn.init.zeros_(self.depth_pos)
        for blk in self.blocks:
            torch.nn.init.uniform_(blk.c_q.weight, -s_v, s_v)
            torch.nn.init.uniform_(blk.c_k.weight, -s_v, s_v)
            torch.nn.init.uniform_(blk.c_v.weight, -s_v, s_v)
            torch.nn.init.zeros_(blk.c_proj.weight)                        # blocks start as identity over rungs
            torch.nn.init.uniform_(blk.c_fc.weight, -s_v * 0.4, s_v * 0.4)
            torch.nn.init.zeros_(blk.mlp_proj.weight)
        torch.nn.init.zeros_(self.q)                                       # q=0 -> uniform (mean) pool at init
        # up_proj is NOT zero-initialized: a zero readout would make norm(0) a singular point.
        # A small random readout keeps the model "below baseline, climbing" but non-degenerate.
        torch.nn.init.uniform_(self.up.weight, -s_v, s_v)

    def readout(self, ladder):
        # ladder: list of (B,T,n_embd), length n_rungs ([x0, h_1..h_L])
        R = torch.stack([norm(h) for h in ladder], dim=2)                  # (B,T,N,d), per-rung RMSNorm
        z = self.down(R) + self.depth_pos.to(R.dtype)                      # (B,T,N,dV)
        B, T, S, D = z.shape
        z = z.view(B * T, S, D)
        for blk in self.blocks:
            z = blk(z)                                                     # (B*T,N,dV), mixes only across rungs
        qd = self.q.to(z.dtype)
        scores = (z @ qd) * (D ** -0.5)                                    # (B*T,N) query-pool logits over rungs
        w = F.softmax(scores, dim=-1)
        r = (w.unsqueeze(-1) * z).sum(dim=1)                               # (B*T,dV) pooled rung
        self._last_w = w.view(B, T, S).detach()
        out = self.up(r).view(B, T, self.n_embd)                          # (B,T,d)
        return out

    def matrix_parameters(self):
        params = [self.down.weight, self.up.weight]
        for blk in self.blocks:
            params += [blk.c_q.weight, blk.c_k.weight, blk.c_v.weight,
                       blk.c_proj.weight, blk.c_fc.weight, blk.mlp_proj.weight]
        return params

    def adamw_parameters(self):
        return [self.depth_pos, self.q]

    def attn_weights(self):
        return self._last_w

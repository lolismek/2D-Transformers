"""WideReader: a full-width depth reader (d_V = n_embd), with NO up/down bottleneck.

At d_V = n_embd the Phase-A VerticalReader's projections become redundant square matrices:
  - `down` (n_embd -> d_V) is absorbable into the first depth-block's input projections
    (c_q/c_k/c_v act on norm(z)), so at full width it adds params + 11x/token FLOPs for no
    extra expressiveness.
  - `up` (d_V -> n_embd) is a square map sitting just before GPT's shared final norm + lm_head,
    which already own the readout (not perfectly absorbable due to the RMSNorm between them, but
    marginal capacity the lm_head covers).
So this reader reads the residual ladder at NATIVE width n_embd and drops both:

  ladder [x0, h_1..h_L]  --RMSNorm per rung-->  (+ depth_pos)
    --> reader_layers bidirectional blocks over the N=L+1 rungs (width n_embd, multi-head, ReLU^2 MLP)
    --> query pool: one learned query attends over the N output rungs -> r in R^{n_embd}
    --> (GPT applies its shared final norm + lm_head)

V owns the readout: no h_L skip, no gate, no identity init (q=0 -> uniform mean-pool at init, so
the model starts below baseline and climbs). Reuses VBlock from nanochat.readers.vertical (it is
width-agnostic) and nanochat's parameter-free `norm`, so reader numerics match the backbone.

COMPUTE: the blocks now run at n_embd^2 over all 11 rungs, so the reader's TRUE fwd+bwd FLOPs are
~2x the backbone. estimate_flops() undercounts this ~11x (it is blind to the rung axis), so run
iso-FLOP by setting --num-iterations explicitly rather than --target-flops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import norm
from nanochat.readers.base import BaseReader
from nanochat.readers.vertical import VBlock  # width-agnostic; reused, vertical.py left untouched


class WideReader(BaseReader):
    needs_ladder = True

    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.dim = config.n_embd                       # full width; config.reader_dim is ignored on purpose
        self.n_rungs = config.n_layer + 1
        # Match H's head_dim (=128) so the reader's attention granularity mirrors the backbone:
        # n_heads = n_embd // 128 -> 5 heads at d10 (n_embd=640, exactly H's n_head=5) and 6 at the
        # throwaway d12 reference base_train builds for sizing (n_embd=768); both divide cleanly.
        # (A fixed head COUNT of 5 would crash the d12 ref: 768 % 5 != 0.) config.reader_heads unused.
        HEAD_DIM = 128
        assert self.dim % HEAD_DIM == 0, f"n_embd {self.dim} must be a multiple of head_dim {HEAD_DIM}"
        self.n_heads = self.dim // HEAD_DIM
        self.mlp_mult = config.reader_mlp_mult
        self.depth_pos = nn.Parameter(torch.zeros(self.n_rungs, self.dim))  # real init in init_weights()
        self.blocks = nn.ModuleList([
            VBlock(self.dim, self.n_heads, self.mlp_mult) for _ in range(config.reader_layers)
        ])
        self.q = nn.Parameter(torch.zeros(self.dim))   # query-pool query; real init below
        self._last_w = None  # cache of query-pool attention (B,T,N) for diagnostics

    @torch.no_grad()
    def init_weights(self):
        s_v = 3 ** 0.5 * self.dim ** -0.5              # match nanochat matrix init (uniform, fan-in dim)
        torch.nn.init.zeros_(self.depth_pos)
        for blk in self.blocks:
            torch.nn.init.uniform_(blk.c_q.weight, -s_v, s_v)
            torch.nn.init.uniform_(blk.c_k.weight, -s_v, s_v)
            torch.nn.init.uniform_(blk.c_v.weight, -s_v, s_v)
            torch.nn.init.zeros_(blk.c_proj.weight)    # blocks start as identity over rungs
            torch.nn.init.uniform_(blk.c_fc.weight, -s_v * 0.4, s_v * 0.4)
            torch.nn.init.zeros_(blk.mlp_proj.weight)
        torch.nn.init.zeros_(self.q)                   # q=0 -> uniform (mean) pool at init
        from nanochat.common import print0             # self-document the resolved architecture in the run log
        print0(f"[WideReader] width={self.dim}  n_heads={self.n_heads} (head_dim={self.dim // self.n_heads})  "
               f"layers={len(self.blocks)}  mlp_mult={self.mlp_mult}  rungs={self.n_rungs}  (no down/up)")

    def readout(self, ladder):
        # ladder: list of (B,T,n_embd), length n_rungs ([x0, h_1..h_L])
        R = torch.stack([norm(h) for h in ladder], dim=2)   # (B,T,N,d), per-rung RMSNorm, native width
        z = R + self.depth_pos.to(R.dtype)                  # (B,T,N,d)  -- NO down-projection
        B, T, S, D = z.shape
        z = z.view(B * T, S, D)
        for blk in self.blocks:
            z = blk(z)                                      # (B*T,N,d), mixes only across rungs
        qd = self.q.to(z.dtype)
        scores = (z @ qd) * (D ** -0.5)                     # (B*T,N) query-pool logits over rungs
        w = F.softmax(scores, dim=-1)
        r = (w.unsqueeze(-1) * z).sum(dim=1)                # (B*T,d) pooled rung
        self._last_w = w.view(B, T, S).detach()
        return r.view(B, T, self.n_embd)                    # NO up-projection; GPT's norm+lm_head own it

    def matrix_parameters(self):
        params = []
        for blk in self.blocks:
            params += [blk.c_q.weight, blk.c_k.weight, blk.c_v.weight,
                       blk.c_proj.weight, blk.c_fc.weight, blk.mlp_proj.weight]
        return params

    def adamw_parameters(self):
        return [self.depth_pos, self.q]

    def attn_weights(self):
        return self._last_w

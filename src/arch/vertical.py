"""
Vertical (over-layers) transformer -- the V1 architecture.

For each token position the base transformer produces a stack of per-layer hidden states
[h_0, h_1, ..., h_L] (S = n_layer + 1 positions; h_0 is the input embedding). This combiner
treats that stack as a sequence, adds a learned depth positional embedding, attends
BIDIRECTIONALLY across the layers with `n_vertical_layer` blocks (shared across all token
positions), and reads out the refined TOP-layer position -- which then feeds ln_f + lm_head in
place of the usual top-layer hidden state.

Init-parity: depth_pos_emb is zero-init and each block's residual output projections are
zero-init (init_identity, called by GPT.__init__), so at initialization this module is the exact
identity on the stack and returns h_L unchanged -> the full model matches vanilla GPT bit-for-bit
until the vertical weights start to move during training.

State-dict namespace: registered under STATE_KEY 'vertical' (keys vertical.depth_pos_emb,
vertical.blocks.*), which is also what pre-existing V1 checkpoints used.
"""

import torch
import torch.nn as nn

from model import LayerNorm, SelfAttention, MLP


class VerticalBlock(nn.Module):
    """ Pre-LN transformer block that attends BIDIRECTIONALLY over the depth (layer) axis.
    Same structure as the base Block but with non-causal attention and a configurable MLP ratio. """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config, is_causal=False, n_head=config.n_vertical_head)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config, mlp_ratio=config.vertical_mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class VerticalCombiner(nn.Module):
    STATE_KEY = 'vertical'   # checkpoint namespace (also matches pre-existing V1 checkpoints)
    needs_stack = True       # forward consumes the full per-layer stack H

    def __init__(self, config):
        super().__init__()
        self.n_stack = config.n_layer + 1                       # S = number of depth positions
        self.depth_pos_emb = nn.Parameter(torch.zeros(self.n_stack, config.n_embd))
        self.blocks = nn.ModuleList([VerticalBlock(config) for _ in range(config.n_vertical_layer)])

    def forward(self, H):
        # H: (B, T, S, d) stack of per-layer hidden states
        B, T, S, d = H.size()
        z = H + self.depth_pos_emb                              # (S, d) broadcasts over (B, T, S, d)
        z = z.view(B * T, S, d)                                 # each token is its own depth-sequence
        for block in self.blocks:
            z = block(z)
        z = z.view(B, T, S, d)
        return z[:, :, -1, :]                                   # readout = refined top-layer (h_L)

    def init_identity(self):
        """Zero the residual output projections + depth pos-emb so this module is the exact identity
        on the stack at init (the full model then matches vanilla GPT-2 bit-for-bit)."""
        torch.nn.init.zeros_(self.depth_pos_emb)
        for block in self.blocks:
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if block.attn.c_proj.bias is not None:
                torch.nn.init.zeros_(block.attn.c_proj.bias)
            if block.mlp.c_proj.bias is not None:
                torch.nn.init.zeros_(block.mlp.c_proj.bias)

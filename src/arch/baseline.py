"""
Baseline architecture: the vanilla GPT readout -- just take the top layer.

This is parameter-less (STATE_KEY=None) and needs_stack=False, so the base GPT runs its native
residual stream and reads out the top layer directly, i.e. bit-for-bit vanilla GPT-2. The
forward() below is only here for completeness / uniformity; the fast path in GPT.forward never
calls it. It exists so 'baseline' is a first-class entry in the architecture registry.
"""

import torch.nn as nn


class BaselineCombiner(nn.Module):
    STATE_KEY = None      # parameter-less -> no checkpoint namespace
    needs_stack = False   # GPT uses its native top-layer readout (no per-layer stack built)

    def __init__(self, config):
        super().__init__()

    def forward(self, H):
        # H: (B, T, S, d) -> top layer (B, T, d)
        return H[:, :, -1, :]

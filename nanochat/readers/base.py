"""Pluggable depth-readers ("vertical" architectures) for nanochat.

A *reader* consumes the per-block residual **ladder** ``[x0, h_1, ..., h_L]`` that
``GPT.forward`` stashes, and returns the pre-final-norm readout vector
``(B, T, n_embd)`` that GPT feeds into its shared final norm + lm_head. This keeps
depth-combination architectures swappable without touching the GPT core beyond one
tiny, architecture-agnostic hook.

To add a new architecture: subclass ``BaseReader``, implement the methods below, and
register it in ``nanochat/readers/__init__.py`` ``REGISTRY``. Nothing in ``gpt.py`` or
``base_train.py`` needs to change for a new reader.
"""

import torch
import torch.nn as nn


class BaseReader(nn.Module):
    # If True, GPT.forward stashes the full ladder [x0, h_1..h_L] and passes it to readout().
    needs_ladder = True

    @torch.no_grad()
    def init_weights(self):
        """Initialize parameters. Called from GPT.init_weights() after to_empty(device)."""
        raise NotImplementedError

    def readout(self, ladder):
        """ladder: list of (B,T,n_embd) tensors, length n_layer+1 ([x0, h_1..h_L]).
        Returns the readout (B,T,n_embd); GPT applies its shared final norm + lm_head."""
        raise NotImplementedError

    # --- optimizer plumbing (kept here so GPT.setup_optimizer stays generic) -----------
    def matrix_parameters(self):
        """2D weight matrices that should be optimized by Muon."""
        raise NotImplementedError

    def adamw_parameters(self):
        """Embedding-like / scalar params that should be optimized by AdamW."""
        raise NotImplementedError

    # --- diagnostics -------------------------------------------------------------------
    def attn_weights(self):
        """Optional: most recent depth-attention over the rungs (B,T,n_rungs), or None."""
        return None

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
    # Subclasses call self._init_gate(config) in __init__ to create self.gate (None when ungated, an
    # nn.Parameter when gated). NOTE: do NOT declare `gate` as a class attribute -- nn.Module's
    # register_parameter rejects assigning a Parameter to a name that already exists. All access is
    # getattr(self, "gate", None)-guarded so readers that never call _init_gate are still safe.

    @torch.no_grad()
    def init_weights(self):
        """Initialize parameters. Called from GPT.init_weights() after to_empty(device)."""
        raise NotImplementedError

    def readout(self, ladder):
        """ladder: list of (B,T,n_embd) tensors, length n_layer+1 ([x0, h_1..h_L]).
        Returns the readout (B,T,n_embd); GPT applies its shared final norm + lm_head."""
        raise NotImplementedError

    # --- additive gate (shared across readers) -----------------------------------------
    def _init_gate(self, config):
        """Create the additive-gate parameter per config.reader_gate. Call from subclass __init__.
        none -> ungated (gate stays None); scalar -> one shared g; channel -> per-channel g (n_embd).
        The actual value is set in the subclass init_weights() (re-init'd after GPT's to_empty)."""
        mode = getattr(config, "reader_gate", "none")
        if mode == "none":
            self.gate = None
        elif mode == "scalar":
            self.gate = nn.Parameter(torch.zeros(1))
        elif mode == "channel":
            self.gate = nn.Parameter(torch.zeros(config.n_embd))
        else:
            raise ValueError(f"unknown reader_gate {mode!r} (expected none|scalar|channel)")

    def combine(self, base, r):
        """Fuse the reader output r with the baseline readout base.
        Ungated (gate is None): the reader owns the readout -> return r (prior behavior).
        Gated: return base + gate * r, so gate=0 reproduces the baseline readout exactly."""
        if getattr(self, "gate", None) is None:
            return r
        return base + self.gate.to(r.dtype) * r

    def gate_parameters(self):
        """The gate parameter(s), for a dedicated optimizer group; empty when ungated."""
        return [] if getattr(self, "gate", None) is None else [self.gate]

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

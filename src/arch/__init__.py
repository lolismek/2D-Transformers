"""
Architecture registry: each "architecture" is a depth-combination head that maps the base
model's per-layer stack H = (B, T, S, d) -> a readout (B, T, d), in place of the vanilla
"take the top layer". Pick one with `config.arch = '<name>'`.

To add a new architecture (e.g. a static learned per-layer weighting):
  1. create src/arch/static.py defining a combiner nn.Module with:
       - class attr STATE_KEY    : str | None   (checkpoint namespace; None if parameter-less)
       - class attr needs_stack  : bool          (does forward need the full per-layer stack?)
       - forward(self, H)        : (B,T,S,d) -> (B,T,d)
       - optional init_identity(self): re-init so the full model starts at the baseline
  2. register it in REGISTRY below.
No edits to model.py / train.py are needed -- the base GPT picks it up by name.
"""

from .baseline import BaselineCombiner
from .vertical import VerticalCombiner

# name -> combiner class
REGISTRY = {
    'baseline': BaselineCombiner,   # native top-layer readout (vanilla GPT-2)
    'vertical': VerticalCombiner,   # the vertical (over-layers) transformer
}


def build_combiner(config):
    """Instantiate the depth-combiner selected by config.arch."""
    if config.arch not in REGISTRY:
        raise ValueError(f"unknown arch {config.arch!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[config.arch](config)

"""H-backbone architecture variants (ours, gated on GPTConfig.h_residual).

Stock nanochat keeps two skip connections per block (around attn, around mlp) and the trunk
re-injects the initial embedding x0 at every layer (x0_lambdas). Together these make every depth
rung a partial sum of one telescoping series -- h_L = x0 + sum of per-layer deltas -- so the top
state h_L already holds the whole sum, and a depth-reader V that re-aggregates the ladder is largely
redundant (this is the mechanism behind every "the ladder adds little over h_L" result so far).

`h_residual="none"` removes that free additive aggregator from H so each rung becomes a *pure deep
transform* of the previous one rather than a partial sum:
  * ResidualFreeBlock (here) drops both in-block skips:  x = attn(norm(x)); x = mlp(norm(x))
  * the x0 injection is dropped in GPT.forward (it lives in the trunk loop, not the block; gated on
    config.h_residual). The learned per-layer gain resid_lambdas is KEPT -- it scales the forward
    signal, it is not an identity/skip path.
  * GPT.init_weights initializes the output projections (attn.c_proj, mlp.c_proj) at fan-in scale
    instead of zeros: with no skip to carry the signal, zero-init would make every block emit 0 at
    init (and relu^2'(0)=0), so the whole trunk would output 0 and no gradient would flow.
  * the residual-free baseline (reader=none) additionally skips the `backout` subtraction, which
    assumes the partial-sum structure (it subtracts a mid-layer partial sum).

Only H is made residual-free; the depth-reader V keeps its own residual connections.

Baseline parity: with h_residual="full" none of this is constructed or run, so the stock nanochat
path is byte-identical.
"""
from nanochat.gpt import Block, norm


class ResidualFreeBlock(Block):
    """A nanochat Block with both skip connections removed (attn + mlp).

    Stock:  x = x + attn(norm(x)); x = x + mlp(norm(x))
    Here:   x =     attn(norm(x)); x =     mlp(norm(x))

    Same submodules, parameters, and shapes as Block (so checkpoints and the optimizer param split
    are unchanged) -- only the two residual adds in forward are gone, so each block fully transforms
    its input instead of refining it.
    """

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = self.mlp(norm(x))
        return x

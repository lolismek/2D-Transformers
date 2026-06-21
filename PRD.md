# PRD: The 2D Transformer

*A second transformer applied over the depth (layer) axis of a standard transformer.*

## Summary

A standard transformer mixes information across the **sequence** with attention, but across **depth** it only combines layers via residual connections — a fixed, additive accumulation. This project explores replacing that fixed depth-wise combination with a *second transformer that attends over the layer axis*. The first transformer runs "horizontally" across tokens as usual; a second transformer runs "vertically" across each token's stack of per-layer hidden states.

## Motivation

Residual connections are the depth-axis analogue of an RNN's fixed recurrence over tokens: each layer's output is summed into a shared stream rather than selectively combined. They work, but they represent only one way to merge layers — additive, with fixed weights, and strictly bottom-up. If attention was the right replacement for fixed recurrence across tokens, it is worth asking whether attention is also a better way to combine information across layers.

## Core idea

For each token position, a transformer produces a stack of hidden states — one per layer. The 2D transformer treats that per-token stack as a sequence along the **depth axis** and applies a second transformer over it. The same second transformer is shared across all token positions.

Two properties distinguish this from a residual stream:
- **Learned, content-based combination** across layers, instead of fixed additive accumulation.
- **Bidirectional flow across depth** — because the second transformer can attend over the whole stack, later-layer information can inform earlier-layer representations, which a standard stack cannot do.

## Hypotheses

- Learned, input-dependent combination across layers is a more expressive generalization of residual connections and of static layer-weighting schemes.
- Bidirectional cross-depth flow may let the model refine early-layer representations using late-layer context.
- *(Longer term)* the second transformer might act as a discardable training scaffold that makes early layers more useful, potentially enabling depth reduction at inference.

## Relation to prior work

Cross-depth information flow is an active area. AttnRes (Kimi, 2026) replaces additive residual accumulation with causal attention over preceding layers; Depth-Attention (2026) mixes earlier-layer values inside the attention module; DenseFormer uses depth-weighted averaging between blocks. This project's distinguishing angles: (a) a **separate, full** second transformer over the depth axis rather than a mechanism interleaved into the main stack; (b) **bidirectional** aggregation across all layers rather than causal/backward-only; and (c) treating the construct as an **instrument for studying layer usefulness**, not only as an architecture for benchmark gains. Existing results suggest the performance effect of better layer-combination is real but likely modest — so characterization matters as much as raw gains.

## V1 scope

Build the simplest end-to-end working version that lets us observe behavior.

- Start from the **nanoGPT** repo (`karpathy/nanoGPT`) as the base codebase.
- Train a small model **from scratch**, jointly optimizing the base (horizontal) transformer and the second (vertical, over-layers) transformer together.
- Small model, small dataset, single-GPU scale — fast enough to iterate.
- Architecture specifics, integration details, tensor layouts, and hyperparameters are left to implementation to decide.

## V1 success criteria

- The combined model trains stably from scratch.
- It can be compared against a vanilla nanoGPT baseline of comparable size/compute on the same data.
- We can see whether the second transformer changes training/validation loss relative to baseline, and ideally inspect what the vertical transformer learns (e.g., which layers it attends to).

## Out of scope for V1 (future directions)

- Frozen-base / probe-only variants.
- The discardable-scaffold + layer-truncation training schedule.
- An "oracle ladder" of aggregators (static weighting → linear → attention → full transformer) and dedicated interpretability readout studies.
- Scaling-law or large-scale runs.

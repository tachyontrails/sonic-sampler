# Quickstart

Getting a first sampler running, and choosing the level of the API to build against.

Prerequisites: [`concepts.md`](concepts.md) for the buffer model and the integration contracts.

## Installation

```
uv pip install sonic-sampler
```

Requires Python 3.12+, Triton 3.5.1, and an SM90+ device. The tuned dispatch tables shipped under
`resources/` cover those architectures.

## The adoption ladder

Three rungs expose the same kernels at decreasing levels of responsibility.

| Rung | Entry point | You own | You get |
| :--- | :--- | :--- | :--- |
| Functional | `fused_singular`, `fused_multistep` | Every buffer, every tuning knob, every tensor argument | Direct access to the fused pipeline |
| Tiling | `TwoStageTiling` | Buffer population and the call itself | Scratchpad allocation, tuned config resolution |
| Modular | `UnshardedFusedSingular`, `UnshardedFusedMultistep` | Request state, through `SamplingBuffers` | Everything above, plus argument marshalling |

The modular rung is the recommended path, and pairing it with `SamplingBuffers` is the intended
integration shape. The rest of this document, and both batching guides, assume it.

Drop to the tiling rung when your engine already owns its parameter tensors and only wants the
tuned dispatch and scratchpad management. Drop to the functional rung when you need control over
buffer placement or want to bypass tuned dispatch entirely. Both are covered in
[`reference.md`](reference.md).

## Initialization

Two objects are constructed once, during engine startup, and live for the process lifetime.

```python
import torch

from sonic_sampler.core import SamplingBuffers
from sonic_sampler.interface import UnshardedFusedSingular, UnshardedFusedMultistep


device = torch.device("cuda:0")

buffers = SamplingBuffers.default(
    size=max_batch_size,            # B
    timesteps=lookahead + 1,        # γ + 1
    vocab_size=model.vocab_size,    # V
    device=DEVICE,
)

sampler = UnshardedFusedSingular.initialize(
    vocab_size=model.vocab_size,
    batch_size=max_batch_size,
    lookahead=lookahead,
)

verifier = UnshardedFusedMultistep.initialize(
    vocab_size=model.vocab_size,
    lookahead=lookahead,
    batch_size=max_batch_size,
)
```

A non-speculative engine constructs `buffers` with `timesteps=1`, initializes `sampler` with
`lookahead=0`, and skips `verifier` entirely.

### What `default` leaves behind

A freshly constructed `SamplingBuffers` is already a valid neutral state, so a request that
specifies no sampling parameters needs no writes at all:

| Member | Initial value | Effect |
| :--- | :--- | :--- |
| `temperature` | $1$ | Unscaled logits |
| `top_k` | $128$ under `bounded=True`, else $V$ | Bounded truncation |
| `top_p` | $1$ | Inactive |
| `min_p` | $0$ | Inactive |
| `bias` | $0$ | Inactive |
| `grammar` | all bits set | Unconstrained |
| `repetition.*` | neutral | Inactive |

`bounded=False` widens `top_k` to $V$, which is only meaningful at the functional rung with an
unbounded selection path. Leave it at the default.

### What `initialize` resolves

`initialize` performs the tuned dispatch lookup and allocates the in-flight buffers:

1. Load the benchmark table for the top-$`k`$ bound and the device architecture, i.e.
   `resources/sm{90,100}_k128.toml`.
2. Match a `VocabBucket` against $V$, which fixes the default `block_n` and therefore the number of
   vocabulary tiles.
3. Allocate the `scratchpad`, plus `values` and `indices` for the verifier, sized to
   $B \cdot (\gamma + 1)$ rows.

Each call then narrows further. `self.tuning(n_rows, ...)` matches a `BatchBucket` against the row
count of the current launch and yields the warp configuration, the top-$`k`$ strategy, and a possibly
revised `block_n`. Passing an explicit `strategy` or `tuning` overrides the resolution.

Initialization raises `ValueError` when the architecture has no shipped table, when no bucket
matches $V$, when $k$ exceeds $128$ or is not a power of two, or when $V$ fits in a single
vocabulary tile. Supplying `block_size` bypasses the table lookup and pins the tile width; supply
`arch` to resolve against a device other than the current one.

## A minimal decode step

With the buffers populated for the active batch, a single autoregressive step is one call:

```python
selection = sampler.autoregressive(
    logits,                     # [ R, V ], bfloat16
    buffers,
    prefill=False,
    update_counts=True,
    logprobs=True,
)

tokens = selection.tokens       # [ R, 1 ], int32
```

`prefill=True` marks the step immediately following prefill, where the decode histogram is empty.
`update_counts=True` folds the selected tokens into `repetition.counts.decode` inside the kernel,
which is what keeps repetition penalties current without a separate pass. Leave it off when the
engine maintains its own histograms.

Under just-in-time batching the kernel writes that histogram into a row of a buffer that the next
iteration reuses for a different request, so the updated `counts.decode` rows must be cloned back
into the engine's per-request metadata after the pass, exactly as the draft weights are. This is
the same feedback pattern in both cases, and [`just-in-time.md`](just-in-time.md) covers it in
full. Slot-based integrations are unaffected, since the histogram already lives at the request's
own slot.

Populating the buffers for that batch is the part that depends on your engine:

- Requests hold a stable slot for their lifetime, and the batch is addressed through a
  `slot_mapping`: see [`slot-based.md`](slot-based.md).
- The batch is composed immediately before the forward pass and the buffers are filled in that
  order: see [`just-in-time.md`](just-in-time.md).

Both guides cover admission, per-iteration upkeep, the full speculative loop, and CUDA graph
capture for their respective strategy.

## Validating an integration

`NaiveSampler` is a pure PyTorch implementation of the same semantics, and `SamplingBuffers.random`
generates a populated instance with a mix of indicators:

```python
from sonic_sampler.core import SamplingBuffers, NaiveSampler

buffers = SamplingBuffers.random(
    size=8,
    timesteps=lookahead + 1,
    vocab_size=model.vocab_size,
    device=device,
    seed=0,
)

reference = NaiveSampler(buffers=buffers)

expected = reference.singular(logits=logits, tokens=None, timestep=0, logprobs=True)
actual = sampler.drafting(logits, buffers, timestep=0)
```

Driving both paths with the same buffers and the same logits pins down the expected tokens and
log-probabilities. Match the scope when comparing: `NaiveSampler.singular` reads the draft weights
and mirrors `drafting(...)`, while `NaiveSampler.multistep` reads the target weights and mirrors
the verifier. Both helpers exist for correctness work rather than production use, so an integration
should not depend on them at runtime.

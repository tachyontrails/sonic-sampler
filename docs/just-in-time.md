# Just-In-Time Batching

Integrating SonicSampler into an engine that composes its batch immediately before each forward
pass.

Prerequisites: [`concepts.md`](concepts.md) and [`quickstart.md`](quickstart.md).

## When this applies

This is the right guide when the scheduler selects a set of requests per iteration, the batch order
is whatever the scheduler produced, and per-request sampling state lives in the engine's own
metadata objects rather than in a fixed-capacity table.

Under this strategy `SamplingBuffers` is a blank template rather than a store. It is repopulated
in batch order before every forward pass, so row $i$ of the logits reads position $i$ of the
buffers and no `slot_mapping` is involved. Only the active prefix takes part, obtained with
`buffers[:rows]`.

The consequence worth internalizing: the buffers are effectively stateless across iterations.
Anything the kernels write that must survive into the next iteration has to be read back into the
engine's per-request metadata, because the row it occupied will belong to a different request.

## Lifecycle

```
per iteration    →  collect per-request state from engine metadata
                      ↓
                    update(...)  populates positions 0 … R - 1
                      ↓
                    forward pass on buffers[:R]
                      ↓
                    read back draft weights and decode counts
```

## Per-iteration population

`update(...)` mirrors `allocate(...)`, with the differences driven by statelessness.

```python
active = buffers[:rows]

event = active.update(
    counts=[(request.context_counts, request.decode_counts) for request in batch],
    target_gumbel=[request.draft_weights for request in batch],
    multiplicative=[request.repetition_penalty for request in batch],
    frequency=[request.frequency_penalty for request in batch],
    presence=[request.presence_penalty for request in batch],
    biases=[request.logit_bias for request in batch],
    temperature=[request.temperature for request in batch],
    top_k=[request.top_k for request in batch],
    top_p=[request.top_p for request in batch],
    min_p=[request.min_p for request in batch],
    top_logprobs=[request.top_logprobs for request in batch],
    prefill=[request.is_prefill for request in batch],
    greedy_drafts=[request.greedy_draft for request in batch],
    bitmasks=[request.bitmask for request in batch],
    generator=[request.generator for request in batch],
    stream=copy_stream,
)
```

Every list is in batch order and has length $R$. `positions` is omitted, which is what selects
sequential population. Passing `positions` is supported for the hybrid case where an otherwise
just-in-time engine wants to touch a subset of rows, but it is not the usual shape.

| Argument | Notes |
| :--- | :--- |
| `counts` | `(context, decode)` pairs of `[ V ]` int32 tensors, copied into the device buffers |
| `target_gumbel` | Per-request `[ γ + 1, V ]` draft weights carried over, or `None` |
| `prefill` | Per-request predicate, or a single broadcast bool |
| `indicators` | Optional list of `(target, draft)` `Indicator` pairs, otherwise derived |
| `greedy_drafts` | Per-request or a single broadcast bool |
| `bitmasks` | Pinned int32 bitmask tensors or `None` per request |
| `generator` | Per-request `torch.Generator` |
| `stream` | Dedicated copy stream, which switches on event recording |

As with admission in the slot-based strategy, the returned `Event` is non-`None` only when a
`stream` was supplied, and the compute stream should wait on it before the forward pass:

```python
if event is not None:

    torch.cuda.current_stream().wait_event(event)
```

Population is issued immediately after the read-back that closes the previous iteration, and shares
its stream, so a single event at the end is all the coordination required:

```
                ── time ──────────────────────────────────────────────────────────────────────────────────────────▶

                                                           ┌───┬───┐
host                                                       │ C │ E │
                                                           └───┴───┘

                                                           ┌───┬───┬───┬───┬───┬───┬───┐
copy stream                                                │ A │ B │ D │ F │ G │ H │ I │ ◆
                                                           └───┴───┴───┴───┴───┴───┴───┘

                ┌────────────────────┬────────┬───────────┐         ┌────────────────────┬───┬────────┬───────────┐
compute stream  │   target forward   │ verify │ γ × draft │         │   target forward   │ ◇ │ verify │ γ × draft │
                └─────────────[ iteration N ]─────────────┘         └─────────────[ iteration N + 1 ]─────────────┘

                ◆  recorded        ◇  compute stream waits on the record
```

| Box | Lane | Work |
| :--- | :--- | :--- |
| `A` | copy | Read back: clone `gumbel.draft[:R]` and `counts.decode[:R]`, then unbind row-wise into the engine's per-request metadata |
| `B` | copy | Grammar bitmasks into `grammar`, already pinned by the caller |
| `C` | host | Resolve the `ScopedIndicators` and pack them into pinned staging |
| `D` | copy | Indicators into `flags.packed` |
| `E` | host | Pack temperature, top-$k$, top-$p$, min-$p$, top logprobs, biases, and repetition penalties into pinned staging |
| `F` | copy | Repetition penalties into their device buffers |
| `G` | copy | The `counts` tuples into `counts.context` and `counts.decode` |
| `H` | copy | Refresh the stochastic weights, then disperse `target_gumbel` into `gumbel.target` |
| `I` | copy | Sampling parameters and biases into their device buffers |

The read-back is issued at the close of iteration $N$ and `update(...)` follows immediately, so `C`
begins alongside `A` while the copy stream is still draining. `D` then follows `C` and `F` follows
`E`, which is the producer-to-dispatch relationship the lanes encode. Because every box shares one
stream, `G` and `H` observe the clones `A` produced without an intermediate event, and the compute
stream's `◇` before `verify` in $N + 1$ is the only synchronization point.

Population is therefore pinned to the boundary between iterations, since `G` and `H` consume what
`A` produced. Slot-based admission is free to run anywhere inside an iteration; this is not. It
still disappears behind the next forward pass, since every box is microseconds against a
millisecond-scale `target forward`. Widths above are indicative rather than proportional.

Two further notes:

- `H` is skipped entirely when no row in the batch has a stochastic target.
- Rows marked `prefill[i]` refresh only `gumbel.target[i, 0]`, so `H` shrinks with a prefill-heavy
  batch.

### Counts instead of encodings

`allocate` takes `encodings`, i.e. raw prompt token ids, and builds the histogram itself.
`update` takes the histograms directly, as `(context, decode)` pairs of `[ V ]` int32 tensors, and
copies them into position $i$.

The engine therefore owns both histograms. The context histogram is built once when the request is
admitted and never changes. The decode histogram grows as tokens are emitted, and if
`update_counts=True` was passed to the sampler then the kernel wrote the newly selected tokens into
`repetition.counts.decode` at the row the request occupied. That row is about to be reused, so the
updated histogram has to be cloned back:

```python
counts = active.repetition.counts.decode.clone()

for request, decode in zip(batch, counts.unbind()):

    request.decode_counts = decode
```

One clone of the active prefix followed by a row-wise unbind is preferable to a clone per row: it
is a single device-to-device copy, and the unbound rows are views into that one allocation.

Engines that maintain their own histograms host-side can leave `update_counts=False` and skip the
read-back entirely.

### Prefill predicates

`allocate` implies `prefill=True`, because a request is admitted exactly once. A just-in-time batch
can mix prefill and decode requests, so the predicate is per request:

| `prefill[i]` | `target` refresh | `draft` refresh |
| :--- | :--- | :--- |
| `True` | Timestep $0$ only | All $\gamma + 1$ timesteps, when the draft is stochastic |
| `False` | All timesteps, plus a fresh `uniform` draw | All timesteps, when the draft is stochastic |

As always, the whole weight update is skipped when no request in the batch has a stochastic target.

### The `target_gumbel` contract

This is the just-in-time counterpart to `step_weights`. Verification is on-policy only when the
verifier sees the same noise the drafter used, and since the draft weights do not survive the
iteration, the engine has to carry them:

1. After the forward pass, clone the draft weights of every request that drafts stochastically:

   ```python
   weights = active.weights.gumbel.draft.clone()

   for request, noise in zip(batch, weights.unbind()):

       if request.stochastic_draft:

           request.draft_weights = noise
   ```

2. Feed them back on the next iteration through `target_gumbel`, where they are copied into the
   target Gumbel noise at the same position.

Three cases determine what each element of the list should be:

| Request regime | `target_gumbel[i]` |
| :--- | :--- |
| Stochastic verify, stochastic draft | The `[ γ + 1, V ]` weights cloned last iteration |
| Stochastic verify, greedy draft | `None`, so the target weights are refreshed instead |
| Greedy verify | `None`, and nothing is refreshed at all |

Passing carried-over weights for a greedy-drafting request would verify against noise the drafter
never used, so the `None` in the second row is load-bearing.

## Grammar

`step_grammar(...)` is available here too, called without `positions` so the bitmask list is
consumed in batch order:

```python
active.step_grammar(
    bitmasks=[request.bitmask_at(timestep) for request in batch],
    timestep=timestep,
)
```

Entries may be `None` for requests without constrained decoding, and the call is a no-op when every
entry is `None`. Omitting `timestep` writes the mask across all $\gamma + 1$ timesteps of the row.

## The speculative loop

The drafter proposes $\gamma$ tokens, the verifier accepts a prefix and appends one more.

```python
tokens = torch.empty(max_batch_size, lookahead + 1, dtype=torch.int32, device=device)

draft_probs = torch.empty(
    (max_batch_size, lookahead, vocab_size), dtype=torch.bfloat16, device=device
)

active = buffers[:rows]

# 1. Draft γ candidate tokens, one timestep at a time.

for timestep in range(lookahead):

    draft_logits = drafter.forward(...)                 # [ R, V ], bfloat16

    selection = sampler.drafting(
        draft_logits,
        active,
        timestep=timestep,
        tokens=tokens[:rows],
        probabilities=draft_probs[:rows, timestep],
        in_place=True,
    )

    active.step_grammar(
        bitmasks=[request.bitmask_at(timestep + 1) for request in batch],
        timestep=timestep + 1,
    )

# 2. Verify the chain against the target model in one call.

verification = verifier(
    target_logits,                                      # [ R, γ + 1, V ], bfloat16
    tokens[:rows],                                      # [ R, γ + 1 ], int32
    active,
    probabilities=draft_probs[:rows],                   # [ R, γ, V ]
    update_counts=True,
    logprobs=True,
    in_place=True,
)

# 3. Read back everything the next iteration needs.

weights = active.weights.gumbel.draft.clone()
counts = active.repetition.counts.decode.clone()

for request, noise, decode in zip(batch, weights.unbind(), counts.unbind()):

    if request.stochastic_draft:

        request.draft_weights = noise

    request.decode_counts = decode
```

Points worth noting:

- No `slot_mapping` appears anywhere. Row $i$ is position $i$ by construction.
- `drafting` reads `flags.draft`, `grammar[:, timestep]`, and `gumbel.draft[:, timestep]`.
- `probabilities` is the draft distribution $q$ the verifier consumes, one contiguous row of width
  $V$ per drafted timestep. It can be omitted when every draft in the batch is greedy.
- The verifier emits $\texttt{offsets}[i] + 1$ tokens for row $i$.
- There is no `step_weights` call. Step 3 plus the next `update(...)` play that role.

## CUDA graph capture

Capture is viable here, with one extra consideration: the kernels are launched against
`buffers[:rows]`, so the sliced views must be the same objects, with the same pointers, every time.

Prefix slices are cached per instance, so `buffers[:rows]` returns the same views once that size
has been sliced at least once. Warm the cache for every batch size you intend to capture before
capturing anything:

```python
buffers.prepopulate(schedule=[1, 2, 4, 8, 16, 32])
```

A schedule of row sizes covers `buffers[:R]`. Engines that also vary the timestep extent can pass
`[(rows, cols), ...]` pairs, or a `(rows, cols)` pair of lists to warm the full cross product.

Inside the captured region:

- The sampler and verifier calls against a fixed `buffers[:rows]` slice.

Outside the captured region:

- `update(...)` and `step_grammar(...)`, which run host-side Python and issue H2D copies.
- The step 3 read-back, which is a device-to-device clone driven from Python.
- Any RNG draw.

Capture requirements:

- Call `prepopulate(...)` with the full set of capture sizes before the first capture.
- Capture one graph per batch size, and pad the batch up to the nearest captured size. Tuned
  configuration resolution happens at capture time and is baked into the graph.
- Pass a caller-owned `tokens` buffer with `in_place=True`, so no output allocation happens inside
  the graph.

Determinism holds across capture: identical buffers and logits produce identical tokens, with or
without programmatic dependent launch. Under tensor parallelism, seeding each request's generator
identically on every rank is sufficient for agreement without a broadcast.

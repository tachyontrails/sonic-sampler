# Slot-Based Pre-Allocation

Integrating SonicSampler into an engine where each request holds a stable slot for its lifetime.

Prerequisites: [`concepts.md`](concepts.md) and [`quickstart.md`](quickstart.md).

## When this applies

This is the right guide when the engine maintains a fixed-capacity request table, assigns a slot
index on admission, releases it on completion, and identifies rows of a forward pass by their slot
rather than by their position in the batch. Slots are then a natural key into `SamplingBuffers`:
parameters are written once at admission and read for as long as the request lives.

The kernels always receive the full `SamplingBuffers` together with a `slot_mapping` tensor, which
maps each row of the current batch to the slot holding its parameters.

## Lifecycle

```
admission        →  allocate(positions=[...])
                      ↓
forward pass     →  autoregressive(...) or drafting(...) or verifier(...)
                      ↓
post-iteration   →  step_weights(positions=[...], indices=...)
                      ↓
completion       →  release the slot, no buffer work required
```

Nothing is written back on release. A slot is reusable as soon as the next `allocate` overwrites
it, and the parameters left behind are never read because the corresponding row disappears from
`slot_mapping`.

## Admission

`allocate(...)` is the single entry point for admitting one or more requests. It takes parallel
lists of parameters plus the `positions` they belong to, and drives the whole host to pinned to
device path internally.

```python
event = buffers.allocate(
    encodings=[request.prompt_ids for request in admitted],
    multiplicative=[request.repetition_penalty for request in admitted],
    frequency=[request.frequency_penalty for request in admitted],
    presence=[request.presence_penalty for request in admitted],
    biases=[request.logit_bias for request in admitted],
    temperature=[request.temperature for request in admitted],
    top_k=[request.top_k for request in admitted],
    top_p=[request.top_p for request in admitted],
    min_p=[request.min_p for request in admitted],
    top_logprobs=[request.top_logprobs for request in admitted],
    positions=[request.slot for request in admitted],
    greedy_drafts=[request.greedy_draft for request in admitted],
    bitmasks=[request.bitmask for request in admitted],
    generator=[request.generator for request in admitted],
    stream=copy_stream,
)
```

All inputs are host-resident, i.e. CPU tensors or plain Python values. Every list has the same
length as `positions`, and element $j$ describes the request landing at slot `positions[j]`.

| Argument | Notes |
| :--- | :--- |
| `encodings` | int64 CPU tensors of prompt token ids, one per request |
| `biases` | `dict[int, float]` or `None` per request |
| `positions` | The target slots, required in this strategy |
| `indicators` | Optional pre-resolved `ScopedIndicators`, otherwise derived from the parameters |
| `greedy_drafts` | Per-request or a single broadcast bool |
| `bitmasks` | Pinned int32 bitmask tensors or `None` per request, applied at timestep $0$ |
| `generator` | Per-request `torch.Generator`, typically seeded from the request's own seed |
| `stream` | Dedicated copy stream, which switches on event recording |

The call performs, in order:

1. Copies the prefill bitmasks into `grammar[position, 0]` for the requests that have one.
2. Resolves the `ScopedIndicators` from the parameters, unless `indicators` was supplied.
3. Stages the indicators and the non-repetition parameters into pinned memory.
4. Scatters each `encodings` tensor into the pinned context histogram, i.e.
   $\texttt{context}[s, v] = \lvert \{\, t : \text{prompt}_s[t] = v \,\} \rvert$.
5. Refreshes the stochastic weights with `prefill=True`, but only when at least one admitted
   request has a stochastic target.
6. Dispatches the H2D transfers and records the event.

The return value is a `torch.Event` when `stream` was given, and `None` otherwise. Have the compute
stream wait on it before the first forward pass that reads these slots:

```python
if event is not None:

    torch.cuda.current_stream().wait_event(event)
```

Running admission on a dedicated copy stream is what lets the transfers overlap with the previous
iteration's compute, which is the intended shape for an engine with overlap scheduling:

```
                ── time ──────────────────────────────────────────────────────────────────────▶

                ┌───┬───┬───┐
host            │ B │ D │ F │
                └───┴───┴───┘

                ┌───┬───┬───┬───┬───┬───┐
copy stream     │ A │ C │ E │ G │ H │ I │ ◆
                └───┴───┴───┴───┴───┴───┘

                ┌────────────────────────────────────┐   ┌────────────────────────────────────┐
compute stream  │                 N                  │ ◇ │               N + 1                │
                └───────────[ iteration ]────────────┘   └───────────[ iteration ]────────────┘

                ◆  event recorded         ◇  compute stream waits on the event
```

| Box | Lane | Work |
| :--- | :--- | :--- |
| `A` | copy | Prefill grammar bitmasks into `grammar[s, 0]`, already pinned by the caller |
| `B` | host | Resolve the `ScopedIndicators` and pack them into pinned staging |
| `C` | copy | Indicators into `flags.packed` |
| `D` | host | Pack temperature, top-$k$, top-$p$, min-$p$, top logprobs, biases, and repetition penalties into pinned staging |
| `E` | copy | Repetition penalties into `repetition.{multiplicative, frequency, presence}` |
| `F` | host | Histogram each prompt into the pinned context counts |
| `G` | copy | Context histogram into `counts.context`, with `counts.decode` zeroed |
| `H` | copy | Prefill Gumbel-Max refresh, a device-side RNG kernel |
| `I` | copy | Sampling parameters and biases into their device buffers |
| `N` | compute | `target forward → verify → γ × draft` |

Letters run in issue order and alternate between the lanes, so each copy box sits to the right of
the host box that staged its source. `A` needs no host box because the caller supplies pinned
bitmasks, and `I` trails the sequence because the parameters packed at `D` have their transfer
deferred to the closing `transfer(...)` that also records the event.

Widths are indicative rather than proportional. Every host and copy box is microseconds against a
millisecond-scale `N`, so admission disappears behind the in-flight iteration. Two further notes:

- `H` is conditional. If no admitted request has a stochastic target, the refresh is skipped and
  the copy lane is one box shorter.
- Without a `stream`, the copy lane serializes onto the compute stream, `◆` and `◇` vanish, and
  admission becomes a stall rather than an overlap.

### Weight initialization at admission

With `prefill=True`, the refresh is deliberately narrow:

| Scope | What is refreshed |
| :--- | :--- |
| `target` | Timestep $0$ only, since prefill produces a single logit row per request |
| `draft` | All $\gamma + 1$ timesteps, but only for requests whose draft is stochastic |
| `uniform` | Untouched, as there is nothing to verify yet |

Requests with a greedy target are skipped entirely, and if the whole admission set is greedy no
weight work happens at all.

## The decode iteration

Build a `slot_mapping` for the rows in the batch, in the order the logits are laid out:

```python
slot_mapping = torch.tensor(
    [request.slot for request in batch],
    dtype=torch.int32,
    device=device,
)
```

Row $i$ of `logits` then reads its parameters from slot `slot_mapping[i]`. A non-speculative step
is a single call:

```python
selection = sampler.autoregressive(
    logits,                         # [ R, V ], bfloat16
    buffers,
    prefill=False,
    update_counts=True,
    logprobs=True,
    slot_mapping=slot_mapping,
)
```

Because `update_counts=True` accumulates into `repetition.counts.decode` at the mapped slot, the
histogram stays correct across iterations with no engine-side bookkeeping.

Follow the pass with the weight step:

```python
positions = [request.slot for request in batch]

buffers.step_weights(
    positions=positions,
    indices=slot_mapping,
    generator=[request.generator for request in batch],
)
```

`step_weights` is the per-iteration counterpart to `allocate`. It is a no-op when every position in
the batch has a greedy target, so calling it unconditionally is safe. When `indices` is provided
and at least one position drafts stochastically, the draft weights at those slots are staged into
the target weights before both scopes are refreshed, which is what keeps verification on-policy.

`indices` is the device-resident tensor of the same `positions`, so `slot_mapping` serves directly.
It can be omitted whenever no staging is possible, i.e. for a purely autoregressive engine with no
drafter, or when the engine fixes greedy drafting for every request. The latter is the common
stochastic-verify-greedy-draft configuration, where only the target weights need refreshing.

Both `stream` and `event` are accepted here as well, for engines that run weight upkeep off the
compute stream. Unlike admission, upkeep consumes the iteration it follows and produces for the
next, so it is bracketed by two events rather than one:

```
                ── time ────────────────────────────────────────────────────────────────────────────────▶

                                                       ┌───┐
host                                                   │ B │
                                                       └───┘

                                                             ┌───┬───┬───┐
copy stream                                                ◇ │ C │ D │ E │ ◆
                                                             └───┴───┴───┘
                                                       └─ step_weights ──┘

                ┌────────────────┬────────┬───────────┐   ┌────────────────┬───┬────────┬───────────┐
compute stream  │ target forward │ verify │ γ × draft │ ◆ │ target forward │ ◇ │ verify │ γ × draft │
                └───────────[ iteration N ]───────────┘   └───────────[ iteration N + 1 ]───────────┘

                ◆  recorded        ◇  waits on the preceding record in the other lane
```

| Box | Lane | Work |
| :--- | :--- | :--- |
| `B` | host | Subset the scoped indicators for `positions` and resolve which rows need work |
| `C` | copy | `indexed_copy` of `gumbel.draft` into `gumbel.target` at `indices`, gated by the draft indicator |
| `D` | copy | Draw fresh $\log \varepsilon$ into `gumbel.draft`, or into `gumbel.target` for greedy-draft rows |
| `E` | copy | Draw fresh $u \sim \mathcal{U}(\texttt{TINY}, 1]$ into `weights.uniform` |

`B`, `C`, `D`, `E` are the internals of a single call, which is what the brace marks.
`step_weights` resolves the scoped subset on the host, then issues the three dispatches onto
`stream`, so the host box and the copy boxes cannot be scheduled apart. The routine overlaps
iteration $N$ because the launches for $N$ have already returned to the host, not because `B` is
separable.

The `◇` on the copy lane is the engine's own `copy_stream.wait_event(...)`, issued before the call,
since `step_weights` accepts an event to record but none to wait on. The matching `◆` is recorded by
the engine on the compute stream at the close of `γ × draft`, which is the earliest point at which
`C` is safe: `verify` has finished reading `gumbel.target`, and the drafter has finished writing
`gumbel.draft`.

The second `◆` is the `event` argument, recorded on the copy stream once `E` completes. Its `◇` sits
between `target forward` and `verify` in $N + 1$, because the forward pass reads no weights. An
engine that would rather not synchronize mid-iteration can hoist that wait to the head of
$N + 1$ at the cost of a little overlap.

The work fits comfortably in the bracket, since `C`, `D`, and `E` are microseconds against a
millisecond-scale `target forward`. Running `step_weights` on the compute stream makes both events
implicit and is the simpler default.

## The speculative loop

One iteration of $\gamma$-lookahead speculative decoding: the drafter proposes $\gamma$ tokens, the
verifier accepts a prefix of them and appends one more, so a row emits between $1$ and
$\gamma + 1$ tokens.

Two caller-owned buffers span the chain. The token buffer covers $\gamma + 1$ timesteps, i.e. the
$\gamma$ drafted columns plus the column the verifier writes the bonus or corrected token into. The
probability buffer only needs the $\gamma$ drafted timesteps, since the bonus position has no draft
distribution to verify against.

```python
tokens = torch.empty(max_batch_size, lookahead + 1, dtype=torch.int32, device=device)

draft_probs = torch.empty(
    (max_batch_size, lookahead, vocab_size), dtype=torch.bfloat16, device=device
)

# 1. Draft γ candidate tokens, one timestep at a time.

for timestep in range(lookahead):

    draft_logits = drafter.forward(...)                 # [ R, V ], bfloat16

    selection = sampler.drafting(
        draft_logits,
        buffers,
        timestep=timestep,
        tokens=tokens[:rows],
        probabilities=draft_probs[:rows, timestep],
        slot_mapping=slot_mapping,
        in_place=True,
    )

    buffers.step_grammar(
        bitmasks=[request.bitmask_at(timestep + 1) for request in batch],
        positions=positions,
        timestep=timestep + 1,
    )

# 2. Verify the chain against the target model in one call.

verification = verifier(
    target_logits,                                      # [ R, γ + 1, V ], bfloat16
    tokens[:rows],                                      # [ R, γ + 1 ], int32
    buffers,
    probabilities=draft_probs[:rows],                   # [ R, γ, V ]
    slot_mapping=slot_mapping,
    update_counts=True,
    logprobs=True,
    in_place=True,
)

# 3. Advance the stochastic weights for the next iteration.

buffers.step_weights(
    positions=positions,
    indices=slot_mapping,
    generator=[request.generator for request in batch],
)
```

Points worth noting:

- `drafting` reads `flags.draft`, `grammar[:, timestep]`, and `gumbel.draft[:, timestep]`, so the
  timestep argument selects the slice of every timestep-indexed buffer.
- `in_place=True` writes the selection into `tokens[:, timestep]`, and from timestep $1$ onward the
  same buffer feeds the repetition penalties as the drafted-token history.
- `probabilities` receives the draft distribution $q$ that the verifier consumes, one contiguous
  row of width $V$ per drafted timestep, i.e. `[ B, γ, V ]`. It is only populated for rows whose
  draft is stochastic, and can be omitted entirely when every draft is greedy. Rows that draft
  greedily are verified against the masked residual instead, i.e. the drafted token's mass is
  zeroed out of $p_t$ before resampling.
- `step_grammar` inserts the next timestep's bitmasks as the drafter advances. Constrained decoding
  is per-timestep here because each drafted token narrows the grammar for the one after it.
- The verifier emits $\texttt{offsets}[i] + 1$ tokens for row $i$, as described in
  [`concepts.md`](concepts.md).

### Cross-vocabulary drafting

When the drafter has a smaller vocabulary than the target, e.g. EAGLE-3, pass the draft-to-target
mapping and the target width so the selection is expressed in target token ids:

```python
selection = sampler.drafting(
    draft_logits,                   # [ R, V_d ]
    buffers,
    timestep=timestep,
    d2t_mapping=d2t,                # [ V_d ], int32
    target_vocab=target_vocab_size,
    slot_mapping=slot_mapping,
)
```

The emitted `probabilities` are then widened to `[ R, V_t ]`, which is what the verifier expects.

## CUDA graph capture

The slot-based strategy is well suited to capture, because `slot_mapping` is the only thing that
varies between iterations and it is a device tensor that can be updated in place.

Inside the captured region:

- The sampler and verifier calls, since every buffer they touch is pre-allocated and pointer-stable.

Outside the captured region:

- `allocate(...)`, `step_weights(...)`, and `step_grammar(...)`, all of which run host-side Python
  and issue H2D copies.
- Any RNG draw, which is why weight refresh is a buffer update rather than an in-kernel draw.

Capture requirements:

- Reuse the same `slot_mapping` tensor across iterations and overwrite its contents with
  `copy_`, rather than constructing a new one. Pad it to the captured batch size.
- Pass a caller-owned `tokens` buffer with `in_place=True`, so no output allocation happens inside
  the graph.
- Capture one graph per batch size you intend to run, and pad the batch up to the nearest captured
  size. The `tuning(...)` resolution happens at capture time and is baked into the graph, so a
  captured graph is specialized to the row count it was captured with.

Determinism holds across capture: identical buffers and logits produce identical tokens, with or
without programmatic dependent launch. Under tensor parallelism, seeding each request's generator
identically on every rank is sufficient for agreement without a broadcast.

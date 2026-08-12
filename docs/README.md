# Integration Documentation

Everything needed to wire SonicSampler into an inference engine. For what the kernels do and why,
see the top-level [`README.md`](../README.md) and the paper.

| Document | Purpose |
| :--- | :--- |
| [`concepts.md`](concepts.md) | Notation, the buffer model, indicators, outputs, and the integration contracts |
| [`quickstart.md`](quickstart.md) | The adoption ladder, initialization, and a minimal decode step |
| [`slot-based.md`](slot-based.md) | Full integration for engines with a persistent slot per request |
| [`just-in-time.md`](just-in-time.md) | Full integration for engines that compose the batch per iteration |
| [`reference.md`](reference.md) | Signatures, shapes, dtypes, constants, and indicator bits |

Start with `concepts.md`, then `quickstart.md`. From there the path forks on how your engine holds
per-request sampling state:

- A fixed-capacity request table, with a slot assigned at admission and a `slot_mapping` per
  forward pass, leads to [`slot-based.md`](slot-based.md).
- Per-request metadata held engine-side, with the buffers repopulated in batch order before every
  forward pass, leads to [`just-in-time.md`](just-in-time.md).

Each of those covers admission, per-iteration upkeep, the speculative decoding loop, and CUDA graph
capture for that strategy in full, so only one of them needs reading.

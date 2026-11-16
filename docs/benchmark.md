# Benchmark

Re-tuning the dispatch tables: sweeping the top-$`k`$ kernel configurations over a node's GPUs, then
reducing the timings into `resources/dispatch/sm{cc}_k{k}.toml`.

Prerequisites: [`quickstart.md`](quickstart.md#what-initialize-resolves) for how a table is
resolved at initialization.

## When to re-tune

A table holds, per vocabulary and batch size, the fastest configuration of each top-$`k`$ strategy:
its vocab block size `block_n` and the warp counts of both stages. Sweep again when:

- The device architecture has no shipped table.
- The kernels change, or the top-$`k`$ bound $k$ does.
- The engine runs vocabulary or batch sizes beyond the table's buckets. A size resolves to the
  first bucket at or above it, and above the largest bucket only up to $1.5\times$ its size.
  Beyond that, a vocabulary size is refused, and a batch size falls back to bitonic at 16 / 8 warps
  on the vocab bucket's tile.

## Installation

```
uv pip install "sonic-sampler[bench]"
```

The `bench` extra adds `rich`, `loguru`, and `nvidia-ml-py`, which every command needs. From a
checkout, `python -m sonic_sampler.benchmark` runs the same commands as `sonic-sampler bench`.

## Preparing the node

Lock the SM clock on the GPUs to be swept, and leave them otherwise idle:

```
sudo nvidia-smi -i 0,1 -lgc 1845,1845
```

An unlocked clock boosts and settles with load, which shows as drift. Lock at a clock the power cap
leaves alone: on a B200, a lock at 1965 MHz runs at 1845 MHz under the software power cap. During a
sweep, `nvidia-smi -q -d PERFORMANCE` should report `SW Power Cap: Not Active`, and the run view
reports the time each GPU was held back by power or thermal limits. Reset the clocks afterwards
with `sudo nvidia-smi -i 0,1 -rgc`.

## Running a sweep

```
sonic-sampler bench run runs/b200 -g 0 1
```

A run passes through three phases, each in fresh worker processes, one per GPU slot:

1. **Probe.** Each GPU reports its identity, clocks, and power limits through NVML. The GPUs must
   share an architecture and their shared memory.
2. **Compile and calibrate.** The timer is calibrated on the first GPU while the others compile
   every kernel, `-j` processes per GPU. On a single GPU, calibration runs first.
3. **Time.** One process per GPU takes buckets heaviest first. A bucket is one vocabulary and batch
   size, and every case in it is timed by one process on one logits tensor.

The default sweep covers 6 vocabulary sizes, 6 vocab block sizes, 14 batch sizes, 4 warp counts per
stage, and 3 strategies. On a B200 that is 30 vocabulary tilings, 90 compile units of 685 kernels,
and 84 buckets of 18,200 cases.

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `-k`, `--top-k` | $128$ | The top-$`k`$ bound, a power of two up to `MAX_K` |
| `-v`, `--vocab-sizes` | $2^{13}, \dots, 2^{18}$ | The vocabulary sizes |
| `-n`, `--block-ns` | $2^{10}, \dots, 2^{15}$ | The vocab block sizes |
| `-b`, `--batch-sizes` | $1, 2, \dots, 32, 48, 64, 96, \dots, 512$ | The batch sizes: powers of two, with midpoints above $32$ |
| `-m`, `--max-blocks` | the register bound | The cap on padded vocab blocks |
| `--warp-floor`, `--no-warp-floor` | on | Skip warp counts whose tile overflows the register budget |
| `--pdl`, `--no-pdl` | on | Launch with programmatic dependent launch |
| `--seed` | $0$ | The base seed of the per-bucket logits |
| `-g`, `--gpus` | every GPU | The NVML indices of the GPUs to run on |
| `-j`, `--jobs` | $8$ | The compile processes per GPU |
| `--warmup` | $3$ | The eager calls before each capture |
| `--unroll`, `--replays` | calibrated | Fix the timer, which skips calibration |
| `--tolerance` | $0.005$ | The tolerance on calibrated dispersion and on drift |
| `--heartbeat` | $5.0$ | The seconds between heartbeats while timing |

Vocab blocks are bounded before anything compiles. Two limits set an exact ceiling: the 16-bit
index of a packed key, and the second stage's staging buffer of $4 \cdot \hat{Z}_v \cdot k$ bytes
in opt-in shared memory, where $\hat{Z}_v$ is the vocab block count rounded up to a power of two.
The register file sets a softer default cap, which `-m` overrides.

The logits are synthetic bf16 rows: Student-$`t`$ draws reshaped to a Zipfian rank profile, plus
Gaussian noise, with the fitted parameters in `resources/benchmark/logits.toml`. Each bucket's seed
derives from `--seed`, its vocabulary size, and its batch size, so every run times the same
logits.

## Measurement

### The timer

Each case is timed by `GraphTimer`:

1. `warmup` eager calls load the kernels.
2. `unroll` back-to-back calls are captured into one CUDA graph. With PDL, each call's first stage
   launches under the tail of the previous call, as in a decode loop.
3. One untimed replay settles the L2 cache.
4. `replays` timed replays follow, sharing boundary events. Each gives a latency per call, the
   replay's time over `unroll`.

A record keeps the median, 10th, and 90th percentile of those latencies.

### Calibration

Most of the noise lies between captures rather than between replays of one capture: each capture
empties PyTorch's allocator cache, so a fresh capture reads memory laid out anew. Calibration
therefore chooses `unroll` and `replays` by how far the median moves from capture to capture.

- **Probes.** The median case of each strategy, in the lightest and the heaviest bucket.
- **Grid.** `unroll` of 25, 50, or 100, and `replays` of 5 or 10. Replay counts are prefixes of
  one set of samples.
- **Repeats.** 10 captures of every probe at every `unroll`, after one discarded round, since the
  first captures after idle run fast.
- **Dispersion.** The relative range of the medians across repeats, without the fastest and the
  slowest, taken at the worst probe. The full range is kept as `spread`.
- **Choice.** The cheapest pair, by its `unroll · (replays + 1)` replayed calls, whose dispersion
  is within the tolerance. When none is, the least dispersed pair is chosen, and the run records
  that calibration did not converge.

### Heartbeats and drift

Each timing process holds one reference: the heaviest bucket's bitonic probe, captured once and
replayed for every heartbeat, so its timings differ only by when they were taken. The process
replays it for 5 s to load the GPU, takes a start beat, then a beat between cases every heartbeat
interval, and an end beat after its last bucket. A beat times at least 20 replays and reads the
GPU's telemetry: SM clock, temperatures, power, energy, and the time held back by power and thermal
limits.

- **Drift** runs from the median of the first three beats between cases to the median of the
  latest three, or of fewer when a process has fewer than six. Drift beyond the tolerance is logged
  as a warning and shown in red.
- **Step** runs from the start beat to the first beats between cases. With no bucket open, the
  reference runs faster than it does between cases: on a B200, by about 0.1% for the default
  sweep's reference and by up to 0.7% for a lighter one. Every case is timed with a bucket open,
  so the step is reported apart and not held to the tolerance.

### Placement and devices

Timings depend on where logits sit in memory, deterministically, by around 1.5% on a B200. Within
a bucket, every case reads the same logits tensor in the same process, so cases are compared like
for like.

A run times each bucket on one GPU, but merged runs can rank records from different GPUs within
one bucket.

## The run directory

| File | Contents |
| :--- | :--- |
| `manifest.toml` | Format version, start, architecture, package versions, sweep, limits, timer, tolerance, compile jobs, heartbeat interval, devices, calibration, and once finished the case count per status and each GPU's drift |
| `records.csv` | One row per timed case: the case, status, GPU, time, median and percentile latencies, and both stages' registers and spills, or the error |
| `heartbeats.csv` | One row per beat: GPU, phase, time, reference latencies, and telemetry |
| `calibration.csv` | Every timed replay of the calibration, labelled by probe, `unroll`, repeat, and replay |
| `sm{cc}_k{k}.toml` | The dispatch table, once reduced |

A record is `ok`, `invalid` when its kernels cannot launch for lack of resources, or `failed` on
any other error, including a crash of its process. Later rows of a case supersede earlier ones.

## Resuming

```
sonic-sampler bench resume runs/b200 [-g 0 1] [-j 8] [--heartbeat 5.0]
```

A resume keeps the run's sweep, limits, timer, tolerance, and start, and its compile jobs and
heartbeat unless given. It times again, in full, every bucket with a case missing or failed,
compiling only their units. A run with every bucket settled starts no process at all: a finished
run is left as it is, and a run stopped after its last record is completed from its heartbeats.

Within a run, a crashed timing process is replaced. Its bucket is timed again in full, except for
the case it crashed on, which is recorded as `failed`. A process that keeps exiting without taking
up a task fails the run after three restarts.

## Reducing

```
sonic-sampler bench reduce runs/b200 [runs/b200-more ...] [-o PATH] [--no-overwrite]
```

The runs merge oldest first, so a case timed again in a later run keeps its later record. The runs
must share an architecture and $k$. In each bucket, the `ok` records rank per strategy by median,
then 90th percentile, then smaller vocab block and warp counts. The table keeps each strategy's
fastest record, the fastest strategy first.

The table is written to `sm{cc}_k{k}.toml` in the latest run by default, over any table already
there unless `--no-overwrite` is given. A bucket missing a strategy is refused. The command also
prints:

| Section | Shows |
| :--- | :--- |
| Standings | Each bucket's winner per strategy, in priority order, with its median |
| Near-ties | Pairs too close to rank with confidence, among adjacent winners and each strategy's two fastest. A pair is close by `margin` when the slower one's margin lies within the larger calibrated dispersion of the runs that timed them (the tolerance, for a fixed timer), or by `overlap` when their p10 and p90 latencies overlap |
| Register findings | Winners that spill in either stage, sit on the smallest warp count the register rule left, or use as many vocab blocks as the cap allows |
| Warnings | Runs still in progress, buckets missing a strategy, and runs that differ in package versions or timer |

## Showing

```
sonic-sampler bench show --runs runs/b200 [...] [--table PATH]
```

`--runs` renders each run's devices, calibration, progress, and drift, then the standings merged
across the runs. A run still in progress shows the standings so far. `--table` renders a dispatch
table's configurations in priority order. Either flag, or both, may be given.

## Shipping a table

Tables are read from `sonic_sampler/resources/dispatch/sm{cc}_k{k}.toml`. From an editable install,
reduce straight into the package:

```
sonic-sampler bench reduce runs/b200 -o src/sonic_sampler/resources/dispatch/sm100_k128.toml
```

Before shipping it, read the run with `show`. Look for:
- A converged calibration.
- No drift warning, and no time held back by power or thermal limits.
- No warnings in the reduction.
- Near-ties in the buckets the engine runs most, which another run may rank the other way.
- Register findings on those buckets' winners.

## From Python

The commands are thin over the runner and the reduction:

```python
from pathlib import Path

from sonic_sampler.benchmark.records import RunDirectory
from sonic_sampler.benchmark.reduction import Reduction
from sonic_sampler.benchmark.runner import Options, Runner
from sonic_sampler.benchmark.sweep import Sweep


if __name__ == "__main__":

    directory = Path("runs/b200")

    Runner.initialize(directory, Sweep(batch_sizes=(1, 8, 64)), Options(gpus=(0, 1))).run()

    run = RunDirectory.load(directory)

    (
        Reduction.initialize([run])
        .summary
        .save(run.table)
    )
```

Workers start with `spawn` and import the main module again, so a script that runs a sweep guards
its entry point with `if __name__ == "__main__"`.

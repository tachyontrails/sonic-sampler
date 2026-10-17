# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026 SonicSampler Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch

from collections.abc import Callable
from itertools import pairwise, product
from operator import attrgetter
from statistics import median, quantiles
from typing import Any, Final

from cytoolz.itertoolz import groupby
from msgspec import Struct

from torch import Event
from torch.cuda import CUDAGraph

from sonic_sampler.benchmark.base import relative_range
from sonic_sampler.benchmark.sweep import Axis


type Probe = Callable[[], Any]


# The candidate unroll and replay counts of a calibration.
UNROLLS: Final[Axis] = (25, 50, 100)
REPLAYS: Final[Axis] = (5, 10)

# The candidate repeat count of a calibration.
REPEATS: Final[int] = 10

# The default tolerance on the relative range of a calibrated median.
TOLERANCE: Final[float] = 0.005

# The microseconds per millisecond, the unit of CUDA event timings.
MICROSECONDS: Final[int] = 1_000


class Timing(Struct, frozen=True):

    """
    The per-call latency statistics over a timer's replays.

    """

    # The median latency per call, in microseconds.
    median_us: float

    # The 10th percentile latency per call, in microseconds.
    p10_us: float

    # The 90th percentile latency per call, in microseconds.
    p90_us: float

    @classmethod
    def initialize(cls, samples: list[float]) -> Timing:

        deciles = quantiles(samples, n=10, method="inclusive")

        return cls(median_us=median(samples), p10_us=deciles[0], p90_us=deciles[-1])


class IndexedRun(Struct):

    # The total call(s) per captured graph.
    unroll: int

    # The corresponding index of the probe.
    index: int

    # The measured latencies of the probe calls.
    latencies: list[float]


class GraphTimer(Struct, frozen=True):

    """
    Times a probe as replays of one CUDA graph of back-to-back calls.

    Note(s):

        • The eager calls load the probe's kernels, and the untimed replay settles the L2 cache.

        • Calls run back to back, so with PDL each call's first stage launches under the tail of
          the previous call, as in a decode loop.

    """

    # The eager call(s) before capture.
    warmup: int = 3

    # The call(s) unrolled into the captured graph.
    unroll: int = 100

    # The timed replay(s) of the graph.
    replays: int = 10

    def __post_init__(self) -> None:

        if self.warmup < 0:

            raise ValueError(f"warm-up calls must be non-negative, got {self.warmup} instead")

        elif min(self.unroll, self.replays) < 1:

            raise ValueError(
                f"unroll and replay counts must be positive, got {self.unroll}, {self.replays}"
            )

    def latency(self, start: Event, end: Event) -> float:

        return (
            (start.elapsed_time(end) * MICROSECONDS)
            / self.unroll
        )

    def sample(self, probe: Probe) -> list[float]:
        """
        The latency per call of each timed replay, in microseconds.

        """
        for _ in range(self.warmup):

            probe()

        with torch.cuda.graph(graph := CUDAGraph()):

            for _ in range(self.unroll):

                probe()

        graph.replay()

        # Consecutive replays share a boundary event, so the samples tile the timeline.

        events = [Event(enable_timing=True) for _ in range(self.replays + 1)]
        events[0].record()

        for event in events[1:]:

            graph.replay()
            event.record()

        events[-1].synchronize()

        return [self.latency(start, end) for start, end in pairwise(events)]

    def measure(self, probe: Probe, index: int) -> IndexedRun:

        return IndexedRun(unroll=self.unroll, index=index, latencies=self.sample(probe))

    def __call__(self, probe: Probe) -> Timing:

        return Timing.initialize(self.sample(probe))


class Series(Struct, frozen=True):

    """
    The replay samples of one probe at one unroll count, per repeat.

    """

    # The calls per captured graph.
    unroll: int

    # The index of the probe.
    probe: int

    # The per-call latency samples of each repeat.
    repeats: list[list[float]]

    @classmethod
    def from_runs(cls, runs: list[IndexedRun]) -> Series:

        unroll = (first := runs[0]).unroll
        index = first.index

        return cls(unroll=unroll, probe=index, repeats=[r.latencies for r in runs])

    def dispersion(self, replays: int) -> float:
        """
        The relative range of the median across repeats, over each repeat's first replays.

        """
        return relative_range([median(samples[:replays]) for samples in self.repeats])


class Trial(Struct, frozen=True):

    """
    The repeatability of one unroll and replay count pair over every probe.

    """

    # The calls per captured graph.
    unroll: int

    # The timed replays per measurement.
    replays: int

    # The worst relative range of the median across repeats, over every probe.
    dispersion: float

    @property
    def cost(self) -> int:
        """
        The replayed calls per case, i.e. the timed replays and the untimed one.

        """
        return self.unroll * (self.replays + 1)

    def timer(self, warmup: int) -> GraphTimer:

        return GraphTimer(warmup=warmup, unroll=self.unroll, replays=self.replays)


class Calibration(Struct):

    """
    The cheapest unroll and replay count pair whose median repeats within tolerance on every probe.

    Note(s):

        • Replay counts are prefixes of one set of samples, so each unroll count is captured once
          per repeat and probe.

        • When no pair is within tolerance, the most repeatable one is chosen instead.

    """

    # The tolerance on a trial's dispersion.
    tolerance: float

    # The trial(s) of every unroll and replay count pair, cheapest first.
    trials: list[Trial]

    @classmethod
    def initialize(cls, series: list[Series], replays: Axis, tolerance: float) -> Calibration:

        bands = groupby(attrgetter("unroll"), series)

        trials = (
            Trial(unroll=unroll, replays=count, dispersion=max(s.dispersion(count) for s in band))
            for (unroll, band), count in product(bands.items(), replays)
        )

        return cls(tolerance=tolerance, trials=sorted(trials, key=attrgetter("cost")))

    @classmethod
    def measure(
        cls,
        probes: list[Probe],
        warmup: int,
        tolerance: float = TOLERANCE,
        unrolls: Axis = UNROLLS,
        replays: Axis = REPLAYS,
        repeats: int = REPEATS,
    ) -> Calibration:

        count = max(replays)
        timers = [GraphTimer(warmup=warmup, unroll=u, replays=count) for u in unrolls]

        # Repeats run outermost, so slow drift spreads across every unroll count and probe.

        schedule = product(range(repeats), timers, enumerate(probes))
        runs = [timer.measure(probe, index) for _, timer, (index, probe) in schedule]

        grouped = groupby(attrgetter("unroll", "index"), runs)
        series = list(map(Series.from_runs, grouped.values()))

        return cls.initialize(series, replays, tolerance)

    @property
    def least_dispersed(self) -> Trial:

        return min(self.trials, key=attrgetter("dispersion"))

    @property
    def choice(self) -> Trial:

        admissible = next(filter(self.admits, self.trials), None)

        return admissible or self.least_dispersed

    @property
    def is_converged(self) -> bool:

        return self.admits(self.choice)

    def admits(self, trial: Trial) -> bool:

        return trial.dispersion <= self.tolerance

    def timer(self, warmup: int) -> GraphTimer:

        return self.choice.timer(warmup)

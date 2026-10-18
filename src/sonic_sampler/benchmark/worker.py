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

from collections.abc import Iterator
from contextlib import suppress
from functools import partial
from multiprocessing.queues import Queue
from typing import Literal

from msgspec import Struct, field

from torch import Tensor

from triton import OutOfResources               # noqa.

from sonic_sampler.benchmark.base import derive_seed
from sonic_sampler.benchmark.environment import Device, pin
from sonic_sampler.benchmark.logits import LogitsFactory
from sonic_sampler.benchmark.records import Record
from sonic_sampler.benchmark.sweep import STRATEGIES, Axis, Bucket, Case, CompileUnit, Group, Sweep
from sonic_sampler.benchmark.timing import Calibration, GraphTimer, Probe, Timing
from sonic_sampler.interface.topk import UnshardedBoundedTopK


type Phase = Literal["start", "end"]


class Message(Struct):

    # The pool slot of the sending worker.
    slot: int


class Probed(Message):

    # The probed GPU.
    device: Device


class Calibrated(Message):

    # The calibration over the plan's probes.
    calibration: Calibration


class Measured(Message):

    # The record of one case.
    record: Record


class Referenced(Message):

    # The NVML index of the worker's GPU.
    device: int

    # Whether the reference was timed before or after the worker's buckets.
    phase: Phase

    # The reference timing.
    timing: Timing


class Started(Message):

    """
    Marks the worker taking up its current task, so a crash before it is not blamed on the task.

    """


class Done(Message):

    """
    Marks the end of the worker's current task, freeing its slot for the next one.

    """


class Workbench(Struct):

    """
    The kernel instances and logits of one GPU-pinned process.

    Note(s):

        • Instances are sized for the sweep's largest batch and shared by every batch size, since
          the batch size only sets the launch grid.

    """

    # The sweep the instances are built for.
    sweep: Sweep

    # The process's GPU.
    device: torch.device

    # The logits factory on the process's GPU.
    factory: LogitsFactory

    # The rows every instance is sized for.
    rows: int

    # The kernel instance per vocabulary tiling.
    instances: dict[Group, UnshardedBoundedTopK] = field(default_factory=dict)

    @classmethod
    def initialize(cls, sweep: Sweep) -> Workbench:

        torch.cuda.set_device(device := torch.device("cuda", 0))
        factory = LogitsFactory.default(device)

        return cls(sweep=sweep, device=device, factory=factory, rows=max(sweep.batch_sizes))

    def instance(self, group: Group) -> UnshardedBoundedTopK:

        if (instance := self.instances.get(group)) is None:

            with self.device:

                instance = self.instances[group] = UnshardedBoundedTopK.initialize(
                    vocab_size=group.vocab_size,
                    block_size=group.block_n,
                    batch_size=self.rows,
                    k=self.sweep.k,
                    pdl=self.sweep.pdl,
                )

        return instance

    def logits(self, vocab_size: int, batch_size: int) -> Tensor:

        seed = derive_seed(self.sweep.seed, vocab_size, batch_size)

        return self.factory.sample(batch_size, vocab_size, seed=seed)

    def probe(self, case: Case, logits: Tensor) -> Probe:

        return partial(self.instance(case.group), logits, strategy=case.top_k, tuning=case.tuning)

    def batched(self, cases: list[Case]) -> list[Probe]:

        return [self.probe(c, self.logits(c.vocab_size, c.batch_size)) for c in cases]

    def compile(self, unit: CompileUnit) -> None:
        """
        Compiles and loads every admitted kernel of the unit, at one row.

        Note(s):

            • One row suffices: the batch size is not part of any kernel's cache key.

            • Failures are left to the timing phase, which records them per case.

        """
        instance = self.instance(unit.group)
        logits = self.logits(unit.group.vocab_size, 1)

        for tuning in unit.tunings:

            with suppress(Exception):

                instance(logits, strategy=STRATEGIES[unit.strategy], tuning=tuning)

        torch.cuda.synchronize()

    def measure(self, case: Case, logits: Tensor, timer: GraphTimer, device: int) -> Record:
        """
        Times one case, then reads both stages' register use from one more launch, whose kernels
        the timer's warm-up has already loaded.

        """
        instance = self.instance(case.group)
        launchers = instance.launchers(logits, case.batch_size, case.top_k, case.tuning)

        try:

            timing = timer(self.probe(case, logits))
            first, second = (launcher() for launcher in launchers)

        except OutOfResources as error:

            return Record.invalid(case, device, error)

        except Exception as error:

            return Record.failed(case, device, error)

        else:

            return Record.measured(case, device, timing, first, second)

    def reference(self, case: Case, timer: GraphTimer) -> Timing:

        logits = self.logits(case.vocab_size, case.batch_size)

        return timer(self.probe(case, logits))


class Worker(Struct):

    """
    One GPU-pinned process of a pool: it takes tasks from its own queue and reports to the shared
    result queue.

    """

    # The pool slot of the worker.
    slot: int

    # The NVML index of the worker's GPU.
    device: int

    # The shared result queue.
    results: Queue[Message]

    # The kernel workbench on the worker's GPU.
    bench: Workbench

    @classmethod
    def initialize(cls, slot: int, device: int, results: Queue[Message], sweep: Sweep) -> Worker:

        pin(device)

        return cls(slot=slot, device=device, results=results, bench=Workbench.initialize(sweep))

    def tasks[T](self, queue: Queue[T | None]) -> Iterator[T]:
        """
        The worker's tasks until the stop sentinel, reporting each as done once it is processed.

        """
        while (task := queue.get()) is not None:

            self.results.put(Started(slot=self.slot))

            yield task

            self.results.put(Done(slot=self.slot))

    def compile(self, queue: Queue[CompileUnit | None]) -> None:

        for unit in self.tasks(queue):

            self.bench.compile(unit)

    def calibrate(
        self, cases: list[Case], warmup: int, tolerance: float, unrolls: Axis, replays: Axis
    ) -> None:

        calibration = Calibration.measure(
            probes=self.bench.batched(cases),
            warmup=warmup,
            tolerance=tolerance,
            unrolls=unrolls,
            replays=replays,
        )

        self.results.put(Calibrated(slot=self.slot, calibration=calibration))

    def time(self, bucket: Bucket, timer: GraphTimer) -> None:

        logits = self.bench.logits(bucket.vocab_size, bucket.batch_size)

        for case in bucket.cases:

            record = self.bench.measure(case, logits, timer, self.device)
            self.results.put(Measured(slot=self.slot, record=record))

            # A failed case can leave the CUDA context unusable. Synchronizing raises in that
            # case, ending the process so the runner restarts it with a fresh context.

            torch.cuda.synchronize()

    def reference(self, phase: Phase, case: Case, timer: GraphTimer) -> None:

        timing = self.bench.reference(case, timer)
        message = Referenced(slot=self.slot, device=self.device, phase=phase, timing=timing)

        self.results.put(message)

    def measure(self, queue: Queue[Bucket | None], timer: GraphTimer, reference: Case) -> None:

        self.reference("start", reference, timer)

        for bucket in self.tasks(queue):

            self.time(bucket, timer)

        self.reference("end", reference, timer)


def probe_device(slot: int, device: int, results: Queue[Message]) -> None:

    pin(device)

    results.put(Probed(slot=slot, device=Device.probe(device)))


def compile_units(
    slot: int,
    device: int,
    queue: Queue[CompileUnit | None],
    results: Queue[Message],
    sweep: Sweep,
) -> None:

    (
        Worker.initialize(slot, device, results, sweep)
        .compile(queue)
    )


def calibrate_timer(
    slot: int,
    device: int,
    results: Queue[Message],
    sweep: Sweep,
    cases: list[Case],
    warmup: int,
    tolerance: float,
    unrolls: Axis,
    replays: Axis,
) -> None:

    (
        Worker.initialize(slot, device, results, sweep)
        .calibrate(cases, warmup, tolerance, unrolls, replays)
    )


def measure_buckets(
    slot: int,
    device: int,
    queue: Queue[Bucket | None],
    results: Queue[Message],
    sweep: Sweep,
    timer: GraphTimer,
    reference: Case,
) -> None:

    (
        Worker.initialize(slot, device, results, sweep)
        .measure(queue, timer, reference)
    )

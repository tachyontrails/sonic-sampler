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

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from csv import DictReader, DictWriter
from datetime import datetime
from functools import partial
from io import TextIOWrapper
from operator import attrgetter
from pathlib import Path
from typing import Any, Final, Literal

from cytoolz.dicttoolz import valmap
from msgspec import Struct, convert, structs, toml

from triton.compiler import CompiledKernel      # noqa.

from sonic_sampler.benchmark.base import describe, nullify
from sonic_sampler.benchmark.environment import Device
from sonic_sampler.benchmark.sweep import Case, Limits, Sweep
from sonic_sampler.benchmark.timing import Calibration, GraphTimer, Timing


type Status = Literal["ok", "invalid", "failed"]


# The file names within a run directory.
MANIFEST: Final[str] = "manifest.toml"
RECORDS: Final[str] = "records.csv"

# The manifest format version.
MANIFEST_VERSION: Final[int] = 1

# Measurement field suffixes.
MEASURES: Final[tuple[str, ...]] = ("_us", "_regs", "_spills")


def has_null_measures(record: Record) -> bool:

    fields = record.__struct_fields__
    predicates = (getattr(record, f) is None for f in fields if f.endswith(MEASURES))

    return any(predicates)


class Record(Case, frozen=True):

    """
    A measured case, i.e. one row of the run's records.

    """

    # The measurement status.
    status: Status

    # The NVML index of the GPU the case ran on.
    device: int

    # The median latency per call, in microseconds.
    median_us: float | None = None

    # The 10th percentile latency per call, in microseconds.
    p10_us: float | None = None

    # The 90th percentile latency per call, in microseconds.
    p90_us: float | None = None

    # The registers per thread of the first-stage kernel.
    first_regs: int | None = None

    # The spilled bytes per thread of the first-stage kernel.
    first_spills: int | None = None

    # The registers per thread of the second-stage kernel.
    second_regs: int | None = None

    # The spilled bytes per thread of the second-stage kernel.
    second_spills: int | None = None

    # The failure message, if any.
    error: str | None = None

    def __post_init__(self) -> None:

        if self.is_ok and has_null_measures(self):

            raise ValueError(f"{self.case} is marked ok but lacks a measurement")

    @classmethod
    def initialize(cls, case: Case, status: Status, device: int, **measures: Any) -> Record:

        return cls(**structs.asdict(case), status=status, device=device, **measures)

    @classmethod
    def measured(
        cls, case: Case, device: int, timing: Timing, first: CompiledKernel, second: CompiledKernel
    ) -> Record:

        registers = dict(
            first_regs=first.n_regs,
            first_spills=first.n_spills,
            second_regs=second.n_regs,
            second_spills=second.n_spills,
        )

        return cls.initialize(case, "ok", device, **structs.asdict(timing), **registers)

    @classmethod
    def invalid(cls, case: Case, device: int, error: BaseException) -> Record:

        return cls.initialize(case, "invalid", device, error=describe(error))

    @classmethod
    def failed(cls, case: Case, device: int, error: BaseException) -> Record:

        return cls.initialize(case, "failed", device, error=describe(error))

    @property
    def case(self) -> Case:

        return convert(self, type=Case, from_attributes=True)

    @property
    def ordering(self) -> tuple[float | None, float | None, int, int, int]:
        """
        The ranking key: the median, then the 90th percentile, then the smaller tile and warp
        counts, so that equal latencies resolve the same way on every reduction.

        """
        return self.median_us, self.p90_us, self.block_n, self.first_warps, self.second_warps

    @property
    def is_viable(self) -> bool:

        return self.status != "failed"

    @property
    def is_ok(self) -> bool:

        return self.status == "ok"

    def tail_overlap(self, other: Record) -> bool:
        """
        Whether this record's p90 tail overlaps with the other / slower record's p10 head.

        """
        return self.p90_us >= other.p10_us


class RecordLog(list[Record]):

    @classmethod
    def load(cls, path: Path) -> RecordLog:

        with path.open(newline="") as handle:

            rows = map(partial(valmap, nullify), DictReader(handle))

            return cls(convert(row, type=Record, strict=False) for row in rows)

    @property
    def latest(self) -> dict[Case, Record]:

        # Later rows supersede earlier rows of the same case, e.g. a retried failure.

        return {record.case: record for record in self}

    @property
    def settled(self) -> set[Case]:

        return {case for case, record in self.latest.items() if record.is_viable}


class RecordSink(Struct):

    """
    Appends records to a run's records as they arrive, flushing every row.

    """

    # The open file handle.
    handle: TextIOWrapper

    # The row writer over the handle.
    writer: DictWriter

    @classmethod
    @contextmanager
    def open(cls, path: Path) -> Iterator[RecordSink]:

        fresh = not (path.exists() and path.stat().st_size)

        with path.open("a", newline="") as handle:

            writer = DictWriter(handle, fieldnames=Record.__struct_fields__)

            if fresh:

                writer.writeheader()

            yield cls(handle=handle, writer=writer)

    def __call__(self, record: Record) -> None:

        self.writer.writerow(structs.asdict(record))
        self.handle.flush()


class Drift(Struct):

    """
    The drift reference of one GPU: one fixed case, timed as the GPU started and finished timing.

    """

    # The NVML index of the GPU.
    device: int

    # The reference timing as the GPU started timing.
    start: Timing

    # The reference timing as the GPU finished timing.
    end: Timing

    @property
    def change(self) -> float:

        return (self.end.median_us / self.start.median_us) - 1


class Completion(Struct):

    # The run end, in UTC.
    end: datetime

    # The case count per status, over every case's latest record.
    statuses: dict[Status, int]

    # The drift reference of each GPU that timed cases.
    drifts: list[Drift]

    @classmethod
    def initialize(cls, end: datetime, records: RecordLog, drifts: list[Drift]) -> Completion:

        statuses = Counter(map(attrgetter("status"), records.latest.values()))

        return cls(end=end, statuses=dict(sorted(statuses.items())), drifts=drifts)


class Manifest(Struct, omit_defaults=True):

    # The manifest format version.
    version: int

    # The run start, in UTC.
    start: datetime

    # The compute capability of the benchmarked device(s), e.g. 100 for SM100.
    arch: int

    # The package version(s) the run executed against.
    versions: dict[str, str]

    # The sweep specification, with the vocab block cap resolved.
    sweep: Sweep

    # The limits applied to the sweep.
    limits: Limits

    # The timer settings every case was measured with.
    timer: GraphTimer

    # The tolerance on calibrated dispersion and on drift.
    tolerance: float

    # The compile processes per GPU.
    compile_jobs: int

    # The GPU(s) the run measured on.
    devices: list[Device]

    # The calibration that chose the timer's unroll and replay counts, unless given.
    calibration: Calibration | None = None

    # The completion summary, once the run has finished.
    completion: Completion | None = None

    @classmethod
    def initialize(
        cls,
        start: datetime,
        versions: dict[str, str],
        sweep: Sweep,
        limits: Limits,
        timer: GraphTimer,
        tolerance: float,
        jobs: int,
        devices: list[Device],
        calibration: Calibration | None = None,
    ) -> Manifest:

        resolved = structs.replace(sweep, max_blocks=limits.requested)

        return cls(
            version=MANIFEST_VERSION,
            start=start,
            arch=devices[0].arch,
            versions=versions,
            sweep=resolved,
            limits=limits,
            timer=timer,
            tolerance=tolerance,
            compile_jobs=jobs,
            devices=devices,
            calibration=calibration,
        )

    @classmethod
    def load(cls, path: Path) -> Manifest:

        return toml.decode(path.read_bytes(), type=cls)

    @property
    def tag(self) -> str:

        return f"sm{self.arch}_k{self.sweep.k}"

    def save(self, path: Path) -> None:

        path.write_bytes(toml.encode(self))


class RunDirectory(Struct):

    # The run directory path.
    path: Path

    # The run manifest.
    manifest: Manifest

    # The run's record(s), in write order.
    records: RecordLog

    @classmethod
    def load(cls, path: Path) -> RunDirectory:

        manifest = Manifest.load(path / MANIFEST)
        records = RecordLog.load(path / RECORDS)

        return cls(path=path, manifest=manifest, records=records)

    @property
    def table(self) -> Path:

        return self.path / f"{self.manifest.tag}.toml"

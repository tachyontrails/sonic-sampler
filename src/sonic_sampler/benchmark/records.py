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
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from csv import DictReader, DictWriter
from datetime import UTC, datetime
from functools import partial
from io import TextIOWrapper
from operator import attrgetter
from pathlib import Path
from statistics import median
from typing import Any, Final, Literal

from cytoolz.dicttoolz import valmap
from cytoolz.itertoolz import groupby
from msgspec import Struct, convert, structs, to_builtins, toml

from triton.compiler import CompiledKernel      # noqa.

from sonic_sampler.benchmark.base import describe, nullify, relative_range
from sonic_sampler.benchmark.environment import Device, Telemetry
from sonic_sampler.benchmark.sweep import Case, Limits, Sweep
from sonic_sampler.benchmark.timing import Calibration, GraphTimer, Series, Timing


type Status = Literal["ok", "invalid", "failed"]

type Phase = Literal["start", "beat", "end"]


# The file names within a run directory.
MANIFEST: Final[str] = "manifest.toml"
RECORDS: Final[str] = "records.csv"
HEARTBEATS: Final[str] = "heartbeats.csv"
CALIBRATION: Final[str] = "calibration.csv"

# The manifest format version.
MANIFEST_VERSION: Final[int] = 3

# The beats between cases, at each end of a timing process, whose median bounds its drift.
WINDOW: Final[int] = 3

# Measurement field suffixes.
MEASURES: Final[tuple[str, ...]] = ("_us", "_regs", "_spills")


def has_null_measures(record: Record) -> bool:

    fields = record.__struct_fields__
    predicates = (getattr(record, f) is None for f in fields if f.endswith(MEASURES))

    return any(predicates)


def load_rows[T: Struct](path: Path, schema: type[T]) -> list[T]:
    """
    The rows of a CSV file written by a `RowSink`, with empty cells read as missing values.

    """
    with path.open(newline="") as handle:

        rows = map(partial(valmap, nullify), DictReader(handle))

        return [convert(row, type=schema, strict=False) for row in rows]


def since_start(heartbeats: list[Heartbeat]) -> list[Heartbeat]:
    """
    The heartbeats of the latest timing process: from its start beat on.

    """
    starts = [index for index, hb in enumerate(heartbeats) if hb.is_starting]

    return heartbeats[starts[-1]:] if starts else []


def between_cases(heartbeats: list[Heartbeat]) -> list[float]:
    """
    The reference medians of the beats taken between cases, i.e. while a bucket was open.

    """
    return [hb.median_us for hb in heartbeats if hb.is_beating]


def save_rows[T: Struct](path: Path, schema: type[T], rows: Iterable[T]) -> None:

    with path.open("w", newline="") as handle:

        writer = DictWriter(handle, fieldnames=schema.__struct_fields__)

        writer.writeheader()
        writer.writerows(map(to_builtins, rows))


class Record(Case, frozen=True):

    """
    A measured case, i.e. one row of the run's records.

    """

    # The measurement status.
    status: Status

    # The NVML index of the GPU the case ran on.
    device: int

    # The time the record was made, in UTC.
    time: datetime

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

        time = datetime.now(UTC)

        return cls(**structs.asdict(case), status=status, device=device, time=time, **measures)

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

        return cls(load_rows(path, Record))

    @property
    def latest(self) -> dict[Case, Record]:

        # Later rows supersede earlier rows of the same case, e.g. a retried failure.

        return {record.case: record for record in self}

    @property
    def settled(self) -> set[Case]:

        return {case for case, record in self.latest.items() if record.is_viable}


class RowSink[T: Struct](Struct):

    """
    Appends rows of one struct type to a CSV file as they arrive, flushing every row.

    Note(s):

        • Rows are written as builtins, so timestamps are ISO 8601 and missing values are empty.

    """

    # The open file handle.
    handle: TextIOWrapper

    # The row writer over the handle.
    writer: DictWriter

    @classmethod
    @contextmanager
    def open(cls, path: Path, schema: type[T]) -> Iterator[RowSink[T]]:

        fresh = not (path.exists() and path.stat().st_size)

        with path.open("a", newline="") as handle:

            writer = DictWriter(handle, fieldnames=schema.__struct_fields__)

            if fresh:

                writer.writeheader()

            yield cls(handle=handle, writer=writer)

    def __call__(self, row: T) -> None:

        self.writer.writerow(to_builtins(row))
        self.handle.flush()


class Sample(Case, frozen=True):

    """
    One timed replay of a calibration probe, i.e. one row of the run's calibration samples.

    """

    # The calls per captured graph.
    unroll: int

    # The repeat the replay belongs to.
    repeat: int

    # The replay's position within its repeat.
    replay: int

    # The latency per call, in microseconds.
    latency_us: float

    @classmethod
    def from_series(cls, series: Series, case: Case) -> list[Sample]:

        fields = structs.asdict(case)

        return [
            cls(**fields, unroll=series.unroll, repeat=repeat, replay=replay, latency_us=latency)
            for repeat, latencies in enumerate(series.repeats)
            for replay, latency in enumerate(latencies)
        ]


class Heartbeat(Telemetry, kw_only=True):

    """
    One timing of a GPU's held reference graph, with the GPU's telemetry right after it, i.e. one
    row of the run's heartbeats.

    """

    # The NVML index of the GPU.
    device: int

    # When the beat fell: before the GPU's first bucket, between its cases, or after its last.
    phase: Phase

    # The beat time, in UTC.
    time: datetime

    # The median latency per call of the reference, in microseconds.
    median_us: float

    # The 10th percentile latency per call of the reference, in microseconds.
    p10_us: float

    # The 90th percentile latency per call of the reference, in microseconds.
    p90_us: float

    @classmethod
    def initialize(
        cls, device: int, phase: Phase, timing: Timing, telemetry: Telemetry
    ) -> Heartbeat:

        return cls(
            **structs.asdict(telemetry),
            device=device,
            phase=phase,
            time=datetime.now(UTC),
            **structs.asdict(timing),
        )

    @property
    def is_starting(self) -> bool:

        return self.phase == "start"

    @property
    def is_beating(self) -> bool:

        return self.phase == "beat"


class Drift(Struct):

    """
    The drift of one GPU's held reference graph while its latest process timed cases: from the
    median of its first beats between cases to the median of its latest.

    Note(s):

        • The start and end beats are taken with no bucket open, and the start beat times faster
          than the beats between cases, so it marks the step into timing rather than drift.

        • Each end takes the median of up to `WINDOW` beats, fewer when the two would overlap.

    """

    # The NVML index of the GPU.
    device: int

    # The heartbeat as the GPU started timing.
    start: Heartbeat

    # The latest heartbeat of the process, for its telemetry.
    end: Heartbeat

    # The beat(s) taken between cases.
    beats: int

    # The median of the first beats between cases, in microseconds.
    first_us: float

    # The median of the latest beats between cases, in microseconds.
    latest_us: float

    # The relative range of the beats between cases.
    spread: float

    @property
    def change(self) -> float:

        return (self.latest_us / self.first_us) - 1

    @property
    def step(self) -> float:
        """
        The change from the start beat to the first beats between cases.

        """
        return (self.first_us / self.start.median_us) - 1

    @classmethod
    def from_series(cls, series: list[Heartbeat]) -> Drift | None:
        """
        The drift over one timing process's heartbeats, or none before two beats between cases.

        """
        medians = between_cases(series)

        if window := min(WINDOW, len(medians) // 2):

            start, end = series[0], series[-1]
            first, latest = median(medians[:window]), median(medians[-window:])

            return cls(
                device=start.device,
                start=start,
                end=end,
                beats=len(medians),
                first_us=first,
                latest_us=latest,
                spread=relative_range(medians),
            )

    @classmethod
    def from_heartbeats(cls, heartbeats: list[Heartbeat]) -> list[Drift]:
        """
        Each GPU's drift so far, over the beats of its latest timing process.

        """
        latest = valmap(since_start, groupby(attrgetter("device"), heartbeats))
        drifts = (cls.from_series(latest[device]) for device in sorted(latest))

        return list(filter(None, drifts))

    def exceeds(self, tolerance: float) -> bool:

        return abs(self.change) > tolerance


class Completion(Struct):

    # The run end, in UTC.
    end: datetime

    # The case count per status, over every case's latest record.
    statuses: dict[Status, int]

    # The drift of each GPU that timed cases.
    drifts: list[Drift]

    @classmethod
    def initialize(cls, end: datetime, records: RecordLog, drifts: list[Drift]) -> Completion:

        statuses = Counter(map(attrgetter("status"), records.latest.values()))

        return cls(end=end, statuses=dict(sorted(statuses.items())), drifts=drifts)


class Header(Struct):

    """
    The part of a manifest every format version shares.

    """

    # The manifest format version.
    version: int


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

    # The seconds between heartbeats while timing.
    heartbeat: float

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
        heartbeat: float,
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
            heartbeat=heartbeat,
            devices=devices,
            calibration=calibration,
        )

    @classmethod
    def load(cls, path: Path) -> Manifest:

        content = path.read_bytes()

        if (version := toml.decode(content, type=Header).version) != MANIFEST_VERSION:

            raise ValueError(f"{path} has manifest format {version}, not {MANIFEST_VERSION}")

        else:

            return toml.decode(content, type=cls)

    @property
    def tag(self) -> str:

        return f"sm{self.arch}_k{self.sweep.k}"

    @property
    def dispersion(self) -> float:
        """
        The repeatability of the run's timings: the calibrated dispersion, or the tolerance when
        the timer was given.

        """
        if (calibration := self.calibration) is None:

            return self.tolerance

        else:

            return calibration.choice.dispersion

    @property
    def has_finished(self) -> bool:

        return self.completion is not None

    def save(self, path: Path) -> None:

        path.write_bytes(toml.encode(self))


class RunDirectory(Struct):

    # The run directory path.
    path: Path

    # The run manifest.
    manifest: Manifest

    # The run's record(s), in write order.
    records: RecordLog

    # The run's heartbeat(s), in write order.
    heartbeats: list[Heartbeat]

    @classmethod
    def load(cls, path: Path) -> RunDirectory:
        """
        Loads a run at any stage after calibration: before timing starts it has no records or
        heartbeats yet.

        """
        manifest = Manifest.load(path / MANIFEST)
        records = RecordLog.load(log) if (log := path / RECORDS).exists() else RecordLog()

        beats = path / HEARTBEATS
        heartbeats = load_rows(beats, Heartbeat) if beats.exists() else []

        return cls(path=path, manifest=manifest, records=records, heartbeats=heartbeats)

    @property
    def table(self) -> Path:

        return self.path / f"{self.manifest.tag}.toml"

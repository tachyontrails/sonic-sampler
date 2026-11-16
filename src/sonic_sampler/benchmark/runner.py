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

from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from itertools import chain, filterfalse
from multiprocessing import get_context
from multiprocessing.context import SpawnContext, SpawnProcess      # noqa.
from multiprocessing.queues import Queue
from operator import attrgetter, methodcaller
from pathlib import Path
from queue import Empty
from typing import Annotated, Any, Final, Never, override

from loguru import logger
from msgspec import Struct, field, structs
from tyro.conf import arg

from sonic_sampler.benchmark.environment import Device, versions, visible_devices
from sonic_sampler.benchmark.records import (
    CALIBRATION,
    HEARTBEATS,
    MANIFEST,
    RECORDS,
    Completion,
    Drift,
    Heartbeat,
    Manifest,
    Record,
    RecordLog,
    RowSink,
    RunDirectory,
    Sample,
    save_rows,
)
from sonic_sampler.benchmark.sweep import Axis, Bucket, Case, CompileUnit, Limits, Plan, Sweep
from sonic_sampler.benchmark.timing import (
    REPEATS,
    REPLAYS,
    TOLERANCE,
    UNROLLS,
    Calibration,
    GraphTimer,
    Series,
)
from sonic_sampler.benchmark.worker import (
    Beat,
    Calibrated,
    Done,
    Measured,
    Message,
    Probed,
    Started,
    calibrate_timer,
    compile_units,
    measure_buckets,
    probe_device,
)


# The start method of every worker process: CUDA does not survive `fork`.
CONTEXT: Final[SpawnContext] = get_context("spawn")

# The waits for a worker message while polling, and while draining a crashed worker's messages.
POLL_SECONDS: Final[float] = 0.5
FLUSH_SECONDS: Final[float] = 0.1

# The consecutive restarts a server may take without its worker taking up a task.
MAX_RESTARTS: Final[int] = 3

# The default seconds between heartbeats while timing.
HEARTBEAT: Final[float] = 5.0


class RunError(Exception):

    """
    Base class for every failure that stops a benchmark run.

    """


class RunExists(RunError):

    """
    Raised when a fresh run targets a directory that already holds a manifest.

    """


class RunMissing(RunError):

    """
    Raised when a resumed run finds no manifest in its directory.

    """


class UnknownDevices(RunError):

    """
    Raised when the requested GPUs include an index NVML does not report.

    """


class MixedDevices(RunError):

    """
    Raised when the selected GPUs differ in architecture or opt-in shared memory.

    """


class WorkerFailed(RunError):

    """
    Raised when a worker that cannot be replaced exits abnormally or without its report, or one
    keeps exiting.

    """


class WorkerExited(RunError):

    """
    Recorded as the error of the case a timing worker was measuring when its process exited.

    """


def is_non_positive(value: int | None) -> bool:

    return value is not None and value < 1


class Options(Struct):

    """
    The execution settings of a run, apart from the sweep it measures.

    """

    # The NVML index(es) of the GPUs to run on, every GPU on the node when empty.
    gpus: Annotated[tuple[int, ...], arg(aliases=("-g",))] = ()

    # The compile processes per GPU.
    jobs: Annotated[int, arg(aliases=("-j",))] = 8

    # The eager warm-up calls per timed case.
    warmup: int = 3

    # The calls per captured graph, calibrated unless given.
    unroll: int | None = None

    # The timed replays per case, calibrated unless given.
    replays: int | None = None

    # The tolerance on calibrated dispersion and on drift.
    tolerance: float = TOLERANCE

    # The seconds between heartbeats while timing.
    heartbeat: float = HEARTBEAT

    def __post_init__(self) -> None:

        if self.jobs < 1:

            raise ValueError(f"compile jobs must be positive, got {self.jobs} instead")

        elif self.warmup < 0 or is_non_positive(self.unroll) or is_non_positive(self.replays):

            raise ValueError("warm-up must be non-negative, unroll and replay counts positive")

        elif not 0 < self.tolerance < 1:

            raise ValueError(f"tolerance must lie in (0, 1), got {self.tolerance} instead")

        elif self.heartbeat <= 0:

            raise ValueError(f"heartbeat must be positive, got {self.heartbeat} instead")

    @property
    def timer(self) -> GraphTimer | None:
        """
        The timer the options fix, or none when calibration chooses the unroll and replay counts.

        """
        if self.unroll is not None and self.replays is not None:

            return GraphTimer(warmup=self.warmup, unroll=self.unroll, replays=self.replays)

    @property
    def unroll_grid(self) -> Axis:

        return UNROLLS if self.unroll is None else (self.unroll,)

    @property
    def replay_grid(self) -> Axis:

        return REPLAYS if self.replays is None else (self.replays,)


class Placement(Struct):

    """
    Where a worker runs and what it runs: its index within its pool, its GPU, and its entry point.

    """

    # The placement's index within its pool.
    index: int

    # The NVML index of the GPU the worker is pinned to.
    device: int

    # The worker entry point.
    target: Callable[..., None]

    # The entry point's argument(s) after its slot, GPU, and queue(s).
    args: tuple[Any, ...]

    @classmethod
    def distribute(
        cls, gpus: Sequence[int], target: Callable[..., None], args: tuple[Any, ...] = ()
    ) -> list[Placement]:

        return [cls(index=i, device=gpu, target=target, args=args) for i, gpu in enumerate(gpus)]

    def spawn(self, *queues: Queue) -> SpawnProcess:

        args = (self.index, self.device, *queues, *self.args)
        process = CONTEXT.Process(target=self.target, args=args, daemon=True)

        process.start()

        return process


class Slot(Placement):

    """
    A worker process that does one job and exits, e.g. probing its GPU or calibrating the timer.

    """

    # The worker process.
    process: SpawnProcess

    @classmethod
    def start(cls, placement: Placement, results: Queue[Message]) -> Slot:

        return cls(**structs.asdict(placement), process=placement.spawn(results))

    @property
    def exit_code(self) -> int | None:

        return self.process.exitcode

    @property
    def has_finished(self) -> bool:

        return self.exit_code == 0

    @property
    def has_crashed(self) -> bool:

        return self.exit_code is not None and not self.has_finished

    def stop(self) -> None:

        if self.process.is_alive():

            self.process.terminate()

        self.process.join()


class Server[T](Slot):

    """
    A worker process that takes tasks from its own queue until it receives the stop sentinel.

    """

    # The worker's task queue.
    queue: Queue[T | None]

    # The task the worker holds, if any.
    task: T | None = None

    # Flag set once the worker has taken up its task.
    started: bool = False

    # The consecutive restarts without the worker taking up a task.
    restarts: int = 0

    # Flag set once the worker has been sent the stop sentinel.
    closed: bool = False

    @override
    @classmethod
    def start(cls, placement: Placement, results: Queue[Message]) -> Server[T]:

        queue = CONTEXT.Queue()
        process = placement.spawn(queue, results)

        return cls(**structs.asdict(placement), process=process, queue=queue)

    @override
    @property
    def has_finished(self) -> bool:

        return self.closed and self.exit_code == 0

    def assign(self, task: T) -> None:

        self.task, self.started = task, False
        self.queue.put(task)

    def begin(self) -> None:

        self.started, self.restarts = True, 0

    def close(self) -> None:

        self.task, self.closed = None, True
        self.queue.put(None)

    def restart(self, results: Queue[Message]) -> None:

        if self.restarts == MAX_RESTARTS:

            raise WorkerFailed(f"worker {self.index} on GPU {self.device} keeps exiting")

        else:

            message = "worker {} on GPU {} exited with code {}, restarting"

            logger.warning(message, self.index, self.device, self.exit_code)

            self.restarts += 1
            self.queue.cancel_join_thread()

            self.queue, self.task = CONTEXT.Queue(), None
            self.started = self.closed = False

            self.process = self.spawn(self.queue, results)


class Phase[T](Struct):

    """
    The bookkeeping of one pool run: what each message means, and what becomes of the task of a
    server whose process crashed.

    """

    def observe(self, message: Message) -> None:

        raise NotImplementedError

    def recover(self, server: Server[T]) -> list[T]:

        raise NotImplementedError


class Pool[T](Struct):

    """
    Servers that each take the next pending task as they free up, beside one-shot slots, all
    reporting to one shared result queue.

    Note(s):

        • A crashed server's messages are drained before its task is recovered, so the phase sees
          every record the server managed to send. The phase decides what of the task to requeue,
          then the server restarts.

        • A one-shot slot that crashes stops the pool, since its job has no task to requeue.

        • The queue is drained once every worker has exited, so no final message is lost.

    """

    # The task-taking server(s), each at its own index.
    servers: list[Server[T]]

    # The one-shot slot(s), indexed after the servers.
    slots: list[Slot]

    # The task(s) not yet handed out, in dispatch order.
    pending: deque[T]

    # The shared result queue.
    results: Queue[Message]

    @classmethod
    @contextmanager
    def start(
        cls,
        servers: list[Placement],
        tasks: list[T],
        jobs: list[Placement] | None = None,
    ) -> Iterator[Pool[T]]:

        results = CONTEXT.Queue()

        started = [Server.start(placement, results) for placement in servers]
        slots = [Slot.start(placement, results) for placement in (jobs or [])]

        try:

            yield cls(servers=started, slots=slots, pending=deque(tasks), results=results)

        finally:

            for worker in chain(started, slots):

                worker.stop()

    @property
    def is_active(self) -> bool:

        return not all(map(attrgetter("has_finished"), chain(self.servers, self.slots)))

    @property
    def crashed_slots(self) -> Iterator[Slot]:

        return filter(attrgetter("has_crashed"), self.slots)

    @property
    def crashed_servers(self) -> Iterator[Server[T]]:

        return filter(attrgetter("has_crashed"), self.servers)

    def dispatch(self, server: Server[T]) -> None:

        if self.pending:

            server.assign(self.pending.popleft())

        else:

            server.close()

    def poll(self, timeout: float) -> Message | None:

        try:

            return self.results.get(timeout=timeout)

        except Empty:

            return None

    def handle(self, message: Message, phase: Phase[T]) -> None:

        match message:

            case Started(slot=index):

                self.servers[index].begin()

            case Done(slot=index):

                self.dispatch(self.servers[index])

        phase.observe(message)

    def flush(self, phase: Phase[T]) -> None:

        while (message := self.poll(FLUSH_SECONDS)) is not None:

            self.handle(message, phase)

    def recover(self, server: Server[T], phase: Phase[T]) -> None:

        self.flush(phase)
        self.pending.extendleft(reversed(phase.recover(server)))

        server.restart(self.results)
        self.dispatch(server)

    def step(self, phase: Phase[T]) -> None:

        if (message := self.poll(POLL_SECONDS)) is not None:

            self.handle(message, phase)

        if crashed := next(self.crashed_slots, None):

            name, code = crashed.target.__name__, crashed.exit_code

            raise WorkerFailed(f"{name} on GPU {crashed.device} exited with code {code}")

        for server in self.crashed_servers:

            self.recover(server, phase)

    def run(self, phase: Phase[T]) -> None:

        for server in self.servers:

            self.dispatch(server)

        while self.is_active:

            self.step(phase)

        # A worker's last messages can still be queued when it exits, e.g. a probe's report or a
        # final drift reference. Each worker flushes its queue before exiting, so they are here.

        self.flush(phase)


class Survey(Phase[Never]):

    # The probed GPU(s), by slot.
    devices: dict[int, Device] = field(default_factory=dict)

    @override
    def observe(self, message: Message) -> None:

        if isinstance(message, Probed):

            self.devices[message.slot] = message.device


class Preparation(Phase[CompileUnit]):

    # The calibration, once reported.
    calibration: Calibration | None = None

    # The timed replays the calibration was scored from, once reported.
    series: list[Series] = field(default_factory=list)

    @override
    def observe(self, message: Message) -> None:

        if isinstance(message, Calibrated):

            self.calibration, self.series = message.calibration, message.series

    def samples(self, probes: list[Case]) -> Iterator[Sample]:
        """
        Every timed replay of the calibration, as rows.

        """
        return chain.from_iterable(Sample.from_series(s, probes[s.probe]) for s in self.series)

    @override
    def recover(self, server: Server[CompileUnit]) -> list[CompileUnit]:

        if not server.started:

            return [server.task] if server.task else []

        else:

            # The unit's kernels then compile when its cases are first timed.

            message = "compiling {} on GPU {} crashed, skipping it"

            logger.warning(message, server.task, server.device)

            return []


class Measurement(Phase[Bucket]):

    """
    The bookkeeping of the timing pool: records and heartbeats as they arrive, and what becomes of
    a crashed worker's bucket.

    Note(s):

        • Every case of a bucket is timed by one process, so a bucket that a crashed worker had
          taken up is timed again in full by its replacement, except the case that crashed it.

    """

    # The record sink of the run.
    records: RowSink[Record]

    # The heartbeat sink of the run.
    heartbeats: RowSink[Heartbeat]

    # The bucket(s) to time.
    buckets: int

    # The bucket(s) timed so far.
    finished: int = 0

    # The record(s) of each worker's current bucket, by slot.
    attempts: dict[int, list[Record]] = field(default_factory=dict)

    # The heartbeat(s) so far, in arrival order.
    beats: list[Heartbeat] = field(default_factory=list)

    @property
    def drifts(self) -> list[Drift]:
        """
        Each GPU's drift over its latest timing process: a replacement process captures its own
        reference graph, so its drift starts afresh.

        """
        return Drift.from_heartbeats(self.beats)

    def finish(self) -> None:

        self.finished += 1

        logger.info("timed {}/{} bucket(s)", self.finished, self.buckets)

    def beat(self, heartbeat: Heartbeat) -> None:

        self.heartbeats(heartbeat)
        self.beats.append(heartbeat)

    def blame(self, server: Server[Bucket], bucket: Bucket) -> Case | None:
        """
        The case that ended the worker's process, recorded as failed unless its own record says
        so already, or none when the worker timed every case of its bucket.

        Note(s):

            • A failed last record means its case left the CUDA context unusable, so the process
              exited on synchronizing after it. Otherwise it exited timing the next case.

        """
        records = self.attempts[server.index]
        cases = bucket.cases

        if records and not records[-1].is_viable:

            return cases[len(records) - 1]

        elif len(records) < len(cases):

            culprit, code = cases[len(records)], server.exit_code
            error = WorkerExited(f"worker on GPU {server.device} exited with code {code}")

            self.records(Record.failed(culprit, server.device, error))

            return culprit

    @override
    def observe(self, message: Message) -> None:

        match message:

            case Started(slot=slot):

                self.attempts[slot] = []

            case Measured(slot=slot, record=record):

                self.records(record)
                self.attempts[slot].append(record)

            case Beat(heartbeat=heartbeat):

                self.beat(heartbeat)

            case Done():

                self.finish()

    @override
    def recover(self, server: Server[Bucket]) -> list[Bucket]:

        if (bucket := server.task) is None:

            return []

        elif not server.started:

            return [bucket]

        elif (culprit := self.blame(server, bucket)) is None or bucket.cases == [culprit]:

            self.finish()

            return []

        else:

            # The bucket is requeued ahead of every other bucket, and its replacement times it
            # again in full, except the culprit.

            rest = [case for case in bucket.cases if case != culprit]

            return [structs.replace(bucket, cases=rest)]


def survey(gpus: tuple[int, ...]) -> list[Device]:

    probes = Placement.distribute(gpus=gpus, target=probe_device)

    with Pool.start([], [], probes) as pool:

        pool.run(phase := Survey())

    if missing := [probe.device for probe in probes if probe.index not in phase.devices]:

        raise WorkerFailed(f"probing GPU(s) {missing} exited without reporting")

    elif len(set(map(attrgetter("model"), devices := list(phase.devices.values())))) > 1:

        raise MixedDevices(f"GPU(s) differ in architecture or shared memory: {devices}")

    else:

        return [phase.devices[probe.index] for probe in probes]


class Runner(Struct):

    """
    Runs a sweep over a node's GPUs into a run directory: it probes the GPUs, compiles every
    kernel while calibrating the timer, then times every bucket.

    Note(s):

        • Each phase starts fresh worker processes, and the runner never creates a CUDA context.

        • Records are flushed as they arrive, so an interrupted run resumes from its last record.

    """

    # The run directory.
    directory: Path

    # The sweep specification.
    sweep: Sweep

    # The execution options.
    options: Options

    # The run start, in UTC.
    start: datetime

    # The record(s) of earlier attempts at the run.
    records: RecordLog

    # The limits of the run being resumed, which keep its plan whatever GPUs resume it.
    limits: Limits | None = None

    # The fixed timer: given explicitly, or reused from an earlier attempt.
    timer: GraphTimer | None = None

    # The calibration of an earlier attempt, if it calibrated.
    calibration: Calibration | None = None

    @classmethod
    def initialize(cls, directory: Path, sweep: Sweep, options: Options) -> Runner:

        if (directory / MANIFEST).exists():

            raise RunExists(f"{directory} already holds a run, resume it instead")

        else:

            start = datetime.now(UTC)

            return cls(
                directory=directory,
                sweep=sweep,
                options=options,
                start=start,
                records=RecordLog(),
                timer=options.timer,
            )

    @classmethod
    def resume(
        cls,
        directory: Path,
        gpus: tuple[int, ...] = (),
        jobs: int | None = None,
        heartbeat: float | None = None,
    ) -> Runner:
        """
        Resumes the run in the directory with its own sweep, limits, timer, tolerance, compile
        jobs, and heartbeat: only the GPUs change, and the compile jobs or heartbeat when given.

        """
        if not (path := directory / MANIFEST).exists():

            raise RunMissing(f"{directory} holds no run to resume")

        else:

            manifest = Manifest.load(path)
            records = RecordLog.load(log) if (log := directory / RECORDS).exists() else RecordLog()

            options = Options(
                gpus=gpus,
                jobs=manifest.compile_jobs if jobs is None else jobs,
                warmup=manifest.timer.warmup,
                tolerance=manifest.tolerance,
                heartbeat=manifest.heartbeat if heartbeat is None else heartbeat,
            )

            return cls(
                directory=directory,
                sweep=manifest.sweep,
                options=options,
                start=manifest.start,
                records=records,
                limits=manifest.limits,
                timer=manifest.timer,
                calibration=manifest.calibration,
            )

    @property
    def has_settled(self) -> bool:
        """
        Whether the run being resumed has settled every bucket, so nothing is left to time.

        """
        limits = self.limits

        return limits is not None and not self.unsettled(Plan.initialize(self.sweep, limits))

    def unsettled(self, plan: Plan) -> list[Bucket]:
        """
        The buckets left to time, heaviest first: a bucket with any unsettled case is timed again
        in full, so that every case of a bucket is timed by one process.

        """
        pending = filterfalse(methodcaller("is_settled", self.records.settled), plan.buckets)

        return sorted(pending, key=attrgetter("cost"), reverse=True)

    def compilers(self, gpus: list[int], units: list[CompileUnit]) -> list[Placement]:
        """
        The compile slots: GPUs take jobs in turn, and no more jobs start than there are units.

        """
        rounds = [gpu for _ in range(self.options.jobs) for gpu in gpus]
        jobs = rounds[:len(units)]

        return Placement.distribute(gpus=jobs, target=compile_units, args=(self.sweep,))

    def calibrator(self, index: int, gpu: int, cases: list[Case]) -> Placement:

        grids = ((options := self.options).unroll_grid, options.replay_grid)
        args = (self.sweep, cases, options.warmup, options.tolerance, *grids, REPEATS)

        return Placement(index=index, device=gpu, target=calibrate_timer, args=args)

    def probe(self) -> list[Device]:

        visible = visible_devices()
        gpus = self.options.gpus or visible

        if missing := (set(gpus) - set(visible)):

            raise UnknownDevices(
                f"GPU(s) {sorted(missing)} not among the node's GPUs {list(visible)}"
            )

        else:

            return survey(gpus)

    def prepare(
        self, plan: Plan, gpus: list[int], units: list[CompileUnit]
    ) -> tuple[GraphTimer, Calibration | None]:

        if (timer := self.timer) is not None:

            logger.info("compiling {} unit(s) on {} GPU(s)", len(units), len(gpus))

            with Pool.start(self.compilers(gpus, units), units) as pool:

                pool.run(Preparation())

            return timer, self.calibration

        else:

            # Calibration has its GPU to itself, so compiling runs on the others, or after
            # calibration when there is only one GPU.

            calibrator, *others = gpus

            compilers = self.compilers(others, units)
            message = "calibrating on GPU {}, compiling {} unit(s) on {} GPU(s)"

            logger.info(message, calibrator, len(units), len(others))

            jobs = [self.calibrator(len(compilers), calibrator, plan.probes)]

            with Pool.start(compilers, units if others else [], jobs) as pool:

                pool.run(preparation := Preparation())

            # Checked before a lone GPU compiles, so a calibrator without its report fails early.

            if (calibration := preparation.calibration) is None:

                raise WorkerFailed(f"calibrating on GPU {calibrator} exited without reporting")

            else:

                message = "calibrated {}, converged: {}"

                logger.info(message, calibration.choice, calibration.has_converged)

                # Every timed replay is kept, so the calibration can be scored again offline.

                save_rows(self.directory / CALIBRATION, Sample, preparation.samples(plan.probes))

            if not others:

                with Pool.start(self.compilers(gpus, units), units) as pool:

                    pool.run(Preparation())

            return calibration.timer(self.options.warmup), calibration

    def measure(
        self, plan: Plan, gpus: list[int], buckets: list[Bucket], timer: GraphTimer
    ) -> list[Drift]:

        args = (self.sweep, timer, plan.reference, self.options.heartbeat)
        timers = Placement.distribute(gpus=gpus, target=measure_buckets, args=args)

        logger.info("timing {} bucket(s) on {} GPU(s)", len(buckets), len(gpus))

        with (
            RowSink.open(self.directory / RECORDS, Record) as records,
            RowSink.open(self.directory / HEARTBEATS, Heartbeat) as heartbeats,
            Pool.start(timers, buckets) as pool,
        ):

            measurement = Measurement(records=records, heartbeats=heartbeats, buckets=len(buckets))

            pool.run(measurement)

        return measurement.drifts

    def run(self) -> Manifest:
        """
        Attempts the run, or concludes a resumed run with nothing left to time without starting
        a single process.

        """
        self.directory.mkdir(parents=True, exist_ok=True)

        return self.conclude() if self.has_settled else self.attempt()

    def attempt(self) -> Manifest:
        """
        One attempt at the run: it probes the GPUs, compiles while calibrating, then times every
        bucket left.

        """
        devices = self.probe()
        gpus = list(map(attrgetter("index"), devices))

        limits = self.limits or Limits.resolve(self.sweep, devices[0].shared_memory)
        plan = Plan.initialize(self.sweep, limits)

        # Only the units of the buckets left are compiled.

        buckets = self.unsettled(plan)
        pending = {case.key for bucket in buckets for case in bucket.cases}

        compiled = [unit for unit in plan.units if unit.key in pending]
        units = sorted(compiled, key=attrgetter("kernels"), reverse=True)

        timer, calibration = self.prepare(plan, gpus, units)

        manifest = Manifest.initialize(
            self.start,
            versions(),
            self.sweep,
            limits,
            timer,
            (options := self.options).tolerance,
            options.jobs,
            options.heartbeat,
            devices,
            calibration,
        )

        manifest.save(self.directory / MANIFEST)

        return self.complete(manifest, self.measure(plan, gpus, buckets, timer))

    def conclude(self) -> Manifest:
        """
        The manifest of a resumed run with every bucket settled: as it stands once finished, or
        completed from the run's heartbeats when it stopped after its last record.

        """
        run = RunDirectory.load(self.directory)

        if (manifest := run.manifest).has_finished:

            logger.info("every bucket is settled, so nothing is left to time")

            return manifest

        else:

            return self.complete(manifest, Drift.from_heartbeats(run.heartbeats))

    def complete(self, manifest: Manifest, drifts: list[Drift]) -> Manifest:
        """
        Saves the manifest with the run's completion, warning of each GPU's drift beyond the
        tolerance.

        """
        tolerance = self.options.tolerance
        drifted = methodcaller("exceeds", tolerance)

        for drift in filter(drifted, drifts):

            message = "GPU {} drifted {:+.3%} between cases, beyond the {:.3%} tolerance"

            logger.warning(message, drift.device, drift.change, tolerance)

        records = RecordLog.load(self.directory / RECORDS)

        completion = Completion.initialize(datetime.now(UTC), records, drifts)
        complete = structs.replace(manifest, completion=completion)

        complete.save(self.directory / MANIFEST)

        logger.info("finished: {}", completion.statuses)

        return complete

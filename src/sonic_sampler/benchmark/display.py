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
from collections.abc import Iterable
from operator import attrgetter
from pathlib import Path
from typing import Any, Final

from cytoolz.itertoolz import groupby
from msgspec import Struct
from rich.box import SIMPLE_HEAD
from rich.console import Group
from rich.table import Table
from rich.text import Text

from sonic_sampler.benchmark.base import relative_range
from sonic_sampler.benchmark.records import Drift, RunDirectory, since_start
from sonic_sampler.benchmark.reduction import Reduction
from sonic_sampler.benchmark.sweep import Plan
from sonic_sampler.interface.dispatch import DispatchSummary


# The nanoseconds per millisecond, for the clock event counters.
NANOSECONDS: Final[int] = 1_000_000

# The column headings of a bucket's priorities.
PRIORITIES: Final[tuple[str, ...]] = ("1st", "2nd", "3rd")

# The placeholder of a value not measured or not reported.
MISSING: Final[str] = "-"


def label(config: Any) -> str:
    """
    A record's or dispatch configuration's tiling and warp counts, e.g. `bitonic n=4096 w=8/16`.

    """
    return f"{config.strategy} n={config.block_n} w={config.first_warps}/{config.second_warps}"


def held_back(start: int | None, end: int | None) -> str:

    return MISSING if start is None or end is None else f"{(end - start) / NANOSECONDS:.1f}"


def heading(*parts: str) -> Text:

    return Text(" · ".join(parts), style="bold")


def tabulate(*columns: str, title: str | None = None) -> Table:
    """
    A table in the one style of every view: a rule under the header, a left-aligned title.

    """
    return Table(*columns, title=title, title_justify="left", box=SIMPLE_HEAD, pad_edge=False)


def bucket_rows(cells: Iterable[tuple[int, int, list[str]]], title: str | None = None) -> Table:
    """
    One row per vocabulary and batch size, its configurations in priority order.

    """
    table = tabulate("V", "B", *PRIORITIES, title=title)

    for vocab_size, batch_size, configs in cells:

        padded = configs + [MISSING] * (len(PRIORITIES) - len(configs))

        table.add_row(str(vocab_size), str(batch_size), *padded)

    return table


class RunView(Struct):

    """
    One run directory: what it ran on and with, how far it got, and how each GPU drifted.

    """

    # The run.
    run: RunDirectory

    # The plan of the run's sweep and limits.
    plan: Plan

    @classmethod
    def load(cls, path: Path) -> RunView:

        manifest = (run := RunDirectory.load(path)).manifest

        return cls(run=run, plan=Plan.initialize(manifest.sweep, manifest.limits))

    @property
    def header(self) -> Group:

        manifest = self.run.manifest
        started = f"started {manifest.start:%Y-%m-%d %H:%M:%S} UTC"

        if (completion := manifest.completion) is None:

            state = "in progress"

        else:

            state = f"finished {completion.end:%H:%M:%S} UTC"

        timer = manifest.timer

        settings = (
            f"timer: {timer.warmup} warm-up, {timer.unroll} unrolled, {timer.replays} replays",
            f"tolerance {manifest.tolerance:.2%}",
            f"heartbeat {manifest.heartbeat} s",
            f"{manifest.compile_jobs} compile jobs",
        )

        versions = " · ".join(f"{name} {version}" for name, version in manifest.versions.items())

        return Group(
            heading(str(self.run.path), manifest.tag, started, state),
            Text(versions, style="dim"),
            Text(" · ".join(settings)),
        )

    @property
    def devices(self) -> Table:

        table = tabulate("GPU", "name", "SM clock", "memory clock", "power limit")

        for device in self.run.manifest.devices:

            clocks = (f"{device.sm_clock}/{device.max_sm_clock} MHz", f"{device.memory_clock} MHz")
            power = f"{device.power_limit:.0f}/{device.default_power_limit:.0f} W"

            table.add_row(str(device.index), device.name, *clocks, power)

        return table

    @property
    def calibration(self) -> Table | None:

        if (calibration := self.run.manifest.calibration) is None:

            return None

        else:

            state = "converged" if calibration.has_converged else "not converged"
            columns = ("N", "R", "cost", "dispersion", "spread", "")

            table = tabulate(*columns, title=f"calibration, {state}")

            for trial in calibration.trials:

                chosen = trial == calibration.choice
                cells = (trial.unroll, trial.replays, trial.cost, f"{trial.dispersion:.3%}")

                table.add_row(
                    *map(str, cells),
                    f"{trial.spread:.3%}",
                    "chosen" if chosen else "",
                    style="bold" if chosen else None,
                )

            return table

    @property
    def progress(self) -> Text:

        latest = self.run.records.latest
        statuses = Counter(map(attrgetter("status"), latest.values()))

        settled = self.run.records.settled
        complete = sum(bucket.is_settled(settled) for bucket in self.plan.buckets)

        counts = ", ".join(f"{status} {count}" for status, count in sorted(statuses.items()))
        cases = f"cases {len(latest)}/{self.plan.size}" + (f" ({counts})" if counts else "")

        return Text(f"{cases} · buckets settled {complete}/{len(self.plan.buckets)}")

    @property
    def drift(self) -> Table:
        """
        Each GPU's drift so far, over its latest timing process.

        """
        tolerance = self.run.manifest.tolerance
        series = groupby(attrgetter("device"), self.run.heartbeats)

        table = tabulate(
            "GPU",
            "beats",
            "start",
            "latest",
            "drift",
            "spread",
            "SM clock",
            "GPU temp",
            "held back, power / thermal",
            title="drift of the held reference, in µs, MHz, °C and ms",
        )

        for drift in Drift.from_heartbeats(self.run.heartbeats):

            start, end = drift.start, drift.end
            beats = since_start(series[drift.device])

            capping = held_back(start.power_capping_ns, end.power_capping_ns)
            thermal = held_back(start.thermal_slowdown_ns, end.thermal_slowdown_ns)

            table.add_row(
                str(drift.device),
                str(len(beats)),
                f"{start.median_us:.2f}",
                f"{end.median_us:.2f}",
                Text(f"{drift.change:+.3%}", style="red" if drift.exceeds(tolerance) else ""),
                f"{relative_range([beat.median_us for beat in beats]):.3%}",
                f"{start.sm_clock} -> {end.sm_clock}",
                f"{start.gpu_temp} -> {end.gpu_temp}",
                f"{capping} / {thermal}",
            )

        return table

    def __rich__(self) -> Group:

        parts = (self.header, self.devices, self.calibration, self.progress, self.drift)

        return Group(*(part for part in parts if part is not None), Text())


class ReductionView(Struct):

    """
    The merged standings of one or more runs: the winners of each bucket, and what to be wary of
    before trusting them.

    """

    # The reduction of the run(s).
    reduction: Reduction

    @property
    def standings(self) -> Table:

        cells = (
            (
                standing.vocab_size,
                standing.batch_size,
                [f"{label(w)} {w.median_us:.2f} µs" for w in standing.winners],
            )
            for standing in self.reduction.standings
        )

        return bucket_rows(cells, title=f"standings of {len(self.reduction.runs)} run(s)")

    @property
    def near_ties(self) -> Table | None:

        if ties := self.reduction.near_ties:

            columns = ("V", "B", "faster", "slower", "margin", "dispersion")
            table = tabulate(*columns, title="near-ties")

            for tie in ties:

                faster, slower = tie.faster, tie.slower
                sizes = (str(faster.vocab_size), str(faster.batch_size))

                margins = (f"{tie.margin:.3%}", f"{tie.dispersion:.3%}")

                table.add_row(*sizes, label(faster), label(slower), *margins, style="yellow")

            return table

    @property
    def findings(self) -> Table | None:

        if findings := self.reduction.findings:

            table = tabulate("V", "B", "winner", "concerns", title="register findings")

            for finding in findings:

                record = finding.record
                sizes = (str(record.vocab_size), str(record.batch_size))

                table.add_row(*sizes, label(record), ", ".join(finding.concerns))

            return table

    @property
    def warnings(self) -> list[Text]:

        reduction = self.reduction
        running = [str(run.path) for run in reduction.runs if not run.manifest.has_finished]

        gaps = [f"V={s.vocab_size} B={s.batch_size}" for s in reduction.gaps]

        notes = (
            (running, f"still in progress or interrupted: {', '.join(running)}"),
            (gaps, f"buckets missing a strategy, so not yet reducible: {', '.join(gaps)}"),
            (reduction.mismatches, f"package versions differ across runs: {reduction.mismatches}"),
            (reduction.is_multi_timer, "the runs were timed with different timers"),
        )

        return [Text(f"warning: {note}", style="red") for raised, note in notes if raised]

    def __rich__(self) -> Group:

        parts = (self.standings, self.near_ties, self.findings, *self.warnings)

        return Group(*(part for part in parts if part is not None), Text())


class TableView(Struct):

    """
    A dispatch table: the configuration of each priority, per vocabulary and batch size.

    """

    # The table's path.
    path: Path

    # The dispatch table.
    summary: DispatchSummary

    @classmethod
    def load(cls, path: Path) -> TableView:

        return cls(path=path, summary=DispatchSummary.read(path, inference=False))

    def __rich__(self) -> Group:

        summary = self.summary
        metadata = summary.metadata

        cells = (
            (vocab.size, batch.size, list(map(label, batch.config)))
            for vocab in summary.vocab for batch in vocab.batch
        )

        title = heading(str(self.path), f"k={summary.k}", f"format {metadata.version}")
        text = Text(f"measured {metadata.timestamp}", style="dim")

        return Group(title, text, bucket_rows(cells))

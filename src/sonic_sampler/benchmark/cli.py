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

from pathlib import Path
from typing import Annotated

from msgspec import Struct, field
from rich.console import Console
from rich.text import Text
from tyro.conf import OmitArgPrefixes, Positional, arg, subcommand

from sonic_sampler.benchmark.display import ReductionView, RunView, TableView
from sonic_sampler.benchmark.records import MANIFEST, RunDirectory
from sonic_sampler.benchmark.reduction import Reduction
from sonic_sampler.benchmark.runner import HEARTBEAT, Options, RunError, Runner
from sonic_sampler.benchmark.sweep import Sweep


class CommandError(Exception):

    """
    Raised when a command's arguments cannot be acted on, e.g. a table it may not overwrite.

    """


def missing_manifest(paths: tuple[Path, ...]) -> list[str]:

    return [str(p) for p in paths if not (p / MANIFEST).exists()]


def load_runs(paths: tuple[Path, ...]) -> list[RunDirectory]:

    if not paths:

        raise CommandError("no run directory given")

    elif missing := missing_manifest(paths):

        raise CommandError(f"no run manifest in {', '.join(missing)}")

    else:

        return list(map(RunDirectory.load, paths))


def reduce_runs(runs: list[RunDirectory]) -> Reduction:

    try:

        return Reduction.initialize(runs)

    except ValueError as error:

        raise CommandError(str(error)) from error


class Run(Struct):

    """
    Sweeps the top-k kernels over the node's GPUs into a new run directory.

    """

    # The run directory, which must not hold a run yet.
    directory: Positional[Path]

    # The sweep specification.
    sweep: OmitArgPrefixes[Sweep] = field(default_factory=Sweep)

    # The execution settings.
    execution: OmitArgPrefixes[Options] = field(default_factory=Options)

    def __call__(self, console: Console) -> None:

        Runner.initialize(self.directory, self.sweep, self.execution).run()

        console.print(RunView.load(self.directory))


class Resume(Struct):

    """
    Resumes a run with its own sweep, timer, and tolerance, timing every bucket it left
    incomplete.

    """

    # The run directory.
    directory: Positional[Path]

    # The NVML index(es) of the GPUs to run on, every GPU on the node when empty.
    gpus: Annotated[tuple[int, ...], arg(aliases=("-g",))] = ()

    # The compile processes per GPU.
    jobs: Annotated[int, arg(aliases=("-j",))] = 8

    # The seconds between heartbeats while timing.
    heartbeat: float = HEARTBEAT

    def __call__(self, console: Console) -> None:

        options = Options(gpus=self.gpus, jobs=self.jobs, heartbeat=self.heartbeat)

        Runner.resume(self.directory, options).run()

        console.print(RunView.load(self.directory))


class Reduce(Struct):

    """
    Ranks the records of one or more runs into a dispatch table, a later run's record of a case
    superseding an earlier one's.

    """

    # The run directories.
    runs: Positional[tuple[Path, ...]]

    # The dispatch table to write, by default `sm{cc}_k{k}.toml` in the latest run directory.
    output: Annotated[Path | None, arg(aliases=("-o",))] = None

    # Flag to overwrite an existing dispatch table.
    overwrite: bool = True

    def __call__(self, console: Console) -> None:

        reduction = reduce_runs(load_runs(self.runs))
        path = self.output or reduction.latest.table

        if path.exists() and not self.overwrite:

            raise CommandError(f"{path} exists, and --no-overwrite keeps it")

        elif gaps := reduction.gaps:

            buckets = ", ".join(f"V={s.vocab_size} B={s.batch_size}" for s in gaps)

            raise CommandError(f"bucket(s) {buckets} lack a measured case for some strategy")

        else:

            reduction.summary.save(path)

            console.print(ReductionView(reduction=reduction))
            console.print(Text(f"wrote {path}", style="bold"))


class Show(Struct):

    """
    Renders runs and their standings so far, a dispatch table, or both.

    """

    # The run directories to render, merged oldest first for their standings.
    runs: tuple[Path, ...] = ()

    # The dispatch table to render.
    table: Path | None = None

    def __call__(self, console: Console) -> None:

        if not self.runs and self.table is None:

            raise CommandError("nothing to show, pass --runs, --table, or both")

        if self.runs:

            runs = load_runs(self.runs)
            views = [RunView.load(run.path) for run in runs]

            console.print(*views, ReductionView(reduction=reduce_runs(runs)))

        if self.table is not None:

            console.print(TableView.load(self.table))


type Bench = (
    Annotated[Run, subcommand("run")]
    | Annotated[Resume, subcommand("resume")]
    | Annotated[Reduce, subcommand("reduce")]
    | Annotated[Show, subcommand("show")]
)


def execute(command: Run | Resume | Reduce | Show) -> int:
    """
    Runs a bench command, reporting a run or command error as a message and exit code 1.

    """
    console = Console()

    try:

        command(console)

    except (CommandError, RunError) as error:

        console.print(Text.assemble(("error: ", "bold red"), str(error)))

        return 1

    else:

        return 0

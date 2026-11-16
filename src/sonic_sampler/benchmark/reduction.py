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

from collections.abc import Callable
from itertools import chain, pairwise
from operator import attrgetter
from typing import Final, Literal

from cytoolz.dicttoolz import valmap
from cytoolz.functoolz import curry
from cytoolz.itertoolz import first, groupby
from msgspec import Struct

from sonic_sampler.benchmark.records import Record, RunDirectory
from sonic_sampler.benchmark.sweep import STRATEGIES, WARPS, Case, Limits
from sonic_sampler.interface.dispatch import (
    BatchBucket,
    DispatchMetadata,
    DispatchSummary,
    PriorityBucket,
    Strategy,
    VocabBucket,
)


type Concern = Literal["first_spill", "second_spill", "first_floor", "second_floor", "block_cap"]


# The dispatch table format version.
DISPATCH_VERSION: Final[int] = 1


def ensure_compatible(runs: list[RunDirectory]) -> None:

    targets = {(run.manifest.arch, run.manifest.sweep.k) for run in runs}

    if len(targets) != 1:

        raise ValueError(
            f"runs must share one architecture and top-k value, got {sorted(targets)}"
        )


@curry
def is_multi_version(versions: list[dict[str, str]], package: str) -> bool:

    return len({v.get(package) for v in versions}) > 1


class NearTie(Struct):

    """
    Two records too close to rank with confidence: the slower one's margin lies within the
    dispersion of the runs that timed them, or their latency spreads overlap.

    Note(s):

        • The calibrated dispersion measures how far one capture's median moves across captures,
          the larger part of the noise, while a record's p10/p90 spread covers one capture alone.

    """

    # The faster record.
    faster: Record

    # The slower record.
    slower: Record

    # The dispersion the margin is judged against: the larger of the two records' runs.
    dispersion: float

    @property
    def margin(self) -> float:

        return (self.slower.median_us / self.faster.median_us) - 1

    @property
    def is_within(self) -> bool:
        """
        Whether the margin lies within the dispersion.

        """
        return self.margin <= self.dispersion

    @property
    def has_overlap(self) -> bool:
        """
        Whether the latency spreads overlap.

        """
        return self.faster.tail_overlap(self.slower)

    @property
    def is_close(self) -> bool:

        return self.is_within or self.has_overlap

    @property
    def tests(self) -> list[str]:
        """
        The test(s) that find the pair close, by name.

        """
        named = (("margin", self.is_within), ("overlap", self.has_overlap))

        return [name for name, passed in named if passed]


class Finding(Struct):

    # The audited dispatch record.
    record: Record

    # The concern(s) raised against the record.
    concerns: list[Concern]

    @classmethod
    def initialize(cls, record: Record, limits: Limits) -> Finding:

        first_warps = limits.first_warps(record.strategy, record.block_n)
        second_warps = limits.second_warps(block_v := record.group.block_v)

        # A warp floor is flagged when the record sits on the smallest admitted warp count and the
        # per-warp rule pruned a smaller one.

        checks = dict(
            first_spill=bool(record.first_spills),
            second_spill=bool(record.second_spills),
            first_floor=record.first_warps == first_warps[0] > WARPS[0],
            second_floor=record.second_warps == second_warps[0] > WARPS[0],
            block_cap=block_v == limits.max_blocks,
        )

        concerns = [concern for concern, raised in checks.items() if raised]

        return cls(record=record, concerns=concerns)


class Standing(Struct):

    """
    The measured record(s) of one vocabulary and batch size, ranked per strategy.

    """

    # The vocabulary size.
    vocab_size: int

    # The batch size, i.e. the row count.
    batch_size: int

    # The measured record(s) per strategy, fastest first.
    entries: dict[Strategy, list[Record]]

    @classmethod
    def initialize(cls, vocab_size: int, batch_size: int, records: list[Record]) -> Standing:

        ordered = sorted(records, key=attrgetter("ordering"))
        entries = groupby(attrgetter("strategy"), ordered)

        return cls(vocab_size=vocab_size, batch_size=batch_size, entries=entries)

    @property
    def is_incomplete(self) -> bool:

        return self.entries.keys() != STRATEGIES.keys()

    @property
    def winners(self) -> list[Record]:

        return sorted(map(first, self.entries.values()), key=attrgetter("ordering"))

    @property
    def bucket(self) -> BatchBucket:

        configs = [record.config(priority) for priority, record in enumerate(self.winners)]

        return BatchBucket(size=self.batch_size, config=configs)

    def near_ties(self, dispersion: Callable[[Record], float]) -> list[NearTie]:
        """
        The close pairs among adjacent winners and each strategy's two fastest records, judged
        against the dispersion of the run behind each record.

        """
        runners = (records[:2] for records in self.entries.values() if len(records) > 1)
        pairs = chain(pairwise(self.winners), runners)

        ties = (
            NearTie(faster, slower, dispersion=max(dispersion(faster), dispersion(slower)))
            for faster, slower in pairs
        )

        return list(filter(attrgetter("is_close"), ties))


class Reduction(Struct):

    """
    The merged record(s) of one or more runs on one architecture and top-k value, ranked into a
    dispatch summary.

    Note(s):

        • Runs merge oldest first, so a case measured again in a later run keeps the later record.

        • Each surviving record is audited against the limits of the run that produced it.

        • Standings may lack a strategy while a run is in progress, so only the summary demands
          every bucket complete.

    """

    # The merged run(s), oldest first.
    runs: list[RunDirectory]

    # The index of the run that produced each surviving record, by case.
    origins: dict[Case, int]

    # The per-bucket standing(s), ordered by vocabulary then batch size.
    standings: list[Standing]

    @classmethod
    def initialize(cls, runs: list[RunDirectory]) -> Reduction:

        # Verify that runs share a common architecture and top-k value.

        ensure_compatible(runs)

        ordered = sorted(runs, key=attrgetter("manifest.start"))

        survivors = (
            {
                record.case: (index, record)
                for index, run in enumerate(ordered) for record in run.records
            }
        )

        origins = valmap(first, survivors)
        measured = [record for _, record in survivors.values() if record.is_ok]

        buckets = groupby(attrgetter("vocab_size", "batch_size"), measured)

        standings = [
            Standing.initialize(vocab_size, batch_size, records)
            for (vocab_size, batch_size), records in sorted(buckets.items())
        ]

        return cls(runs=ordered, origins=origins, standings=standings)

    @property
    def latest(self) -> RunDirectory:

        return self.runs[-1]

    @property
    def gaps(self) -> list[Standing]:

        return list(filter(attrgetter("is_incomplete"), self.standings))

    @property
    def summary(self) -> DispatchSummary:

        if gaps := self.gaps:

            buckets = list(map(attrgetter("vocab_size", "batch_size"), gaps))

            raise ValueError(f"bucket(s) {buckets} lack a measured case for some strategy")

        else:

            bands = groupby(attrgetter("vocab_size"), self.standings)

            vocab = PriorityBucket(
                VocabBucket.initialize(size=size, batch=map(attrgetter("bucket"), standings))
                for size, standings in bands.items()
            )

            manifest = self.latest.manifest
            stamp = manifest.start.isoformat()

            metadata = DispatchMetadata(version=DISPATCH_VERSION, timestamp=stamp)

            return DispatchSummary(metadata=metadata, k=manifest.sweep.k, vocab=vocab)

    @property
    def near_ties(self) -> list[NearTie]:

        ties = (standing.near_ties(self.dispersion) for standing in self.standings)

        return list(chain.from_iterable(ties))

    @property
    def findings(self) -> list[Finding]:

        winners = chain.from_iterable(map(attrgetter("winners"), self.standings))
        audited = (Finding.initialize(record, self.limits(record)) for record in winners)

        return list(filter(attrgetter("concerns"), audited))

    @property
    def mismatches(self) -> list[str]:

        versions = [run.manifest.versions for run in self.runs]
        packages = frozenset(chain.from_iterable(versions))

        return sorted(filter(is_multi_version(versions), packages))

    @property
    def is_multi_timer(self) -> bool:

        return len({run.manifest.timer for run in self.runs}) > 1

    def origin(self, record: Record) -> RunDirectory:

        return self.runs[self.origins[record.case]]

    def limits(self, record: Record) -> Limits:

        return self.origin(record).manifest.limits

    def dispersion(self, record: Record) -> float:

        return self.origin(record).manifest.dispersion

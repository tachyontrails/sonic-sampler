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

from itertools import chain, cycle, islice, product
from operator import attrgetter
from typing import Annotated, Final, get_type_hints

from cytoolz.itertoolz import groupby, unique
from msgspec import Struct
from tyro.conf import arg

from triton import cdiv     # noqa.

from sonic_sampler.benchmark.base import (
    REGISTER_FILE,
    floor_power_of_2,
    is_power_of_2,
    power_of_2_ladder,
    power_of_2_range,
    register_budget,
)
from sonic_sampler.interface.base import TopKStrategy
from sonic_sampler.interface.dispatch import RuntimeConfig, Strategy, TwoStageWarpConfig
from sonic_sampler.ops.base import MAX_K, next_power_of_2


type Axis = tuple[int, ...]


# The keys addressable by the 16-bit index half of a packed key.
INDEX_LIMIT: Final[int] = 1 << 16

# The bytes per key of the second stage's full-tile shared-memory staging buffer.
STAGING_BYTES: Final[int] = 4

# The registers held per key: stage-1 bitonic sorts and folds its tile in place, while stage-1
# radix / adaptive and stage 2 keep a second full-tile array live alongside the packed keys.
FOLDED_REGISTERS: Final[int] = 1
PAIRED_REGISTERS: Final[int] = 2

# The swept warp counts, shared by both stages.
WARPS: Final[Axis] = (4, 8, 16, 32)

# The default vocabulary and vocab block sizes.
VOCAB_SIZES: Final[Axis] = tuple(power_of_2_range(start=13, end=19))
BLOCK_SIZES: Final[Axis] = tuple(power_of_2_range(start=10, end=16))

# The default batch sizes: powers of 2 up to 512, with half-octave midpoints above 32.
BATCH_SIZES: Final[Axis] = tuple(sorted(power_of_2_ladder(end=10, ridge=5)))

# The swept top-k strategies, keyed by their dispatch name.
STRATEGIES: Final[dict[Strategy, TopKStrategy]] = TopKStrategy.factory()


class Sweep(Struct):

    """
    The benchmark search space over vocabulary, vocab block, and batch sizes.

    """

    # The target top-k value.
    k: Annotated[int, arg(name="top-k", aliases=("-k",))] = MAX_K

    # The vocabulary sizes to sweep.
    vocab_sizes: Annotated[Axis, arg(aliases=("-v",))] = VOCAB_SIZES

    # The vocab block sizes to sweep.
    block_ns: Annotated[Axis, arg(aliases=("-n",))] = BLOCK_SIZES

    # The batch sizes, i.e. row counts, to sweep.
    batch_sizes: Annotated[Axis, arg(aliases=("-b",))] = BATCH_SIZES

    # The cap on padded vocab blocks, defaulting to the register bound for the top-k value.
    max_blocks: Annotated[int | None, arg(aliases=("-m",))] = None

    # Flag to skip warp counts whose tile overflows the register budget.
    warp_floor: bool = True

    # Flag to indicate PDL enablement.
    pdl: bool = True

    # The base seed of the per-bucket logits.
    seed: int = 0

    def __post_init__(self) -> None:

        if not (is_power_of_2(self.k) and self.k <= MAX_K):

            raise ValueError(f"top-k must be a power of 2 within [ 1, {MAX_K} ], got {self.k}")

        elif not (self.vocab_sizes and self.block_ns and self.batch_sizes):

            raise ValueError("vocabulary, vocab block, and batch sizes must each be non-empty")

        elif min(self.vocab_sizes + self.batch_sizes) < 1:

            raise ValueError("vocabulary and batch sizes must be positive")

        elif not all((is_power_of_2(n) and n <= INDEX_LIMIT) for n in self.block_ns):

            raise ValueError(f"vocab block sizes must be powers of 2 within [ 1, {INDEX_LIMIT} ]")

        elif self.max_blocks is not None and self.max_blocks < 2:

            raise ValueError(f"maximum vocab blocks must be at least 2, got {self.max_blocks}")

        else:

            self.sanitize()

    def sanitize(self) -> None:

        for k, v in get_type_hints(self.__class__).items():

            if v is Axis:

                setattr(self, k, tuple(unique(getattr(self, k))))


class Group(Struct, frozen=True):

    """
    A vocabulary tiling, which fixes every specialization of both kernels except the strategy and
    warp counts.

    """

    # The vocabulary size.
    vocab_size: int

    # The vocab block size.
    block_n: int

    @property
    def blocks(self) -> int:

        return cdiv(self.vocab_size, self.block_n)

    @property
    def block_v(self) -> int:

        return next_power_of_2(self.blocks)


class Limits(Struct):

    """
    The bounds applied to a sweep before any kernel is compiled.

    Note(s):

        • The ceiling is exact: the packed key's 16-bit index bounds Ẑ_v • k by 2^16, and the
          second stage stages 4 • Ẑ_v • k bytes against the opt-in shared memory.

        • The per-warp rule is approximate: a tile of T keys at r registers per key must satisfy
          r • T ≤ min(2^16, 255 • 32 • w), pruning only where a spill is likely.

    """

    # The target top-k value.
    k: int

    # The largest padded vocab block count admitted by the index field and shared memory.
    ceiling: int

    # The requested padded vocab block cap.
    requested: int

    # Flag to apply the per-warp register rule.
    warp_floor: bool

    @classmethod
    def resolve(cls, sweep: Sweep, shared_memory: int) -> Limits:

        blocks = INDEX_LIMIT // (k := sweep.k)
        staged = floor_power_of_2(shared_memory // (STAGING_BYTES * k))

        ceiling = min(blocks, staged)
        requested = sweep.max_blocks or REGISTER_FILE // (PAIRED_REGISTERS * k)

        return cls(k=k, ceiling=ceiling, requested=requested, warp_floor=sweep.warp_floor)

    @property
    def max_blocks(self) -> int:

        return min(self.requested, self.ceiling)

    @property
    def is_clamped(self) -> bool:

        return self.requested > self.ceiling

    def admits(self, group: Group) -> bool:

        return group.blocks > 1 and group.block_v <= self.max_blocks

    def warps(self, tile: int, registers: int) -> Axis:

        if self.warp_floor:

            total = registers * tile

            return tuple(w for w in WARPS if total <= register_budget(w))

        else:

            return WARPS

    def first_warps(self, strategy: Strategy, block_n: int) -> Axis:

        registers = FOLDED_REGISTERS if strategy == "bitonic" else PAIRED_REGISTERS

        return self.warps(block_n, registers)

    def second_warps(self, block_v: int) -> Axis:

        return self.warps(block_v * self.k, PAIRED_REGISTERS)


class Case(Struct, frozen=True):

    # The vocabulary size.
    vocab_size: int

    # The vocab block size.
    block_n: int

    # The batch size, i.e. the row count.
    batch_size: int

    # The top-k strategy.
    strategy: Strategy

    # The first-stage warp count.
    first_warps: int

    # The second-stage warp count.
    second_warps: int

    @property
    def group(self) -> Group:

        return Group(vocab_size=self.vocab_size, block_n=self.block_n)

    @property
    def top_k(self) -> TopKStrategy:

        return STRATEGIES[self.strategy]

    @property
    def tuning(self) -> TwoStageWarpConfig:

        return TwoStageWarpConfig(first=self.first_warps, second=self.second_warps)

    def config(self, priority: int) -> RuntimeConfig:

        return RuntimeConfig(
            priority=priority,
            block_n=self.block_n,
            strategy=self.strategy,
            first_warps=self.first_warps,
            second_warps=self.second_warps,
        )


class CompileUnit(Struct):

    """
    The kernels of one vocabulary tiling and strategy, compiled together ahead of timing.

    """

    # The vocabulary tiling the kernels are specialized for.
    group: Group

    # The top-k strategy.
    strategy: Strategy

    # The admitted first-stage warp count(s).
    first_warps: Axis

    # The admitted second-stage warp count(s).
    second_warps: Axis

    @classmethod
    def initialize(cls, group: Group, strategy: Strategy, limits: Limits) -> CompileUnit:

        first = limits.first_warps(strategy, group.block_n)
        second = limits.second_warps(group.block_v)

        return cls(group=group, strategy=strategy, first_warps=first, second_warps=second)

    @property
    def can_launch(self) -> bool:

        return bool(self.first_warps and self.second_warps)

    @property
    def tunings(self) -> list[TwoStageWarpConfig]:
        """
        The fewest warp pairs that launch every admitted kernel of both stages at least once.

        """
        width = max(len(self.first_warps), len(self.second_warps))
        pairs = islice(zip(cycle(self.first_warps), cycle(self.second_warps)), width)

        return [TwoStageWarpConfig(first=first, second=second) for first, second in pairs]

    @property
    def kernels(self) -> int:

        return len(self.first_warps) + len(self.second_warps)

    def cases(self, batch_size: int) -> list[Case]:

        return [
            Case(
                vocab_size=self.group.vocab_size,
                block_n=self.group.block_n,
                batch_size=batch_size,
                strategy=self.strategy,
                first_warps=first,
                second_warps=second,
            )
            for first, second in product(self.first_warps, self.second_warps)
        ]


class Bucket(Struct):

    # The vocabulary size.
    vocab_size: int

    # The batch size, i.e. the row count.
    batch_size: int

    # The case(s), ordered by vocab block size, strategy, then warp counts.
    cases: list[Case]

    @classmethod
    def initialize(cls, vocab_size: int, batch_size: int, units: list[CompileUnit]) -> Bucket:

        cases = list(chain.from_iterable((u.cases(batch_size) for u in units)))

        return cls(vocab_size=vocab_size, batch_size=batch_size, cases=cases)

    @property
    def strategies(self) -> set[Strategy]:

        return set(map(attrgetter("strategy"), self.cases))

    @property
    def is_imbalanced(self) -> bool:

        return self.strategies != STRATEGIES.keys()

    @property
    def size(self) -> int:

        return len(self.cases)

    @property
    def probes(self) -> list[Case]:
        """
        Each strategy's median case, i.e. a mid-range vocab block size and warp counts.

        """
        lanes = groupby(attrgetter("strategy"), self.cases)

        return [cases[len(cases) // 2] for cases in lanes.values()]


class Plan(Struct):

    """
    The enumerated work of a sweep under its limits: compile units, then timed buckets.

    """

    # The vocabulary tiling(s) admitted by the limits.
    groups: list[Group]

    # The compile unit(s), one per launchable tiling and strategy.
    units: list[CompileUnit]

    # The timed bucket(s), one per vocabulary and batch size.
    buckets: list[Bucket]

    def __post_init__(self) -> None:

        if gaps := list(filter(attrgetter("is_imbalanced"), self.buckets)):

            vocab_sizes = sorted({bucket.vocab_size for bucket in gaps})

            raise ValueError(f"vocab size(s) {vocab_sizes} admit no case for some strategy")

    @classmethod
    def initialize(cls, sweep: Sweep, limits: Limits) -> Plan:

        tilings = (Group(v, n) for v, n in product(sweep.vocab_sizes, sweep.block_ns))
        groups = list(filter(limits.admits, tilings))

        candidates = (
            CompileUnit.initialize(group, strategy, limits)
            for group, strategy in product(groups, STRATEGIES)
        )

        units = list(filter(attrgetter("can_launch"), candidates))
        bands = groupby(attrgetter("group.vocab_size"), units)

        buckets = [
            Bucket.initialize(vocab, batch, bands.get(vocab) or [])
            for vocab, batch in product(sweep.vocab_sizes, sweep.batch_sizes)
        ]

        return cls(groups=groups, units=units, buckets=buckets)

    @property
    def kernels(self) -> int:

        return sum(map(attrgetter("kernels"), self.units))

    @property
    def size(self) -> int:

        return sum(map(attrgetter("size"), self.buckets))

    @property
    def probes(self) -> list[Case]:
        """
        The calibration case(s): each strategy's median case in the lightest and heaviest bucket.

        """
        key = attrgetter("vocab_size", "batch_size")

        smallest = min(self.buckets, key=key).probes
        largest = max(self.buckets, key=key).probes

        return list(unique(chain(smallest, largest)))

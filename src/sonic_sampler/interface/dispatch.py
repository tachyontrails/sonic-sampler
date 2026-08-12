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

from bisect import bisect_left
from importlib import resources
from operator import attrgetter
from typing import Literal, Type, Any, get_args, get_origin, override

from msgspec import Struct, toml, field, convert

from sonic_sampler.ops.base import MAX_K, next_power_of_2


Strategy = Literal["bitonic", "adaptive", "radix"]


class TwoStageWarpConfig(Struct):

    first: int
    second: int

    @classmethod
    def default(cls) -> TwoStageWarpConfig:

        return TwoStageWarpConfig(first=16, second=8)


class ThreeStageWarpConfig(TwoStageWarpConfig):

    # The speculative verification warp configuration.
    third: int

    @property
    def two_stage(self) -> TwoStageWarpConfig:

        return TwoStageWarpConfig(first=self.first, second=self.second)

    @classmethod
    def from_config(
        cls, config: TwoStageWarpConfig, third: int | None = None, gamma: int = 3
    ) -> ThreeStageWarpConfig:

        return ThreeStageWarpConfig(
            first=config.first,
            second=config.second,
            third=third or next_power_of_2(gamma + 1),
        )

    @override
    @classmethod
    def default(cls, gamma: int = 3) -> ThreeStageWarpConfig:

        return cls.from_config(config=TwoStageWarpConfig.default(), gamma=gamma)


class RuntimeConfig(Struct):

    # The config priority and vocab block size.
    priority: int
    block_n: int

    # The top-k strategy.
    strategy: Strategy

    # The tuned warp configuration(s).
    first_warps: int
    second_warps: int

    @property
    def warp_config(self) -> TwoStageWarpConfig:

        return TwoStageWarpConfig(first=self.first_warps, second=self.second_warps)


class SizedBucket(Struct):

    # The target size representing this bucket.
    size: int


def relaxed_upper(index: int, values: list[int], margin: float, target: int) -> bool:

    return values[index] * (1 + margin) > target


class PriorityBucket[T: SizedBucket](list[T]):

    def get(self, size: int, margin: float = 0.5) -> T | None:

        index = bisect_left(self, size, key=attrgetter("size"))

        if index < len(self):

            return self[index]

        elif self and size < self[-1].size * (1 + margin):

            return self[-1]


class BatchBucket(SizedBucket):

    # The priority-ordered list of dispatch configuration(s).
    config: list[RuntimeConfig]

    # The strategy-based configuration mapping.
    mapping: dict[Strategy, RuntimeConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:

        self.mapping = {config.strategy: config for config in self.config}

    @property
    def best(self) -> RuntimeConfig:

        return self.config[0]

    def get(self, strategy: Strategy) -> RuntimeConfig:

        return self.mapping[strategy]


class VocabBucket(SizedBucket):

    # The ordered hierarchical batch-wise dispatch configuration(s).
    batch: PriorityBucket[BatchBucket]

    @property
    def block_n(self) -> int:

        return min(c.block_n for b in self.batch for c in b.config)


class DispatchMetadata(Struct):

    # The underlying format versioning.
    version: int

    # The benchmark timestamp.
    timestamp: str


def decode_hook(target: Type, source: Any) -> Any:

    if get_origin(target) is PriorityBucket:

        if not isinstance(source, list):

            raise TypeError(f"Expected a list for {target!r}, got {type(source).__name__}")

        elif len(args := get_args(target)) != 1:

            raise TypeError(f"Expected a parameterized PriorityBucket, got {target!r}")

        else:

            items = convert(
                source,
                type=list[args[0]],         # noqa.
                dec_hook=decode_hook,
                str_keys=True,
            )

            return PriorityBucket(items)

    else:

        raise NotImplementedError(f"Objects of type {target!r} are not supported")


class DispatchSummary(Struct):

    # The encapsulated benchmark metadata.
    metadata: DispatchMetadata

    # The target top-k value.
    k: int

    # The ordered hierarchical vocab-wise dispatch configuration(s).
    vocab: PriorityBucket[VocabBucket]

    @classmethod
    def default(cls) -> DispatchSummary | None:

        return cls.load(k=MAX_K, arch=100)

    @classmethod
    def load(cls, k: int = MAX_K, arch: int | None = None) -> DispatchSummary | None:

        if (cc := arch) is None:

            # Retrieve current device compute capability assuming active device context.

            major, minor = torch.cuda.get_device_capability()
            cc = (major * 10) + minor

        data_path = (
            resources.files("sonic_sampler")
            / "resources"
            / f"sm{cc}_k{k}.toml"
        )

        if data_path.is_file():

            return toml.decode(data_path.read_bytes(), type=cls, dec_hook=decode_hook)

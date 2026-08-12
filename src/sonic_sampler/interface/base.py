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

from msgspec import Struct

from torch import Tensor

from triton import cdiv     # noqa.

from sonic_sampler.interface.dispatch import (
    DispatchSummary,
    RuntimeConfig,
    Strategy,
    TwoStageWarpConfig,
    VocabBucket,
)
from sonic_sampler.ops.base import MAX_K


FP32: torch.dtype = torch.float32
BF16: torch.dtype = torch.bfloat16

I32: torch.dtype = torch.int32
U32: torch.dtype = torch.uint32


class Vocabulary(Struct):

    # The total tokens / columns of the logit(s).
    size: int

    # The total vocab block(s) / tile(s).
    blocks: int

    @classmethod
    def resolve(cls, size: int, block_n: int) -> Vocabulary:

        return cls(size=size, blocks=cdiv(size, block_n))

    def evolve(self, block_n: int) -> Vocabulary:

        return Vocabulary(size=self.size, blocks=cdiv(self.size, block_n))


class TopKStrategy(Struct):

    radix: bool = False
    sparse: bool = False

    scale: int = 4

    @classmethod
    def adaptive(cls, scale: int = 4) -> TopKStrategy:

        return cls(radix=True, sparse=True, scale=scale)

    @classmethod
    def resolve(cls, config: RuntimeConfig) -> TopKStrategy:

        match config.strategy:

            case "bitonic":

                return cls()

            case "radix":

                return cls(radix=True)

            case "adaptive":

                return cls.adaptive()

    @property
    def key(self) -> Strategy:

        if self.radix:

            return "adaptive" if self.sparse else "radix"

        else:

            return "bitonic"

    @property
    def is_adaptive(self) -> bool:

        return self.radix and self.sparse

    @property
    def is_alternating(self) -> bool:

        return not self.radix

    @property
    def inverse(self) -> float | None:

        if self.is_adaptive:

            return 1 / self.scale


def resolve_dispatch(
    k: int,
    vocab_size: int,
    block_size: int | None,
    arch: int | None,
) -> tuple[VocabBucket | None, int]:

    if k > MAX_K:

        raise ValueError(f"maximum supported top-k value is {MAX_K}, got {k} instead")

    elif int.bit_count(k) > 1:

        raise ValueError(f"top-k value must be a power of 2, got {k} instead")

    elif block_size is None:

        if (dispatch := DispatchSummary.load(k=k, arch=arch)) is None:

            raise ValueError("unable to resolve dispatch summary for given top-k and architecture")

        elif (bucket := dispatch.vocab.get(size=vocab_size)) is None:

            raise ValueError("unable to match tuning configuration for given vocabulary size")

        else:

            return bucket, bucket.block_n

    else:

        return None, block_size


class TwoStageTiling(Struct):

    # Flag to indicate PDL enablement.
    pdl: bool

    # The tile-size per vocab block.
    block_n: int

    # The (local) vocabulary specific(s).
    vocabulary: Vocabulary

    # The underlying uint32 bitpacked buffer.
    scratchpad: Tensor

    # The tuned configuration dispatch.
    dispatch: VocabBucket | None

    @classmethod
    def resolve(
        cls,
        vocab_size: int,
        k: int = MAX_K,
        block_size: int | None = None,
        arch: int | None = None,
        ensure_multiblock: bool = True,
    ) -> tuple[VocabBucket | None, Vocabulary, int]:

        dispatch, block_n = resolve_dispatch(k, vocab_size, block_size, arch)
        vocabulary = Vocabulary.resolve(vocab_size, block_n)

        if ensure_multiblock and (blocks := vocabulary.blocks) <= 1:

            raise ValueError(f"total vocab blocks must be greater than 1, got {blocks} instead")

        else:

            return dispatch, vocabulary, block_n

    @classmethod
    def factory(
        cls,
        vocab_size: int,
        batch_size: int = 32,
        lookahead: int = 0,
        k: int = MAX_K,
        block_size: int | None = None,
        arch: int | None = None,
        enable_pdl: bool = True,
        ensure_multiblock: bool = True,
        unpacked_buffers: bool = True,
        values_dtype: torch.dtype = FP32,
    ) -> tuple[TwoStageTiling, Tensor | None, Tensor | None]:

        dispatch, vocabulary, block_n = cls.resolve(
            vocab_size=vocab_size,
            k=k,
            block_size=block_size,
            arch=arch,
            ensure_multiblock=ensure_multiblock,
        )

        total_rows = batch_size * (lookahead + 1)
        scratchpad = torch.empty(total_rows, vocabulary.blocks * k, dtype=U32)

        values = indices = None

        if unpacked_buffers:

            values = torch.empty(total_rows, k, dtype=values_dtype)
            indices = torch.empty(total_rows, k, dtype=I32)

        instance = TwoStageTiling(
            pdl=enable_pdl,
            block_n=block_n,
            vocabulary=vocabulary,
            scratchpad=scratchpad,
            dispatch=dispatch,
        )

        return instance, values, indices

    def tuning(
        self,
        size: int,
        strategy: TopKStrategy | None = None,
        tuning: TwoStageWarpConfig | None = None,
    ) -> tuple[TwoStageWarpConfig, TopKStrategy, int]:

        block_size = self.block_n
        k_strategy = strategy

        if (config := tuning) is None:

            # Resolve tuning configuration for given batch size if dispatch bucket is available.

            if self.dispatch and (dispatch := self.dispatch.batch.get(size=size)):

                if strategy:

                    config = (target := dispatch.get(strategy.key)).warp_config
                    block_size = target.block_n

                else:

                    config = (best := dispatch.best).warp_config

                    k_strategy = TopKStrategy.resolve(config=best)
                    block_size = best.block_n

        # Ensure defaults for warp configuration and top-k strategy.

        config = config or TwoStageWarpConfig.default()
        k_strategy = k_strategy or TopKStrategy()

        return config, k_strategy, block_size

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

from typing import Self, Tuple

from torch import Tensor

from sonic_sampler.interface.base import TwoStageTiling, TopKStrategy
from sonic_sampler.interface.dispatch import TwoStageWarpConfig
from sonic_sampler.interface.functional.base import collapse_2d
from sonic_sampler.ops.base import next_power_of_2, MAX_K
from sonic_sampler.ops.topk.bitpack import bitpacked_reduction_kernel
from sonic_sampler.ops.topk.unpack import merge_unpack_kernel


BF16: torch.dtype = torch.bfloat16

I32: torch.dtype = torch.int32
U32: torch.dtype = torch.uint32


class UnshardedBoundedTopK(TwoStageTiling):

    # The target top-k values to select.
    k: int

    # The top-k values buffer.
    values: Tensor

    # The top-k indices buffer.
    indices: Tensor

    @classmethod
    def initialize(
        cls,
        vocab_size: int,
        block_size: int | None = None,
        batch_size: int = 32,
        k: int = MAX_K,
        pdl: bool = True,
        arch: int | None = None,
    ) -> UnshardedBoundedTopK:

        # TODO: Support top-k over single vocab blocks in a single-stage.

        factory, values, indices = cls.factory(
            vocab_size=vocab_size,
            batch_size=batch_size,
            k=MAX_K,
            block_size=block_size,
            arch=arch,
            enable_pdl=pdl,
            ensure_multiblock=True,
            unpacked_buffers=True,
            values_dtype=BF16,
        )

        return cls(
            k=k,
            pdl=factory.pdl,
            block_n=factory.block_n,
            vocabulary=factory.vocabulary,
            scratchpad=factory.scratchpad,
            values=values,
            indices=indices,
            dispatch=factory.dispatch,
        )

    def clear(self) -> Self:
        """
        Helper method to clear out the internal buffer(s) between consecutive correctness check(s).

        """
        self.scratchpad.zero_()

        self.values.fill_(-float('inf'))
        self.indices.fill_(-1)

        return self

    def __call__(
        self,
        logits: Tensor,
        *,
        strategy: TopKStrategy | None = None,
        tuning: TwoStageWarpConfig | None = None,
    ) -> Tuple[Tensor, Tensor]:

        *batch_step, _ = logits.shape
        n_rows = collapse_2d(logits).size(0)

        config, k_strategy, block_n = self.tuning(n_rows, strategy, tuning)
        vocabulary = self.vocabulary.evolve(block_n=block_n)

        grid = (n_rows, vocab_blocks := vocabulary.blocks)

        stride_s = self.scratchpad.stride(0)
        block_v = next_power_of_2(vocab_blocks)

        total_cols = block_v * self.k
        vocab_size = vocabulary.size

        bitpacked_reduction_kernel[grid](
            logits,
            self.scratchpad,
            vocab_size,
            self.k,
            k_strategy.scale,
            k_strategy.is_adaptive,
            k_strategy.radix,
            self.pdl,
            logits.stride(0),
            stride_s,
            block_n,
            num_warps=config.first,
        )

        merge_unpack_kernel[(n_rows,)](
            self.scratchpad,
            self.values,
            self.indices,
            vocab_blocks,
            None,
            None,
            self.k,
            k_strategy.inverse,
            None,
            total_cols,
            False,
            k_strategy.is_adaptive,
            k_strategy.is_alternating,
            self.pdl,
            stride_s,
            None,
            None,
            block_n,
            block_v,
            num_warps=config.second,
        )

        values = self.values[:n_rows].unflatten(0, batch_step)
        indices = self.indices[:n_rows].unflatten(0, batch_step)

        return values, indices

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

from typing import override

from torch import Tensor

from triton import cdiv     # noqa.

from sonic_sampler.base.sampler import Verification
from sonic_sampler.core.buffer import SamplingBuffers
from sonic_sampler.ops.base import MAX_K
from sonic_sampler.interface.base import TwoStageTiling, TopKStrategy
from sonic_sampler.interface.dispatch import ThreeStageWarpConfig
from sonic_sampler.interface.functional.base import collapse_2d
from sonic_sampler.interface.functional.multistep import fused_multistep


BF16: torch.dtype = torch.bfloat16
FP32: torch.dtype = torch.float32

I32: torch.dtype = torch.int32
U32: torch.dtype = torch.uint32


class UnshardedFusedMultistep(TwoStageTiling):

    # The maximum speculation lookahead, γ.
    lookahead: int

    # The fp32 top-k values buffer with shape [ B • (γ + 1), M ].
    values: Tensor

    # The int32 top-k indices buffer with shape [ B • (γ + 1), M ].
    indices: Tensor

    @classmethod
    def initialize(
        cls,
        vocab_size: int,
        lookahead: int,
        batch_size: int = 32,
        pdl: bool = True,
        block_size: int | None = None,
        arch: int | None = None,
    ) -> UnshardedFusedMultistep:

        factory, values, indices = cls.factory(
            vocab_size=vocab_size,
            batch_size=batch_size,
            lookahead=lookahead,
            k=MAX_K,
            block_size=block_size,
            arch=arch,
            enable_pdl=pdl,
            ensure_multiblock=True,
            unpacked_buffers=True,
        )

        return cls(
            pdl=factory.pdl,
            block_n=factory.block_n,
            lookahead=lookahead,
            vocabulary=factory.vocabulary,
            scratchpad=factory.scratchpad,
            values=values,
            indices=indices,
            dispatch=factory.dispatch,
        )

    @override
    def tuning(
        self,
        size: int,
        strategy: TopKStrategy | None = None,
        tuning: ThreeStageWarpConfig | None = None,
    ) -> tuple[ThreeStageWarpConfig, TopKStrategy, int]:

        config, k_strategy, block_n = super().tuning(size, strategy, tuning and tuning.two_stage)

        config = ThreeStageWarpConfig.from_config(
            config=config,
            third=tuning and tuning.third,
            gamma=self.lookahead,
        )

        return config, k_strategy, block_n

    def __call__(
        self,
        logits: Tensor,
        tokens: Tensor,
        buffers: SamplingBuffers,
        probabilities: Tensor | None = None,
        slot_mapping: Tensor | None = None,
        update_counts: bool = False,
        logprobs: bool = False,
        in_place: bool = False,
        strategy: TopKStrategy | None = None,
        tuning: ThreeStageWarpConfig | None = None,
    ) -> Verification:

        counts = (penalties := buffers.repetition).counts
        config, k_strategy, block_n = self.tuning(collapse_2d(logits).size(0), strategy, tuning)

        return fused_multistep(
            logits=logits,
            indicators=buffers.flags.target,
            drafted_tokens=tokens,
            lookahead=self.lookahead,
            enable_pdl=self.pdl,
            update_counts=update_counts,
            return_logprobs=logprobs,
            block_n=self.block_n,
            scratchpad=self.scratchpad,
            values=self.values,
            indices=self.indices,
            draft_probabilities=probabilities,
            slot_mapping=slot_mapping,
            grammar=buffers.grammar,
            context_counts=counts.context,
            decode_counts=counts.decode,
            repetition_penalties=penalties.multiplicative,
            frequency_penalties=penalties.frequency,
            presence_penalties=penalties.presence,
            logit_bias=buffers.bias,
            temperature=buffers.temperature,
            top_k=buffers.top_k,
            top_p=buffers.top_p,
            min_p=buffers.min_p,
            top_k_logprobs=buffers.top_logprobs,
            gumbel_noise=(weights := buffers.weights).gumbel.target,
            uniform_noise=weights.uniform,
            output_tokens=tokens if in_place else None,
            topk_strategy=k_strategy,
            warp_config=config,
        )

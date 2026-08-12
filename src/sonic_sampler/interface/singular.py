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

from typing import Tuple

import torch

from torch import Tensor

from triton import cdiv     # noqa.

from sonic_sampler.base.sampler import Selection
from sonic_sampler.core.buffer import SamplingBuffers
from sonic_sampler.interface.dispatch import TwoStageWarpConfig
from sonic_sampler.interface.base import TwoStageTiling, TopKStrategy
from sonic_sampler.interface.functional.singular import fused_singular
from sonic_sampler.ops.base import MAX_K


BF16: torch.dtype = torch.bfloat16

I32: torch.dtype = torch.int32
U32: torch.dtype = torch.uint32


def resolve_tokens[T: Tensor | None](tokens: T, timestep: int, in_place: bool) -> Tuple[T, T]:

    output = drafted = None

    if tokens is not None:

        if in_place:

            output = tokens[:, timestep, None]

        if timestep:

            drafted = tokens

    return output, drafted


class UnshardedFusedSingular(TwoStageTiling):

    # The speculation lookahead, γ.
    lookahead: int

    @classmethod
    def initialize(
        cls,
        vocab_size: int,
        batch_size: int = 32,
        lookahead: int = 0,
        pdl: bool = True,
        block_size: int | None = None,
        arch: int | None = None,
    ) -> UnshardedFusedSingular:

        factory, _, _ = cls.factory(
            vocab_size=vocab_size,
            batch_size=batch_size,
            lookahead=lookahead,
            k=MAX_K,
            block_size=block_size,
            arch=arch,
            enable_pdl=pdl,
            ensure_multiblock=True,
            unpacked_buffers=False,
        )

        return cls(
            pdl=factory.pdl,
            block_n=factory.block_n,
            lookahead=lookahead,
            vocabulary=factory.vocabulary,
            scratchpad=factory.scratchpad,
            dispatch=factory.dispatch,
        )

    def autoregressive(
        self,
        logits: Tensor,
        buffers: SamplingBuffers,
        prefill: bool = False,
        update_counts: bool = False,
        logprobs: bool = False,
        slot_mapping: Tensor | None = None,
        strategy: TopKStrategy | None = None,
        tuning: TwoStageWarpConfig | None = None,
    ) -> Selection:

        counts = (penalties := buffers.repetition).counts
        config, k_strategy, block_n = self.tuning(logits.size(0), strategy, tuning)

        return fused_singular(
            logits=logits,
            indicators=buffers.flags.target,
            enable_pdl=self.pdl,
            is_prefill=prefill,
            update_counts=update_counts,
            return_logprobs=logprobs,
            block_n=block_n,
            scratchpad=self.scratchpad,
            slot_mapping=slot_mapping,
            grammar=buffers.grammar[:, 0],
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
            gumbel_noise=buffers.weights.gumbel.target[:, 0],
            topk_strategy=k_strategy,
            warp_config=config,
        )

    def drafting(
        self,
        logits: Tensor,
        buffers: SamplingBuffers,
        timestep: int = 0,
        tokens: Tensor | None = None,
        probabilities: Tensor | None = None,
        slot_mapping: Tensor | None = None,
        d2t_mapping: Tensor | None = None,
        target_vocab: int | None = None,
        in_place: bool = False,
        strategy: TopKStrategy | None = None,
        tuning: TwoStageWarpConfig | None = None,
    ) -> Selection:

        counts = (penalties := buffers.repetition).counts
        config, k_strategy, block_n = self.tuning(logits.size(0), strategy, tuning)

        output_tokens, drafted_tokens = resolve_tokens(tokens, timestep, in_place)
        return_probabilities = not buffers.flags.all_greedy

        return fused_singular(
            logits=logits,
            indicators=buffers.flags.draft,
            lookahead=self.lookahead,
            target_vocab=target_vocab,
            enable_pdl=self.pdl,
            return_probabilities=return_probabilities,
            block_n=block_n,
            scratchpad=self.scratchpad,
            slot_mapping=slot_mapping,
            d2t_mapping=d2t_mapping,
            grammar=buffers.grammar[:, timestep],
            context_counts=counts.context,
            decode_counts=counts.decode,
            drafted_tokens=drafted_tokens,
            repetition_penalties=penalties.multiplicative,
            frequency_penalties=penalties.frequency,
            presence_penalties=penalties.presence,
            logit_bias=buffers.bias,
            temperature=buffers.temperature,
            top_k=buffers.top_k,
            top_p=buffers.top_p,
            min_p=buffers.min_p,
            top_k_logprobs=buffers.top_logprobs,
            gumbel_noise=buffers.weights.gumbel.draft[:, timestep],
            output_tokens=output_tokens,
            draft_probabilities=probabilities,
            topk_strategy=k_strategy,
            warp_config=config,
        )

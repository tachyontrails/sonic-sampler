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


MAX_TOP_LP: int = 5

I32: torch.dtype = torch.int32
BF16: torch.dtype = torch.bfloat16


class TopTargets(Struct):

    # The associated top-k token(s) with shape [ B, T ] or [ B, γ + 1, T ].
    tokens: Tensor

    # The corresponding token log-probabilities with shape [ B, T ] or [ B, γ + 1, T ].
    logprobs: Tensor

    @classmethod
    def empty_blocked(cls, batch_size: int, timesteps: int = 1) -> TopTargets:

        shape = (batch_size, timesteps, MAX_TOP_LP) if timesteps > 1 else (batch_size, MAX_TOP_LP)

        tokens = torch.empty(shape, dtype=I32)
        logprobs = torch.empty(shape, dtype=BF16)

        return cls(tokens=tokens, logprobs=logprobs)

    @classmethod
    def empty_varlen(cls, num_tokens: int) -> TopTargets:

        tokens = torch.empty(num_tokens, MAX_TOP_LP, dtype=I32)
        logprobs = torch.empty(num_tokens, MAX_TOP_LP, dtype=BF16)

        return cls(tokens=tokens, logprobs=logprobs)


class LogProbabilities(Struct):

    # The selected token log-probabilities with shape [ B, 1 ] or [ B, γ + 1 ].
    selected: Tensor

    # The top-k token and log-probabilities with shape [ B, T ] or [ B, γ + 1, T ].
    top_k: TopTargets | None = None

    @classmethod
    def empty_blocked(
        cls, batch_size: int, timesteps: int = 1, top_k: bool = False
    ) -> LogProbabilities:

        selected = torch.empty(batch_size, timesteps, dtype=BF16)
        targets = TopTargets.empty_blocked(batch_size, timesteps) if top_k else None

        return cls(selected=selected, top_k=targets)

    @classmethod
    def empty_varlen(cls, num_tokens: int, top_k: bool = False) -> LogProbabilities:

        selected = torch.empty(num_tokens, dtype=BF16)
        targets = TopTargets.empty_varlen(num_tokens) if top_k else None

        return cls(selected=selected, top_k=targets)


class Selection(Struct):

    # The sampled / selected next tokens with shape [ B, 1 ].
    tokens: Tensor

    # The finalized probability distribution with shape [ B, V_t ].
    probabilities: Tensor | None = None

    # The corresponding token log-probabilities.
    logprobs: LogProbabilities | None = None

    @classmethod
    def from_buffers(
        cls, tokens: Tensor, probabilities: Tensor | None = None, logprobs: bool = False,
    ) -> Selection:

        logprobs_buffer = None

        if logprobs:

            with torch.device(tokens.device):

                logprobs_buffer = LogProbabilities.empty_blocked(tokens.size(0), top_k=True)

        return cls(tokens=tokens, probabilities=probabilities, logprobs=logprobs_buffer)

    @classmethod
    def empty(
        cls,
        batch_size: int,
        vocab_size: int,
        device: torch.device,
        probs: bool = False,
        logprobs: bool = False,
        top_logprobs: bool = False,
    ) -> Selection:

        probs_buffer = logprobs_buffer = None

        with torch.device(device):

            tokens = torch.empty((batch_size, 1), dtype=I32)

            if probs:

                probs_buffer = torch.empty((batch_size, vocab_size), dtype=BF16)

            if logprobs:

                logprobs_buffer = LogProbabilities.empty_blocked(batch_size, top_k=top_logprobs)

            return cls(tokens=tokens, probabilities=probs_buffer, logprobs=logprobs_buffer)


class Verification(Struct):

    # The drafted and selected next tokens with shape [ B, γ + 1 ].
    tokens: Tensor

    # The total accepted tokens i.e. rejection offsets with shape [ B, 1 ].
    offsets: Tensor

    # The associated selected token log-probabilities with shape [ B, γ + 1 ].
    logprobs: LogProbabilities | None = None

    @classmethod
    def from_tokens(
        cls,
        tokens: Tensor,
        batch_size: int,
        timesteps: int,
        logprobs: bool = False,
        varlen: bool = False,
    ) -> Verification:

        logprobs_buffer = None

        with torch.device(tokens.device):

            offsets = tokens.new_empty(batch_size, 1, dtype=I32)

            if logprobs:

                if varlen:

                    logprobs_buffer = LogProbabilities.empty_varlen(
                        num_tokens=tokens.size(0),
                        top_k=True,
                    )

                else:

                    logprobs_buffer = LogProbabilities.empty_blocked(
                        batch_size=batch_size,
                        timesteps=timesteps,
                        top_k=True,
                    )

            return cls(tokens=tokens, offsets=offsets, logprobs=logprobs_buffer)

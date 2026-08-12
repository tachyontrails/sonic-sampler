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

from math import inf
from typing import Tuple

from cytoolz.functoolz import thread_first
from msgspec import Struct

from torch import Tensor

from sonic_sampler.base import (
    Selection,
    Verification,
    LogProbabilities,
    TopTargets,
    MAX_TOP_LP,
)
from sonic_sampler.core import Indicator, SamplingBuffers
from sonic_sampler.ops import MAX_K


ValueIndices = Tuple[Tensor, Tensor]
LogProbsPair = Tuple[Tensor, Tensor]


F32: torch.dtype = torch.float32
I32: torch.dtype = torch.int32

BF16: torch.dtype = torch.bfloat16
BOOL: torch.dtype = torch.bool

ONE: int = 0x1


def softmax(values: Tensor) -> LogProbsPair:

    logprobs = torch.log_softmax(values, dim=-1, dtype=F32)
    probabilities = logprobs.exp()

    return probabilities, logprobs.to(dtype=BF16)


def subset_select(
    values: Tensor, mapping: Tensor | None, *, dim: int = -1
) -> Tensor:

    return values if mapping is None else values.index_select(dim=dim, index=mapping)


def remap(indices: Tensor, mapping: Tensor | None) -> Tensor:

    if mapping is None:

        return indices

    else:

        return (
            mapping.gather(0, indices.flatten())
            .reshape_as(indices)
        )


class NaiveSampler(Struct):

    """
    Implements all logit processing, sampling, and verification methods in pure torch-based
    routines via a compositional structure around a given instance of `SamplingBuffers`.

    N.B. While primarily used for verifying correctness of fused kernels, CUDA Graph benchmarks
    are supported under a static configuration of indicator(s).

    Example Usage:

        >>> buffers = SamplingBuffers(size=4, timesteps=5, vocab_size=129_280)
        >>> sampler = NaiveSampler(buffers=buffers)

        >>> logits = torch.randn(4, 129_280, dtype=torch.bfloat16, device=buffers.device)
        >>> selection = sampler.singular(logits=logits, timestep=0, probs=True)

    """

    # The encapsulated buffer(s) for selection and verification.
    buffers: SamplingBuffers

    # Flag enabling bounded top-k.
    bounded: bool = True

    def is_enabled(self, flag: Indicator) -> bool:

        return any((flag & value) for value in self.buffers.indicators.target)

    @property
    def is_greedy(self) -> bool:

        return self.buffers.flags.all_greedy

    @property
    def has_ordering(self) -> bool:

        ordering = Indicator.cumulative()

        return any((value & ordering) for value in self.buffers.indicators.target)

    def grammar(
        self,
        logits: Tensor,
        bitmask: Tensor,
        mapping: Tensor | None = None,
    ) -> Tensor:

        if self.is_enabled(Indicator.GRAMMAR):

            shifts = (
                torch.arange(32, dtype=I32, device=logits.device)
                .__getitem__((None,) * logits.dim())
            )

            expanded = (
                bitmask.view(I32)[..., None]
                .bitwise_right_shift(shifts)
                .flatten(-2)
            )

            remapped = subset_select(expanded, mapping)

            finalized = (
                remapped.bitwise_and_(ONE)
                .bitwise_xor_(ONE)
                .bool()
            )

            logits.masked_fill_(finalized, -inf)

        return logits

    def repetition(
        self,
        logits: Tensor,
        timestep: int | None = None,
        tokens: Tensor | None = None,
        mapping: Tensor | None = None,
    ) -> Tensor:

        if self.is_enabled(Indicator.repetition()):

            context = (counts := self.buffers.repetition.counts).context
            decode = counts.decode

            original = decode[..., 0]
            new_dims = (None,) * (logits.dim() - 1)

            if tokens is not None:

                size, steps = tokens.shape

                values = (
                    tokens[:, None]
                    .expand(-1, steps, steps)
                    .tril(diagonal=-1)
                )

                if timestep is None:

                    decode = (
                        decode[:, None]
                        .expand(size, steps, -1)
                        .scatter_add(-1, values, torch.ones_like(values))
                    )

                    context = context[:, None]

                else:

                    decode = decode.scatter_add(-1, values[:, timestep], torch.ones_like(tokens))

                decode[..., 0].copy_(original)

            decode = subset_select(decode, mapping)

            if self.is_enabled(Indicator.MULTIPLICATIVE):

                total = subset_select(context, mapping) + decode

                scales = (
                    self.buffers.repetition.multiplicative
                    .__getitem__((slice(None), *new_dims))
                    .expand_as(logits)
                    .masked_fill(total <= 0, 1.0)
                )

                logits = torch.where(logits < 0, logits * scales, logits / scales)

            if self.is_enabled(Indicator.FREQUENCY):

                scales = (
                    self.buffers.repetition.frequency
                    .__getitem__((slice(None), *new_dims))
                )

                logits.addcmul_(decode, scales, value=-1)

            if self.is_enabled(Indicator.PRESENCE):

                scales = (
                    self.buffers.repetition.presence
                    .__getitem__((slice(None), *new_dims))
                )

                logits.addcmul_(decode > 0, scales, value=-1)

        return logits

    def bias(self, logits: Tensor, mapping: Tensor | None = None) -> Tensor:

        if self.is_enabled(Indicator.BIAS):

            new_dims = (None,) * (logits.dim() - 2)

            logit_bias = (
                subset_select(values=self.buffers.bias, mapping=mapping)
                .__getitem__((slice(None), *new_dims))
            )

            logits.add_(logit_bias)

        return logits

    def temperature(self, logits: Tensor) -> Tensor:

        if self.is_enabled(Indicator.TEMPERATURE):

            new_dims = (None,) * (logits.dim() - 1)

            scale = (
                self.buffers.temperature
                .__getitem__((slice(None), *new_dims))
            )

            logits.div_(scale)

        return logits

    def top_k(self, values: Tensor) -> Tensor:

        if self.is_enabled(Indicator.TOP_K):

            new_dims = (None,) * (values.dim() - 2)

            batch_size = values.size(0)
            vocab_size = values.size(-1)

            mask = (
                torch.arange(vocab_size, dtype=I32, device=values.device)[None]
                .expand(batch_size, -1)
                .ge(self.buffers.top_k[:, None])
                .__getitem__((slice(None), *new_dims))
            )

            values.masked_fill_(mask, -inf)

        return values

    def top_p(self, values: Tensor) -> Tensor:

        if self.is_enabled(Indicator.TOP_P):

            new_dims = (None,) * (values.dim() - 1)

            upper = (
                self.buffers.top_p
                .__getitem__((slice(None), *new_dims))
            )

            mask = (
                values.softmax(dim=-1, dtype=F32)
                .cumsum_(dim=-1)
                .gt(upper)
            )

            mask[..., 0].zero_()

            values.masked_fill_(mask, -inf)

        return values

    def min_p(self, values: Tensor) -> Tensor:

        if self.is_enabled(Indicator.MIN_P):

            scores = values.softmax(dim=-1)
            new_dims = (None,) * (values.dim() - 1)

            lower = (
                self.buffers.min_p
                .__getitem__((slice(None), *new_dims))
                .mul(scores[..., :1])
            )

            values.masked_fill_(scores < lower, -inf)

        return values

    def cumulative(self, logits: Tensor) -> ValueIndices | None:

        if not self.is_greedy:

            if self.bounded:

                values, indices = logits.topk(k=MAX_K, dim=-1)

                values = thread_first(values, self.top_k, self.top_p, self.min_p)

                return values, indices.to(dtype=I32)

            elif self.has_ordering:

                values, indices = logits.sort(dim=-1, stable=True, descending=True)

                values = thread_first(values, self.top_k, self.top_p, self.min_p)

                return values, indices.to(dtype=I32)

    def probabilities(
        self,
        values: Tensor,
        indices: Tensor | None = None,
        mapping: Tensor | None = None,
    ) -> Tensor:

        if indices is None:

            if mapping is None:

                return values

            else:

                return (
                    torch.zeros_like(self.buffers.weights.gumbel.target[:, 0])
                    .index_copy_(-1, mapping, values)
                )

        else:

            return (
                torch.zeros_like(self.buffers.weights.gumbel.target[:, 0])
                .scatter_(-1, remap(indices, mapping), values)
            )

    def selection(
        self,
        logits: Tensor,
        weights: Tensor,
        pair: ValueIndices | None = None,
        mapping: Tensor | None = None,
        logprobs: bool = False,
        probs: bool = False,
    ) -> Selection:

        log_probs = top_logprobs = probabilities = None

        if self.is_greedy:

            tokens = remap(logits.argmax(dim=-1, keepdim=True), mapping=mapping)

            if logprobs:

                log_probs = LogProbabilities(selected=logits.new_zeros(tokens.shape))

            if probs:

                values = logits.new_ones(tokens.shape)
                probabilities = self.probabilities(values, indices=tokens)

        elif pair is None:

            # Unbounded & Unordered.

            targets = (
                (logits - subset_select(weights, mapping))
                .argmax(dim=-1, keepdim=True)
            )

            tokens = remap(targets, mapping=mapping)

            if probs:

                scores, log_scores = softmax(logits)

                if logprobs:

                    selected = log_scores.gather(-1, targets)

                    if self.is_enabled(Indicator.TOP_K_LOGPROBS):

                        values, indices = log_scores.topk(k=MAX_TOP_LP, dim=-1)

                        targets = remap(indices, mapping=mapping)
                        top_logprobs = TopTargets(tokens=targets, logprobs=values)

                    log_probs = LogProbabilities(selected=selected, top_k=top_logprobs)

                probabilities = self.probabilities(scores, mapping=mapping)

            elif logprobs:

                selected = (
                    logits.gather(-1, targets)
                    .sub_(lse := logits.logsumexp(dim=-1, keepdim=True))
                )

                if self.is_enabled(Indicator.TOP_K_LOGPROBS):

                    values, indices = logits.topk(k=MAX_TOP_LP, dim=-1)

                    targets = remap(indices, mapping=mapping)
                    top_logprobs = TopTargets(tokens=targets, logprobs=values.sub_(lse))

                log_probs = LogProbabilities(selected=selected, top_k=top_logprobs)

        else:

            # Bounded or Ordered.

            values, indices = pair
            indices = remap(indices, mapping=mapping)

            targets = (
                (values - weights.gather(-1, indices))
                .argmax(dim=-1, keepdim=True)
            )

            tokens = indices.gather(-1, targets)

            if probs:

                scores, log_scores = softmax(values)

                if logprobs:

                    selected = log_scores.gather(-1, targets)

                    if self.is_enabled(Indicator.TOP_K_LOGPROBS):

                        targets = indices[:, :MAX_TOP_LP]
                        log_scores = log_scores[:, :MAX_TOP_LP]

                        top_logprobs = TopTargets(tokens=targets, logprobs=log_scores)

                    log_probs = LogProbabilities(selected=selected, top_k=top_logprobs)

                probabilities = self.probabilities(scores.to(dtype=BF16), indices=indices)

            elif logprobs:

                lse = values.to(F32).logsumexp(dim=-1, keepdim=True)
                selected = values.gather(-1, targets).sub_(lse)

                if self.is_enabled(Indicator.TOP_K_LOGPROBS):

                    targets = indices[:, :MAX_TOP_LP]
                    log_scores = values[:, :MAX_TOP_LP].sub_(lse)

                    top_logprobs = TopTargets(tokens=targets, logprobs=log_scores)

                log_probs = LogProbabilities(selected=selected, top_k=top_logprobs)

        return Selection(tokens=tokens, probabilities=probabilities, logprobs=log_probs)

    def verification(
        self,
        logits: Tensor,
        weights: Tensor,
        tokens: Tensor,
        probs: Tensor,
        pair: ValueIndices | None = None,
        logprobs: bool = False,
    ) -> Verification:

        log_probs = top_logprobs = None

        if self.is_greedy:

            selections = logits.argmax(dim=-1).type_as(tokens)
            acceptances = selections == tokens

            acceptances[:, -1].zero_()

            offsets = (
                acceptances.int()
                .argmin(dim=-1, keepdim=True)
            )

            if logprobs:

                log_probs = LogProbabilities(selected=logits.new_zeros(tokens.shape))

        elif pair is None:

            # Unbounded & Unordered.

            scores, log_scores = softmax(logits)

            drafted = tokens[..., None]

            acceptances = (
                probs.gather(-1, drafted)
                .mul_(self.buffers.weights.uniform[..., None])
                .le(scores.gather(-1, drafted))
                .squeeze(-1)
            )

            acceptances[:, -1].zero_()

            offsets = acceptances.int().argmin(dim=-1, keepdim=True)
            rejected = offsets[..., None].expand(-1, 1, logits.size(-1))

            sampled = (
                scores.gather(1, rejected)
                .sub_(probs.gather(1, rejected))
                .squeeze_(1)
                .clamp_min_(0)
                .log_()
                .sub_(weights.gather(1, rejected).squeeze_(1))
                .argmax(dim=-1, keepdim=True)
            )

            selections = tokens.scatter(-1, offsets, sampled)

            if logprobs:

                selected = (
                    log_scores.gather(-1, selections[..., None])
                    .squeeze_(-1)
                )

                if self.is_enabled(Indicator.TOP_K_LOGPROBS):

                    values, indices = log_scores.topk(k=MAX_TOP_LP, dim=-1)

                    top_logprobs = TopTargets(tokens=indices, logprobs=values)

                log_probs = LogProbabilities(selected=selected, top_k=top_logprobs)

        else:

            # Bounded or Ordered.

            values, indices = pair
            scores, log_scores = softmax(values)

            drafted = tokens[..., None]

            positions = (
                (selections := (indices == drafted))
                .int()
                .argmax(dim=-1, keepdim=True)
            )

            matched = selections.any(dim=-1)

            targets = (
                scores.gather(-1, positions)
                .squeeze(-1)
                .masked_fill_(~matched, 0)
            )

            acceptances = (
                probs.gather(-1, drafted)
                .squeeze(-1)
                .mul_(self.buffers.weights.uniform)
                .le(targets)
            )

            acceptances[:, -1].zero_()

            offsets = acceptances.int().argmin(dim=-1, keepdim=True)
            rejected = offsets[..., None].expand(-1, 1, MAX_K)

            targets = indices.gather(1, rejected).squeeze_(1)
            nested = (offsets * probs.size(-1)) + targets

            marginals = probs.flatten(1, 2).gather(-1, nested)

            sampled = (
                scores.gather(1, rejected)
                .squeeze_(1)
                .sub_(marginals)
                .clamp_min_(0)
                .log_()
                .sub_(weights.flatten(1, 2).gather(-1, nested))
                .argmax(dim=-1, keepdim=True)
            )

            targets = targets.gather(-1, sampled).type_as(tokens)
            selections = tokens.scatter(-1, offsets, targets)

            if logprobs:

                positions = (
                    positions.squeeze_(-1)
                    .scatter_(-1, offsets, sampled)
                    .unsqueeze_(-1)
                )

                selected = (
                    log_scores.gather(-1, positions)
                    .squeeze_(-1)
                )

                if self.is_enabled(Indicator.TOP_K_LOGPROBS):

                    top_logprobs = TopTargets(
                        tokens=indices[..., :MAX_TOP_LP],
                        logprobs=log_scores[..., :MAX_TOP_LP],
                    )

                log_probs = LogProbabilities(selected=selected, top_k=top_logprobs)

        return Verification(tokens=selections, offsets=offsets, logprobs=log_probs)

    def preamble(
        self,
        logits: Tensor,
        tokens: Tensor | None = None,
        timestep: int | None = None,
        mapping: Tensor | None = None,
    ) -> ValueIndices | None:

        grammar = self.buffers.grammar
        drafted = tokens

        if timestep is not None:

            grammar = grammar[:, timestep]

        if timestep == 0:

            drafted = None

        masked = self.grammar(logits=logits, bitmask=grammar, mapping=mapping)
        penalized = self.repetition(masked, timestep=timestep, tokens=drafted, mapping=mapping)

        biased = self.bias(penalized, mapping=mapping)
        scaled = self.temperature(biased)

        return self.cumulative(scaled)

    def singular(
        self,
        logits: Tensor,
        tokens: Tensor | None,
        timestep: int,
        *,
        mapping: Tensor | None = None,
        logprobs: bool = False,
        probs: bool = False,
    ) -> Selection:

        pair = self.preamble(logits=logits, tokens=tokens, timestep=timestep, mapping=mapping)
        weights = self.buffers.weights.gumbel.draft[:, timestep]

        return self.selection(logits, weights, pair, mapping, logprobs, probs)

    def multistep(
        self,
        logits: Tensor,
        tokens: Tensor,
        probs: Tensor,
        *,
        logprobs: bool = False,
    ) -> Tuple[Verification, ValueIndices | None]:

        pair = self.preamble(logits=logits, tokens=tokens)
        weights = self.buffers.weights.gumbel.target

        verification = self.verification(logits, weights, tokens, probs, pair, logprobs)

        return verification, pair

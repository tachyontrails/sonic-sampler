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

from collections.abc import Iterable
from contextlib import nullcontext
from itertools import chain, compress, repeat
from random import seed as py_seed
from typing import List, Self, Tuple

from torch import Event, Generator, Stream, Tensor

from triton import cdiv     # noqa.

from sonic_sampler.base import (
    Layout,
    PrefixContiguousTensor,
    SliceBuffer,
    MAX_TOP_LP,
)
from sonic_sampler.core.flags import Flags, Indicator, ScopedIndicators
from sonic_sampler.ops.base import MAX_K
from sonic_sampler.ops.copy import indexed_copy


__all__ = ["SamplingBuffers"]


I32: torch.dtype = torch.int32
I64: torch.dtype = torch.int64

U32: torch.dtype = torch.uint32
BOOL: torch.dtype = torch.bool

BF16: torch.dtype = torch.bfloat16
TINY: float = torch.finfo(BF16).tiny


class GumbelNoise(SliceBuffer):

    # The noise weights to apply during verification via rejection sampling.
    target: PrefixContiguousTensor

    # The noise weights used during drafting for speculative token selection.
    draft: PrefixContiguousTensor

    @classmethod
    def empty(cls, size: int, timesteps: int, vocab_size: int) -> GumbelNoise:

        shape = (2, size, timesteps, vocab_size)
        layout = Layout(size, timesteps)

        target, draft = PrefixContiguousTensor(torch.empty(shape, dtype=BF16))

        return cls(target=target, draft=draft, layout=layout)

    @property
    def device(self) -> torch.device:

        return self.target.device

    @property
    def vocab_size(self) -> int:

        return self.target.size(-1)

    def subset(self, indices: Tensor) -> GumbelNoise:

        return GumbelNoise(
            target=self.target.index_select(dim=0, index=indices),
            draft=self.draft.index_select(dim=0, index=indices),
            layout=Layout(indices.size(0), self.layout.cols),
        )

    def populate(
        self,
        row: slice | int,
        generator: Generator | None,
        prefill: bool = False,
        draft: bool = False,
    ) -> Self:

        if prefill:

            tensors = (self.target[row, 0], self.draft[row]) if draft else (self.target[row, 0],)

        else:

            # During decoding, the `stage_weights` method call is expected to precede this
            # `populate` call, if `draft` is True i.e. draft sampling is stochastic and we
            # aim for on-policy rejection sampling.

            selected = (self.draft if draft else self.target)
            tensors = (selected[row],)

        for weight in tensors:

            weight.exponential_(1.0, generator=generator).log_()

        return self

    def update(
        self,
        targets: List[Tensor | None],
        positions: List[int] | None = None,
        stream: Stream | None = None,
    ) -> Self:
        """
        Updates the underlying device-resident `target` noise weights given a list of
        device-resident weights (when non-null) either via sequential ordering or at
        the targeted `positions` when provided.

        Note(s):

            • The copy operation may conditionally take place under the given `stream` when
              provided.

        """
        if any(selectors := [t is not None for t in targets]):

            iterator = zip(positions, targets) if positions else enumerate(targets)

            with (stream or nullcontext()):

                for i, target in compress(iterator, selectors):

                    self.target[i].copy_(target, non_blocking=True)

        return self

    def stage_weights(
        self, indices: Tensor, indicator: Tensor, stream: Stream | None = None
    ) -> Self:

        with (stream or nullcontext()):

            indexed_copy(self.draft, self.target, indices, indicator)

        return self


DraftTargetSelector = tuple[bool, bool]
WeightUpdateContext = tuple[int | None, Generator | None, DraftTargetSelector, bool]


def resolve_updates(
    size: int,
    generator: Generator | List[Generator] | None = None,
    selectors: List[DraftTargetSelector] | None = None,
    positions: List[int] | None = None,
    prefill: bool | List[bool] = False,
) -> Iterable[WeightUpdateContext]:

    match generator, positions, selectors, prefill:

        case (Generator() | None) as shared, [] | None, [] | None, bool() as initial:

            return ((None, shared, (True, True), initial),)

        case (Generator() | None) as shared, [] | None, [] | None, []:

            return ((None, shared, (True, True), False),)

        case _:

            indices = positions or range(size)
            predicates = selectors or repeat((True, True))

            generators = generator if isinstance(generator, list) else repeat(generator)
            initials = prefill if isinstance(prefill, list) and prefill else repeat(bool(prefill))

            return zip(indices, generators, predicates, initials)


class StochasticWeights(SliceBuffer):

    # X ~ U(0, 1] for verification with shape [ B, γ + 1 ].
    uniform: PrefixContiguousTensor

    # X ~ -Gumbel(0, 1) for multinomial sampling with shape [ B, γ + 1, V ].
    gumbel: GumbelNoise

    @classmethod
    def random(
        cls, size: int, timesteps: int, vocab_size: int, generator: Generator | None = None
    ) -> StochasticWeights:

        return (
            cls.empty(size, timesteps, vocab_size)
            .update(generator=generator)
        )

    @classmethod
    def empty(cls, size: int, timesteps: int, vocab_size: int) -> StochasticWeights:

        uniform = PrefixContiguousTensor(torch.empty((size, timesteps), dtype=BF16))
        gumbel = GumbelNoise.empty(size, timesteps, vocab_size)

        layout = Layout(size, timesteps)

        return cls(uniform=uniform, gumbel=gumbel, layout=layout)

    @property
    def device(self) -> torch.device:

        return self.gumbel.device

    def subset(self, indices: Tensor) -> StochasticWeights:

        return StochasticWeights(
            uniform=self.uniform.index_select(dim=0, index=indices),
            gumbel=self.gumbel.subset(indices=indices),
            layout=Layout(indices.size(0), self.layout.cols),
        )

    def refresh(
        self,
        position: int | None,
        generator: Generator | None,
        prefill: bool,
        draft: bool,
    ) -> None:

        row = slice(None) if position is None else position

        self.gumbel.populate(row=row, generator=generator, prefill=prefill, draft=draft)

        if not prefill:

            self.uniform[row].uniform_(TINY, 1.0, generator=generator)

    def update(
        self,
        generator: Generator | List[Generator] | None = None,
        selectors: List[tuple[bool, bool]] | None = None,
        targets: List[Tensor | None] | None = None,
        positions: List[int] | None = None,
        prefill: bool | List[bool] = False,
        stream: Stream | None = None,
        event: Event | None = None,
    ) -> Self:
        """
        Refreshes the underlying weights for both Gumbel-Max sampling and stochastic verification
        between successive iterations. In addition to supporting a singular `Generator`, the
        following mode(s) are supported:

            • Non-targeted batch-wise `Generator`(s) i.e. via:

                weights.update(generator=[ G_0, G_1, G_2, ... ], positions=None)

              which is typically coupled with just-in-time batch composition strategies.

            • Targeted batch-wise `Generator`(s) i.e. via:

                weights.update(generator=[ G_3, G_1, G_7, ... ], positions=[ 3, 1, 7, ... ])

              which is more applicable for slot-based pre-allocated batching strategies.

            • The prefill-conditioned update constrains the weight refresh to only the underlying
              `target` gumbel noise at the zero-th timestep.

            • Supports updating the underlying `target` gumbel noise with the given list of
              `targets` (when non-None) during decode i.e. prefill is False, either via sequential
              ordering or at the targeted `positions` when provided.

        """
        updates = resolve_updates(self.layout.rows, generator, selectors, positions, prefill)

        with (stream or nullcontext()):

            # Refresh weights at the targeted positions with the optionally given generator
            # accounting for prefill or decode step.

            for i, rng, (target, draft), initial in updates:

                if target:

                    self.refresh(position=i, generator=rng, prefill=initial, draft=draft)

            # Update the underlying `target` gumbel noise with the given list of `targets`.
            # This is primarily applicable for just-in-time batch composition strategies.

            if targets and not prefill:

                self.gumbel.update(targets=targets, positions=positions, stream=stream)

            if event:

                event.record(stream)

        return self

    def step(
        self,
        positions: List[int],
        scoped: ScopedIndicators,
        indicator: Tensor,
        indices: Tensor | None = None,
        generator: Generator | List[Generator] | None = None,
        stream: Stream | None = None,
        event: Event | None = None,
    ) -> None:
        """
        Stages the `draft` Gumbel-Max weights at the given `indices` (i.e. the device-resident
        Tensor of `positions`) into the underlying `target` and refreshes the next `draft` weights
        at those `positions`.

        Note(s):

            • This is intended to run lockstep following each decoding iteration and largely
              applicable for slot-based pre-allocated batching strategies.

        """
        if indices is not None and scoped.has_stochastic_draft:

            self.gumbel.stage_weights(indices=indices, indicator=indicator, stream=stream)

        self.update(
            generator=generator,
            selectors=scoped.stochastic_selectors,
            positions=positions,
            stream=stream,
            event=event,
        )


class TokenCounts(SliceBuffer):

    # The static context / prompt token counts.
    context: Tensor

    # The continually updated generation / decode token counts.
    decode: Tensor

    # The host-resident pinned buffer for transferring context counts.
    pinned: Tensor

    @classmethod
    def random(
        cls, size: int, timesteps: int, vocab_size: int, generator: Generator | None = None
    ) -> TokenCounts:

        context, decode = torch.randint(
            low=0,
            high=32,
            size=(2, size, vocab_size),
            dtype=I32,
            generator=generator,
        )

        pinned = torch.empty_like(context, device="cpu", pin_memory=True)
        layout = Layout(size, timesteps)

        return cls(context=context, decode=decode, layout=layout, pinned=pinned)

    @classmethod
    def zeros(cls, size: int, timesteps: int, vocab_size: int) -> TokenCounts:

        context, decode = torch.zeros(size=(2, size, vocab_size), dtype=I32)

        pinned = torch.empty_like(context, device="cpu", pin_memory=True)
        layout = Layout(size, timesteps)

        return cls(context=context, decode=decode, layout=layout, pinned=pinned)

    def subset(self, indices: Tensor) -> TokenCounts:

        return TokenCounts(
            context=self.context.index_select(dim=0, index=indices),
            decode=self.decode.index_select(dim=0, index=indices),
            pinned=self.pinned.index_select(dim=0, index=indices),
            layout=Layout(indices.size(0), self.layout.cols),
        )

    def update(
        self,
        counts: List[Tuple[Tensor, Tensor]],
        positions: List[int] | None = None,
        stream: Stream | None = None,
    ) -> Self:
        """
        Updates the underlying device-resident `context` and `decode` counts given a list of
        context-decode counts pairs, which may possibly be host-resident.

        Note(s):

            • This method performs either a targeted or ordered non-blocking in-place copies of
              the given Tensor(s) into the underlying device-resident Tensor(s).

        """
        iterator = zip(positions, counts) if positions else enumerate(counts)

        with (stream or nullcontext()):

            for i, (context, decode) in iterator:

                self.context[i].copy_(context, non_blocking=True)
                self.decode[i].copy_(decode, non_blocking=True)

        return self

    def populate(self, encodings: List[Tensor], positions: List[int]) -> Self:

        for i, encoding in zip(positions, encodings):

            (
                self.pinned[i]
                .zero_()
                .scatter_add_(0, encoding, torch.ones_like(encoding, dtype=I32))
            )

        return self

    def allocate(
        self, encodings: List[Tensor], positions: List[int], stream: Stream | None = None
    ) -> Self:
        """
        Executes a targeted pinned-memory assisted H2D transfer into the underlying device-resident
        `context` counts at the given `positions` after histogramming the given `encodings` into
        the former pinned Tensor, optionally under an alternative given `stream`.

        Note(s):

            • This is largely applicable for slot-based pre-allocated batching strategies where
              allocation is done at request admission.

        """
        self.populate(encodings, positions)

        with (stream or nullcontext()):

            for i in positions:

                self.context[i].copy_(self.pinned[i], non_blocking=True)
                self.decode[i].zero_()

        return self

    def increment(self, tokens: Tensor) -> Self:
        """
        Step-wise update of the underlying device-resident `decode` counts primarily for testing
        and debugging purposes.

        Note(s):

            • The actual update should take place within the `singular` or `multistep` kernels
              via the `update_counts` flag.

        """
        self.decode.scatter_add_(1, tokens, torch.ones_like(tokens))

        return self


class RepetitionPenalties(SliceBuffer):

    # The associated counts buffer of shape [ B, V ].
    counts: TokenCounts

    # The frequency penalties of shape [ B ].
    frequency: Tensor

    # The presence penalties of shape [ B ].
    presence: Tensor

    # The multiplicative i.e. original repetition penalties of shape [ B ].
    multiplicative: Tensor

    # The host-resident pinned buffer for transferring the penalties.
    pinned: Tensor

    @classmethod
    def random(
        cls,
        size: int,
        timesteps: int,
        vocab_size: int,
        generator: Generator | None = None,
        flags: Flags | None = None,
    ) -> RepetitionPenalties:

        counts = TokenCounts.random(size, timesteps, vocab_size, generator)

        pinned = torch.empty((3, size), dtype=BF16, device="cpu", pin_memory=True)
        layout = Layout(size, timesteps)

        frequency, presence = (
            torch.empty((2, size), dtype=BF16)
            .uniform_(-2, 2, generator=generator)
        )

        multiplicative = (
            torch.empty(size, dtype=BF16)
            .uniform_(0.5, 1.5, generator=generator)
        )

        if flags:

            frequency.masked_fill_(~flags.selectors(Indicator.FREQUENCY), 0)
            presence.masked_fill_(~flags.selectors(Indicator.PRESENCE), 0)

            multiplicative.masked_fill_(~flags.selectors(Indicator.MULTIPLICATIVE), 1)

        return cls(
            counts=counts,
            frequency=frequency,
            presence=presence,
            multiplicative=multiplicative,
            pinned=pinned.T,
            layout=layout,
        )

    @classmethod
    def default(cls, size: int, timesteps: int, vocab_size: int) -> RepetitionPenalties:

        frequency, presence = torch.zeros((2, size), dtype=BF16)
        multiplicative = torch.ones(size, dtype=BF16)

        pinned = torch.empty((3, size), dtype=BF16, device="cpu", pin_memory=True)
        layout = Layout(size, timesteps)

        return cls(
            counts=TokenCounts.zeros(size, timesteps, vocab_size),
            frequency=frequency,
            presence=presence,
            multiplicative=multiplicative,
            pinned=pinned.T,
            layout=layout,
        )

    def subset(self, indices: Tensor) -> RepetitionPenalties:

        return RepetitionPenalties(
            counts=self.counts.subset(indices=indices),
            frequency=self.frequency.index_select(dim=0, index=indices),
            presence=self.presence.index_select(dim=0, index=indices),
            multiplicative=self.multiplicative.index_select(dim=0, index=indices),
            pinned=self.pinned.index_select(dim=0, index=indices),
            layout=Layout(indices.size(0), self.layout.cols),
        )

    def populate(
        self,
        multiplicative: List[float],
        frequency: List[float],
        presence: List[float],
        positions: List[int] | None = None,
    ) -> Self:

        values = torch.tensor([multiplicative, frequency, presence], dtype=BF16, device="cpu")

        if positions:

            indices = torch.tensor(positions, device="cpu", dtype=I64)

            self.pinned.index_copy_(0, indices, values.T)

        else:

            self.pinned.copy_(values.T)

        return self

    def transfer(self, positions: List[int] | None, stream: Stream | None = None) -> Self:

        with (stream or nullcontext()):

            if positions:

                for i in positions:

                    self.multiplicative[i].copy_(self.pinned[i, 0], non_blocking=True)

                    self.frequency[i].copy_(self.pinned[i, 1], non_blocking=True)
                    self.presence[i].copy_(self.pinned[i, 2], non_blocking=True)

            else:

                self.multiplicative.copy_(self.pinned[:, 0], non_blocking=True)

                self.frequency.copy_(self.pinned[:, 1], non_blocking=True)
                self.presence.copy_(self.pinned[:, 2], non_blocking=True)

        return self

    def update(
        self,
        counts: List[Tuple[Tensor, Tensor]],
        multiplicative: List[float],
        frequency: List[float],
        presence: List[float],
        positions: List[int] | None = None,
        stream: Stream | None = None,
    ) -> Self:
        """
        Packs the given the `multiplicative`, `frequency`, and `presence` values into a host
        Tensor that's copied to the underlying host-resident pinned-memory Tensor, either
        directly or at the given `positions` before executing the corresponding H2D transfer
        into their respective device-resident Tensor(s).

        Note(s):

            • For how the underlying counts are updated given the `counts` tuples, see the
              `update` method in `TokenCounts`.

            • This is largely applicable for just-in-time batch composition strategies where
              prefix-sliced buffers are updated between compute iterations.

        """
        (
            self.populate(multiplicative, frequency, presence, positions)
            .transfer(positions, stream)
        )

        self.counts.update(counts, positions, stream)

        return self

    def allocate(
        self,
        encodings: List[Tensor],
        multiplicative: List[float],
        frequency: List[float],
        presence: List[float],
        positions: List[int],
        stream: Stream | None = None,
    ) -> Self:
        """
        Packs the given the `multiplicative`, `frequency`, and `presence` values into a host
        Tensor that's copied to the underlying host-resident pinned-memory Tensor at the given
        `positions` before executing the targeted H2D transfer into their respective
        device-resident Tensor(s), optionally under an alternative given `stream`.

        Note(s):

            • For how the underlying counts are allocated given the `encodings`, see the
              `allocate` method in `TokenCounts`.

            • This is largely applicable for slot-based pre-allocated batching strategies where
              allocation is done at request admission.

        """
        (
            self.populate(multiplicative, frequency, presence, positions)
            .transfer(positions, stream)
        )

        self.counts.allocate(encodings, positions, stream)

        return self


def initialize_generator(seed: int | None, device: torch.device) -> Generator | None:

    if seed is not None:

        py_seed(seed)

        return (
            torch.Generator(device)
            .manual_seed(seed)
        )


class PinnedRelay(SliceBuffer):

    # The logit bias buffers.
    bias: Tensor

    # The packed temperature, top_p, and min_p buffers.
    float_values: Tensor

    # The top-k and top-k logprobs buffers.
    int_values: Tensor

    @classmethod
    def default(cls, size: int, timesteps: int, vocab_size: int) -> PinnedRelay:

        float_values = torch.empty((3, size), dtype=BF16, device="cpu", pin_memory=True)
        int_values = torch.empty((2, size), dtype=I32, device="cpu", pin_memory=True)

        bias = torch.empty((size, vocab_size), dtype=BF16, device="cpu", pin_memory=True)
        layout = Layout(size, timesteps)

        return cls(bias=bias, float_values=float_values.T, int_values=int_values.T, layout=layout)

    def subset(self, indices: Tensor) -> PinnedRelay:

        return PinnedRelay(
            bias=self.bias.index_select(dim=0, index=indices),
            float_values=self.float_values.index_select(dim=0, index=indices),
            int_values=self.int_values.index_select(dim=0, index=indices),
            layout=Layout(indices.size(0), self.layout.cols),
        )

    def update_bias(
        self, biases: List[dict[int, float] | None], positions: List[int] | None
    ) -> Self:

        iterator = zip(positions, biases) if positions else enumerate(biases)

        targets, tokens, values, lengths = zip(
            *(
                (i, mapping.keys(), mapping.values(), len(mapping))
                for i, mapping in iterator if mapping
            )
        )

        tokens = torch.tensor(list(chain.from_iterable(tokens)), dtype=I64, device="cpu")
        values = torch.tensor(list(chain.from_iterable(values)), dtype=BF16, device="cpu")

        offsets = list(chain.from_iterable(map(repeat, targets, lengths)))
        targets = torch.tensor(targets, dtype=I64, device="cpu")

        self.bias.index_fill_(0, targets, 0)

        offsets = (
            torch.tensor(offsets, dtype=I64, device="cpu")
            .mul_(self.bias.size(-1))
            .add_(tokens)
        )

        self.bias.flatten().scatter_(0, offsets, values)

        return self

    def populate(
        self,
        biases: List[dict[int, float] | None],
        temperature: List[float],
        top_k: List[int],
        top_p: List[float],
        min_p: List[float],
        top_logprobs: List[int],
        positions: List[int] | None = None,
    ) -> Self:

        float_values = torch.tensor([temperature, top_p, min_p], dtype=BF16, device="cpu")
        int_values = torch.tensor([top_k, top_logprobs], dtype=I32, device="cpu")

        if positions:

            indices = torch.tensor(positions, device="cpu", dtype=I64)

            self.float_values.index_copy_(0, indices, float_values.T)
            self.int_values.index_copy_(0, indices, int_values.T)

        else:

            self.float_values.copy_(float_values.T)
            self.int_values.copy_(int_values.T)

        return self.update_bias(biases, positions) if any(biases) else self


class SamplingBuffers(SliceBuffer):

    # The grammar bitpacked masking tensor.
    grammar: PrefixContiguousTensor

    # The repetition penalty buffers.
    repetition: RepetitionPenalties

    # The logit bias buffers.
    bias: Tensor

    # The temperature buffers.
    temperature: Tensor

    # The top-k buffers.
    top_k: Tensor

    # The top-p buffers.
    top_p: Tensor

    # The min-p buffers.
    min_p: Tensor

    # The top-k logprobs buffers.
    top_logprobs: Tensor

    # The pinned relay for non-blocking transfer(s).
    pinned: PinnedRelay

    # The stochastic weight buffers.
    weights: StochasticWeights

    # The bit-packed dynamic control tensor of shape [ B ].
    flags: Flags

    @classmethod
    def random(
        cls,
        size: int,
        timesteps: int,
        vocab_size: int,
        indicators: Indicator | List[Indicator] | None = None,
        device: torch.device | None = None,
        seed: int | None = None,
        bounded: bool = True,
    ) -> SamplingBuffers:
        """
        Note(s):

            • The chain of `random` methods across the nested struct hierarchy are primarily
              used for simplifying testing and debugging purposes.

        """
        target_device = device or torch.device(f'cuda:{torch.cuda.current_device()}')
        generator = initialize_generator(seed, target_device)

        with torch.device(target_device):

            flags = Flags.resolve(size, timesteps, target=indicators, random=True)
            layout = Layout(size, timesteps)

            repetition = RepetitionPenalties.random(
                size=size,
                timesteps=timesteps,
                vocab_size=vocab_size,
                generator=generator,
                flags=flags,
            )

            grammar = torch.randint(
                high=(1 << 32),
                size=(size, timesteps, cdiv(vocab_size, 32)),
                dtype=U32,
                generator=generator,
            )

            (
                grammar.view(I32)
                .masked_fill_(~flags.selectors(Indicator.GRAMMAR)[:, None, None], -1)
            )

            weights = StochasticWeights.random(size, timesteps, vocab_size, generator)

            temperature = (
                torch.empty(size, dtype=BF16)
                .uniform_(0.4, 1.25, generator=generator)
                .masked_fill_(~flags.selectors(Indicator.TEMPERATURE), 1)
            )

            upper = MAX_K if bounded else vocab_size

            top_k = (
                torch.randint(low=2, high=upper, size=(size,), dtype=I32, generator=generator)
                .masked_fill_(flags.selectors(Indicator.GREEDY), 1)
                .masked_fill_(~flags.selectors(Indicator.TOP_K), upper)
            )

            shape = (size, vocab_size)

            (
                (mask := torch.randint(2, shape, dtype=BOOL, generator=generator))
                .masked_fill_(flags.selectors(Indicator.BIAS)[:, None], False)
            )

            bias = (
                torch.empty(shape, dtype=BF16)
                .uniform_(-2.5, 2.5, generator=generator)
                .masked_fill_(mask, 0)
            )

            top_p = (
                torch.empty(size, dtype=BF16)
                .uniform_(0.5, 1, generator=generator)
                .masked_fill_(~flags.selectors(Indicator.TOP_P), 1)
            )

            min_p = (
                torch.empty(size, dtype=BF16)
                .uniform_(0.01, 0.5, generator=generator)
                .masked_fill_(~flags.selectors(Indicator.MIN_P), 1)
            )

            top_logprobs = (
                torch.randint(low=1, high=MAX_TOP_LP, size=(size,), dtype=I32, generator=generator)
                .masked_fill_(~flags.selectors(Indicator.TOP_K_LOGPROBS), 0)
            )

            pinned = PinnedRelay.default(size, timesteps, vocab_size)

            return cls(
                grammar=PrefixContiguousTensor(grammar),
                repetition=repetition,
                bias=bias,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                top_logprobs=top_logprobs,
                pinned=pinned,
                weights=weights,
                flags=flags,
                layout=layout,
            )

    @classmethod
    def default(
        cls,
        size: int,
        timesteps: int,
        vocab_size: int,
        device: torch.device | None = None,
        bounded: bool = True,
    ) -> SamplingBuffers:

        target_device = device or torch.device(f'cuda:{torch.cuda.current_device()}')

        with torch.device(target_device):

            repetition = RepetitionPenalties.default(size, timesteps, vocab_size)

            weights = StochasticWeights.random(size, timesteps, vocab_size)
            grammar = torch.full((size, timesteps, cdiv(vocab_size, 32)), -1, dtype=U32)

            upper = MAX_K if bounded else vocab_size

            top_k = torch.full(size=(size,), fill_value=upper, dtype=I32)
            bias = torch.zeros((size, vocab_size), dtype=BF16)

            temperature = torch.ones(size, dtype=BF16)
            top_logprobs = torch.zeros((size,), dtype=I32)

            top_p = torch.ones((size,), dtype=BF16)
            min_p = torch.zeros((size,), dtype=BF16)

            pinned = PinnedRelay.default(size, timesteps, vocab_size)

            flags = Flags.default(size, timesteps)
            layout = Layout(size, timesteps)

            return cls(
                grammar=PrefixContiguousTensor(grammar),
                repetition=repetition,
                bias=bias,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                top_logprobs=top_logprobs,
                pinned=pinned,
                weights=weights,
                flags=flags,
                layout=layout,
            )

    @property
    def indicators(self) -> ScopedIndicators:

        return self.flags.indicators

    @property
    def device(self) -> torch.device:

        return self.weights.device

    @property
    def vocab_size(self) -> int:

        return self.weights.gumbel.vocab_size

    def subset(self, host_indices: List[int], device_indices: Tensor) -> SamplingBuffers:
        """
        Note(s):

            • The chain of `subset` methods across the nested struct hierarchy are primarily
              used for simplifying testing and debugging purposes.

        """
        return SamplingBuffers(
            grammar=self.grammar.index_select(dim=0, index=device_indices),
            repetition=self.repetition.subset(indices=device_indices),
            bias=self.bias.index_select(dim=0, index=device_indices),
            temperature=self.temperature.index_select(dim=0, index=device_indices),
            top_k=self.top_k.index_select(dim=0, index=device_indices),
            top_p=self.top_p.index_select(dim=0, index=device_indices),
            min_p=self.min_p.index_select(dim=0, index=device_indices),
            top_logprobs=self.top_logprobs.index_select(dim=0, index=device_indices),
            pinned=self.pinned.subset(indices=device_indices),
            weights=self.weights.subset(indices=device_indices),
            flags=self.flags.subset(host_indices=host_indices, device_indices=device_indices),
            layout=Layout(len(host_indices), self.layout.cols),
        )

    def step_grammar(
        self,
        bitmasks: List[Tensor | None] | None,
        positions: List[int] | None = None,
        timestep: int | None = None,
        stream: Stream | None = None,
        event: Event | None = None,
    ) -> Self:

        if bitmasks and any(selectors := [m is not None for m in bitmasks]):

            iterator = positions or range(len(bitmasks))

            with (stream or nullcontext()):

                for i in compress(iterator, selectors):

                    (
                        (self.grammar[i] if timestep is None else self.grammar[i, timestep])
                        .copy_(bitmasks[i], non_blocking=True)
                    )

                if event:

                    event.record(stream)

        return self

    def step_weights(
        self,
        positions: List[int],
        indices: Tensor | None = None,
        generator: Generator | List[Generator] | None = None,
        stream: Stream | None = None,
        event: Event | None = None,
    ) -> Self:
        """
        Executes the post-forward pass update of the underlying stochastic weight buffer(s) at each
        decode iteration for the given `positions`.

        Note(s):

            • This is largely applicable for slot-based pre-allocated batching strategies.

            • This method would be a no-op if the target is greedy for all `positions`.

            • The draft-to-target indexed-copy only takes place if the `indices` are not `None`
              and at least one of the `positions` has stochastic drafting.

            • Under a stochastic-verify-greedy-drafting regime, only the underlying `target`
              buffer(s) are refreshed with the given `generator`(s).

            • This method can optionally execute under a given dedicated `stream` and mark the
              given `event`, if also provided.

        """
        scoped = self.flags.indicators.subset(positions)

        if scoped.has_stochastic_target:

            self.weights.step(
                positions=positions,
                scoped=scoped,
                indicator=self.flags.draft,
                indices=indices,
                generator=generator,
                stream=stream,
                event=event,
            )

        return self

    def transfer(
        self,
        positions: List[int] | None,
        stream: Stream | None = None,
        event: Event | None = None,
    ) -> Self:

        pinned = self.pinned

        with (stream or nullcontext()):

            if positions:

                for i in positions:

                    self.bias[i].copy_(pinned.bias[i], non_blocking=True)
                    self.temperature[i].copy_(pinned.float_values[i, 0], non_blocking=True)

                    self.top_p[i].copy_(pinned.float_values[i, 1], non_blocking=True)
                    self.min_p[i].copy_(pinned.float_values[i, 2], non_blocking=True)

                    self.top_k[i].copy_(pinned.int_values[i, 0], non_blocking=True)
                    self.top_logprobs[i].copy_(pinned.int_values[i, 1], non_blocking=True)

            else:

                self.bias.copy_(pinned.bias, non_blocking=True)
                self.temperature.copy_(pinned.float_values[:, 0], non_blocking=True)

                self.top_p.copy_(pinned.float_values[:, 1], non_blocking=True)
                self.min_p.copy_(pinned.float_values[:, 2], non_blocking=True)

                self.top_k.copy_(pinned.int_values[:, 0], non_blocking=True)
                self.top_logprobs.copy_(pinned.int_values[:, 1], non_blocking=True)

            if event is not None:

                event.record(stream)

        return self

    def update(
        self,
        counts: List[Tuple[Tensor, Tensor]],
        target_gumbel: List[Tensor | None],
        multiplicative: List[float],
        frequency: List[float],
        presence: List[float],
        biases: List[dict[int, float] | None],
        temperature: List[float],
        top_k: List[int],
        top_p: List[float],
        min_p: List[float],
        top_logprobs: List[int],
        positions: List[int] | None = None,
        indicators: List[Tuple[Indicator, Indicator]] | None = None,
        greedy_drafts: bool | List[bool] = False,
        prefill: bool | List[bool] = False,
        bitmasks: List[Tensor | None] | None = None,
        stream: Stream | None = None,
        generator: Generator | List[Generator] | None = None,
    ) -> Event | None:

        # Initialize conditional copy-stream event marker.

        event = Event() if stream else None

        # Dispatch (possibly pinned-memory) transfers to populate grammar bitmask(s).

        self.step_grammar(bitmasks, positions, stream=stream)

        # Resolve the scoped indicators (if not provided) given sampling parameter(s).

        if (scoped := indicators) is None:

            scoped = ScopedIndicators.from_params(
                size=len(counts),
                timesteps=self.layout.cols,
                multiplicative=multiplicative,
                frequency=frequency,
                presence=presence,
                biases=biases,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                top_logprobs=top_logprobs,
                bitmasks=bitmasks,
                greedy_drafts=greedy_drafts,
            )

        else:

            scoped = ScopedIndicators.from_pairs(scoped, timesteps=self.layout.cols)

        # Allocate i.e. populate and dispatch pinned-memory transfer of indicator(s).

        self.flags.update(indicators=scoped, positions=positions, stream=stream)

        # Populate host-resident pinned-memory buffers for non-repetition parameter(s).

        self.pinned.populate(
            biases=biases,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            top_logprobs=top_logprobs,
            positions=positions,
        )

        # Populate the underlying pinned-memory buffers with repetition penalties.

        self.repetition.update(
            counts=counts,
            multiplicative=multiplicative,
            frequency=frequency,
            presence=presence,
            positions=positions,
            stream=stream,
        )

        if scoped.has_stochastic_target:

            # Populate the Gumbel-Max weights for initial prefill and drafting.

            self.weights.update(
                generator=generator,
                selectors=scoped.stochastic_selectors,
                targets=target_gumbel,
                positions=positions,
                prefill=prefill,
                stream=stream,
            )

        # Dispatch pinned-memory transfers for repetition penalties and record the event.

        self.transfer(positions, stream=stream, event=event)

        return event

    def allocate(
        self,
        encodings: List[Tensor],
        multiplicative: List[float],
        frequency: List[float],
        presence: List[float],
        biases: List[dict[int, float] | None],
        temperature: List[float],
        top_k: List[int],
        top_p: List[float],
        min_p: List[float],
        top_logprobs: List[int],
        positions: List[int],
        indicators: ScopedIndicators | None = None,
        greedy_drafts: bool | List[bool] = False,
        bitmasks: List[Tensor | None] | None = None,
        stream: Stream | None = None,
        generator: Generator | List[Generator] | None = None,
    ) -> Event | None:
        """
        Updates the sampling parameter(s) and support buffer(s) at the given `positions` while
        accounting for overlap scheduling by populating pinned-memory buffer(s) before dispatching
        the eventual H2D transfer under the optionally given `torch.Stream` instance.

        Note(s):

            • The transfers are finalized with an `torch.Event` record that's conditionally
              returned when a `stream` instance is provided.

            • The `encodings` are expected to be a list of int64 host-resident Tensor(s) i.e.
              corresponding to the token IDs of a given request prompt.

            • A `ScopedIndicators` instance will be resolved from the given set of parameter(s)
              when one is not provided with the `indicators` argument.

            • The `bitmasks` are expected to be a list of optional pinned-memory host-resident
              int32 Tensor(s) or `None` for positions which have `grammar` disabled.

            • In general, this method requires that all inputs are fully host-resident i.e. CPU
              tensors or pure Python object(s).

        """
        # Initialize conditional copy-stream event marker.

        event = Event() if stream else None

        # Dispatch pinned-memory transfers to initialize prefill grammar bitmask(s).

        self.step_grammar(bitmasks, positions, timestep=0, stream=stream)

        # Resolve the scoped indicators (if not provided) given sampling parameter(s).

        if (scoped := indicators) is None:

            scoped = ScopedIndicators.from_params(
                size=len(encodings),
                timesteps=self.layout.cols,
                multiplicative=multiplicative,
                frequency=frequency,
                presence=presence,
                biases=biases,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                top_logprobs=top_logprobs,
                bitmasks=bitmasks,
                greedy_drafts=greedy_drafts,
            )

        # Allocate i.e. populate and dispatch pinned-memory transfer of indicator(s).

        self.flags.update(indicators=scoped, positions=positions, stream=stream)

        # Populate host-resident pinned-memory buffers for non-repetition parameter(s).

        self.pinned.populate(
            biases=biases,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            top_logprobs=top_logprobs,
            positions=positions,
        )

        # Populate the underlying pinned-memory buffers with repetition penalties.

        self.repetition.allocate(
            encodings=encodings,
            multiplicative=multiplicative,
            frequency=frequency,
            presence=presence,
            positions=positions,
            stream=stream,
        )

        if scoped.has_stochastic_target:

            # Populate the Gumbel-Max weights for initial prefill and drafting.

            self.weights.update(
                generator=generator,
                selectors=scoped.stochastic_selectors,
                positions=positions,
                prefill=True,
                stream=stream,
            )

        # Dispatch pinned-memory transfers for repetition penalties and record the event.

        self.transfer(positions, stream=stream, event=event)

        return event

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

from contextlib import nullcontext
from enum import unique, StrEnum, auto, IntFlag
from functools import partial
from itertools import compress, repeat
from operator import lshift
from random import randint
from typing import Callable, List, Self

from torch import Stream, Tensor

from sonic_sampler.base import SliceBuffer, Layout, MAX_TOP_LP
from sonic_sampler.ops import MAX_K


BOOL: torch.dtype = torch.bool
U16: torch.dtype = torch.uint16

I16: torch.dtype = torch.int16
I64: torch.dtype = torch.int64


@unique
class SamplingMode(StrEnum):

    # Assigns indicator to the minimal greedy state i.e. 0x40.
    MINIMAL = auto()

    # Removes stochastic indicators and enables greedy.
    GREEDY = auto()

    # Enables all but greedy indicator i.e. 0x7BF.
    MAXIMAL = auto()

    # Disables the grammar indicator.
    NON_GRAMMAR = auto()


@unique
class Indicator(IntFlag):

    # Grammar Enablement -> 0x1.
    GRAMMAR = auto()

    # Repetition (Multiplicative) -> 0x2.
    MULTIPLICATIVE = auto()

    # Repetition (Frequency) -> 0x4.
    FREQUENCY = auto()

    # Repetition (Presence) -> 0x8.
    PRESENCE = auto()

    # Logit Bias -> 0x10.
    BIAS = auto()

    # Temperature -> 0x20.
    TEMPERATURE = auto()

    # Top-1 -> 0x40.
    GREEDY = auto()

    # Top-K -> 0x80.
    TOP_K = auto()

    # Top-P -> 0x100.
    TOP_P = auto()

    # Min-P -> 0x200.
    MIN_P = auto()

    # Output Top-K Log-Probs -> 0x400.
    TOP_K_LOGPROBS = auto()

    # Stochastic Verify over Greedy Drafting -> 0x800.
    STOCHASTIC_VERIFY_GREEDY_DRAFT = auto()

    @classmethod
    def all(cls) -> Indicator:

        value = 1 << len(cls.__members__)

        return cls(value - 1)

    @classmethod
    def stochastic(cls) -> Indicator:

        return (
            Indicator.TEMPERATURE
            | Indicator.TOP_K
            | Indicator.TOP_P
            | Indicator.MIN_P
        )

    @classmethod
    def maximal(cls) -> Indicator:

        return cls.all() ^ cls.GREEDY

    @classmethod
    def random(cls) -> Indicator:

        value = randint(0, cls.all())

        if (indicator := cls(value)) & cls.stochastic():

            indicator &= cls.maximal()

        return indicator

    @classmethod
    def default(cls) -> Indicator:

        return cls(0)

    @classmethod
    def repetition(cls) -> Indicator:

        return cls.MULTIPLICATIVE | cls.FREQUENCY | cls.PRESENCE

    @classmethod
    def indirection(cls) -> Indicator:

        return cls.GRAMMAR | cls.repetition() | cls.BIAS

    @classmethod
    def cumulative(cls) -> Indicator:

        return cls.TOP_K | cls.TOP_P | cls.MIN_P

    @classmethod
    def from_params(
        cls,
        grammar: bool = False,
        multiplicative: float = 1.0,
        frequency: float = 0.0,
        presence: float = 0.0,
        logit_bias: bool = False,
        temperature: float = 1.0,
        top_k: int = MAX_K,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k_logprobs: int = 0,
        greedy_draft: bool = False,
    ) -> Indicator:

        flags = [
            grammar,
            multiplicative != 1.0,
            frequency != 0.0,
            presence != 0.0,
            logit_bias,
            temperature != 1.0,
            (greedy := (top_k == 1)),
            1 < top_k < MAX_K,
            not greedy and (0.0 < top_p < 1.0),
            not greedy and (0.0 < min_p < 1.0),
            not greedy and (0 < top_k_logprobs <= MAX_TOP_LP),
            not greedy and greedy_draft,
        ]

        positions = range(len(flags))
        bits = map(int, flags)

        return cls(sum(map(lshift, bits, positions)))

    def greedy(self) -> Indicator:

        mask = (
            Indicator.all()
            ^ self.__class__.stochastic()
            ^ Indicator.TOP_K_LOGPROBS
            ^ Indicator.STOCHASTIC_VERIFY_GREEDY_DRAFT
        )

        return (self & mask) | Indicator.GREEDY

    def disable(self, value: Indicator) -> Indicator:

        mask = self.__class__.all() ^ value

        return self & mask

    def resolve(self, mode: SamplingMode) -> Indicator:

        match mode:

            case SamplingMode.MINIMAL:

                return Indicator.GREEDY

            case SamplingMode.MAXIMAL:

                return Indicator.maximal()

            case SamplingMode.GREEDY:

                return self.greedy()

            case SamplingMode.NON_GRAMMAR:

                return self.disable(Indicator.GRAMMAR)


def indexify(selectors: List[bool]) -> List[int]:

    return list(compress(range(len(selectors)), selectors))


class ScopedIndicators(SliceBuffer):

    # The target verifier-specific list of enabled flag(s).
    target: List[Indicator]

    # The drafter-specific list of enabled flag(s).
    draft: List[Indicator]

    @classmethod
    def shared(
        cls, size: int, timesteps: int, factory: Callable[[], Indicator]
    ) -> ScopedIndicators:

        draft = (
            (target := [factory() for _ in range(size)])
            .copy()
        )

        layout = Layout(size, timesteps)

        return cls(target=target, draft=draft, layout=layout)

    @classmethod
    def random(cls, size: int, timesteps: int) -> ScopedIndicators:

        return cls.shared(size=size, timesteps=timesteps, factory=Indicator.random)

    @classmethod
    def default(cls, size: int, timesteps: int) -> ScopedIndicators:

        return cls.shared(size=size, timesteps=timesteps, factory=Indicator.default)

    @classmethod
    def from_pairs(
        cls, pairs: List[tuple[Indicator, Indicator]], timesteps: int
    ) -> ScopedIndicators:

        target, draft = list(zip(*pairs))
        layout = Layout(len(target), timesteps)

        return cls(target=list(target), draft=list(draft), layout=layout)

    @classmethod
    def from_values(
        cls,
        size: int,
        timesteps: int,
        target: Indicator | List[Indicator],
        draft: Indicator | List[Indicator] | None = None,
    ) -> ScopedIndicators:

        layout = Layout(size, timesteps)

        match target, draft:

            case list() as primary, None:

                return cls(target=primary, draft=primary.copy(), layout=layout)

            case list() as primary, list() as secondary:

                return cls(target=primary, draft=secondary, layout=layout)

            case Indicator() as first, None:

                primary = [first] * size

                return cls(target=primary, draft=primary.copy(), layout=layout)

            case Indicator() as first, Indicator() as second:

                primary = [first] * size
                secondary = [second] * size

                return cls(target=primary, draft=secondary, layout=layout)

            case _:

                raise ValueError('unrecognized target or draft indicator(s)')

    @classmethod
    def from_params(
        cls,
        size: int,
        timesteps: int,
        multiplicative: List[float] | None = None,
        frequency: List[float] | None = None,
        presence: List[float] | None = None,
        biases: List[dict[int, float] | None] | None = None,
        temperature: List[float] | None = None,
        top_k: List[int] | None = None,
        top_p: List[float] | None = None,
        min_p: List[float] | None = None,
        top_logprobs: List[int] | None = None,
        bitmasks: List[Tensor | None] | None = None,
        greedy_drafts: bool | List[bool] = False,
    ) -> ScopedIndicators:

        if top_k and any(k > MAX_K for k in top_k):

            raise ValueError(f'`top_k` greater than {MAX_K} is not supported')

        else:

            factory = partial(repeat, times=size)
            drafter = greedy_drafts

            if not isinstance(drafter, list):

                drafter = list(factory(greedy_drafts))

            iterator = zip(
                (m is not None for m in bitmasks) if bitmasks else factory(False),
                multiplicative or factory(1.0),
                frequency or factory(0.0),
                presence or factory(0.0),
                (b is not None for b in biases) if biases else factory(False),
                temperature or factory(0.0),
                top_k or factory(MAX_K),
                top_p or factory(1.0),
                min_p or factory(0.0),
                top_logprobs or factory(0),
                drafter,
            )

            target = [Indicator.from_params(*args) for args in iterator]
            layout = Layout(size, timesteps)

            draft = [
                indicator.greedy() if greedy else indicator
                for greedy, indicator in zip(drafter, target)
            ]

            return cls(target=target, draft=draft, layout=layout)

    @property
    def greedy_draft(self) -> bool:

        return all(Indicator.GREEDY in value for value in self.draft)

    @property
    def greedy_target(self) -> bool:

        return all(Indicator.GREEDY in value for value in self.target)

    @property
    def has_stochastic_draft(self) -> bool:

        return any(Indicator.GREEDY not in value for value in self.draft)

    @property
    def has_stochastic_target(self) -> bool:

        return any(Indicator.GREEDY not in value for value in self.target)

    @property
    def stochastic_selectors(self) -> List[tuple[bool, bool]]:

        target = (Indicator.GREEDY not in value for value in self.target)
        draft = (Indicator.GREEDY not in value for value in self.draft)

        return list(zip(target, draft))

    def as_tensor(self, pinned: bool = False, host: bool = False) -> Tensor:

        match pinned, host:

            case True, _:

                kwargs = dict(device='cpu', pin_memory=True)

            case False, True:

                kwargs = dict(device='cpu')

            case _:

                kwargs = {}

        return torch.tensor([self.target, self.draft], dtype=U16, **kwargs)

    def subset(self, indices: List[int]) -> ScopedIndicators:

        return ScopedIndicators(
            target=[self.target[i] for i in indices],
            draft=[self.draft[i] for i in indices],
            layout=Layout(rows=len(indices), cols=self.layout.cols),
        )

    def update(self, other: ScopedIndicators, positions: List[int] | None = None) -> Self:

        if positions:

            for i, j in enumerate(positions):

                self.target[j] = other.target[i]
                self.draft[j] = other.draft[i]

        else:

            self.target[:] = other.target
            self.draft[:] = other.draft

        return self


class Flags(SliceBuffer):

    """
    Encapsulates the indicators per-request across the batch for both the target verifier
    and drafter with their Tensor representation(s) having the shape [ B, 2 ] and stride
    [ 1, B ].

    """

    # The readable list of enabled flag(s).
    indicators: ScopedIndicators

    # The host-side pinned bit-packed representation.
    pinned: Tensor

    # The device-side bit-packed representation.
    packed: Tensor

    @classmethod
    def from_scoped(cls, indicators: ScopedIndicators) -> Flags:

        layout = Layout((lt := indicators.layout).rows, lt.cols)

        pinned = indicators.as_tensor(pinned=True)
        packed = torch.as_tensor(pinned, dtype=U16)

        return cls(indicators=indicators, pinned=pinned.T, packed=packed.T, layout=layout)

    @classmethod
    def random(cls, size: int, timesteps: int) -> Flags:

        indicators = ScopedIndicators.random(size=size, timesteps=timesteps)

        return cls.from_scoped(indicators=indicators)

    @classmethod
    def default(cls, size: int, timesteps: int) -> Flags:

        indicators = ScopedIndicators.default(size=size, timesteps=timesteps)

        return cls.from_scoped(indicators=indicators)

    @classmethod
    def from_values(
        cls,
        size: int,
        timesteps: int,
        target: Indicator | List[Indicator],
        draft: Indicator | List[Indicator] | None = None,
    ) -> Flags:

        indicators = ScopedIndicators.from_values(
            size=size,
            timesteps=timesteps,
            target=target,
            draft=draft,
        )

        return cls.from_scoped(indicators=indicators)

    @classmethod
    def resolve(
        cls,
        size: int,
        timesteps: int,
        target: Indicator | List[Indicator] | None = None,
        draft: Indicator | List[Indicator] | None = None,
        random: bool = False,
    ) -> Flags:

        if target is None:

            return (cls.random if random else cls.default)(size, timesteps)

        else:

            return cls.from_values(size, timesteps, target, draft)

    @property
    def target(self) -> Tensor:

        return self.packed[:, 0]

    @property
    def draft(self) -> Tensor:

        return self.packed[:, 1]

    def subset(self, host_indices: List[int], device_indices: Tensor) -> Flags:

        host_selector = torch.tensor(host_indices, device='cpu')

        return Flags(
            indicators=self.indicators.subset(indices=host_indices),
            pinned=self.pinned.int().index_select(dim=0, index=host_selector).to(dtype=U16),
            packed=self.packed.index_select(dim=0, index=device_indices),
            layout=Layout(len(host_indices), self.layout.cols),
        )

    @property
    def all_greedy(self) -> bool:

        return self.indicators.greedy_target and self.indicators.greedy_draft

    def selectors(self, flag: Indicator) -> Tensor:

        return torch.tensor([(flag in value) for value in self.indicators.target], dtype=BOOL)

    def update(
        self, indicators: ScopedIndicators, positions: List[int] | None, stream: Stream | None
    ) -> Self:

        self.indicators.update(indicators, positions)

        values = indicators.as_tensor(pinned=False, host=True)

        if positions:

            indices = torch.tensor(positions, dtype=I64, device='cpu')

            (
                self.pinned.view(dtype=I16)
                .index_copy_(0, indices, values.T.view(dtype=I16))
            )

            with (stream or nullcontext()):

                for i in positions:

                    self.packed[i].copy_(self.pinned[i], non_blocking=True)

        else:

            self.pinned.copy_(values.T)

            with (stream or nullcontext()):

                self.packed.copy_(self.pinned, non_blocking=True)

        return self

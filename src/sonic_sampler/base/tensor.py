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

from typing import override

from torch import Tensor


class PrefixContiguousTensor(Tensor):

    """
    A `torch.Tensor` subclass that ensures that all 2D prefix slices are maximally
    contiguous.

    """

    def __slice__(self, rows: int, steps: int) -> Tensor:

        return (
            Tensor.__getitem__(self.flatten(0, 1), slice(rows * steps))
            .unflatten(0, (rows, steps))
        )

    def __can_slice__(self, s: slice, t: slice) -> tuple[int, int] | None:

        rows, steps, *_ = self.shape

        match (s.indices(rows), t.indices(steps)):

            case ((0, b, 1), (0, k, 1)) if b and k:

                return b, k

    @override
    def __getitem__(self, *args, **kwargs) -> Tensor:

        match args:

            case ((slice() as s, slice() as t),):

                if p := self.__can_slice__(s, t):

                    return self.__slice__(*p)

        return super().__getitem__(*args, **kwargs)


class TimeDifferentiatedTensor(PrefixContiguousTensor):

    """
    A `PrefixContiguousTensor` subclass that assumes that the 2D grid is batch-major in lieu
    of time-major as it is in the base class while still ensuring that all prefix slices are
    maximally contiguous.

    """

    @override
    def __getitem__(self, *args, **kwargs) -> Tensor:

        match (final := args):

            case ((slice() as rows, slice() as steps, *rest),):

                steps = slice(steps.start, (s := steps.stop) and (s - 1), steps.step)

                final = ((steps, rows, *rest),)

            case (slice() as cols,):

                final = ((slice(None), cols),)

        return super().__getitem__(*final, **kwargs)

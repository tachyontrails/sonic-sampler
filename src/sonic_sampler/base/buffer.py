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

from functools import partial
from itertools import product
from typing import Tuple, Self, Iterable, Iterator, Callable, Dict
from weakref import WeakKeyDictionary

from msgspec import Struct, field

from torch import Tensor

from sonic_sampler.base.tensor import PrefixContiguousTensor


__all__ = ["Layout", "SliceBuffer"]


GridKey = Tuple[int, int]
SliceIndexing = slice | Tuple[slice, slice]

Schedule = Iterable[int] | Iterable[GridKey] | Tuple[Iterable[int], Iterable[int]]


class Layout(Struct):

    # The batch size of the corresponding 1D / 2D grid.
    rows: int

    # The timesteps of the corresponding 2D grid.
    cols: int

    def __getitem__(self, indices: SliceIndexing) -> Layout:

        match indices:

            case slice() as s:

                _, end, _ = s.indices(self.rows)

                return Layout(end, self.cols)

            case (slice() as s, slice() as t):

                _, rows_end, _ = s.indices(self.rows)
                _, cols_end, _ = t.indices(self.cols)

                return Layout(rows_end, cols_end)

            case _:

                raise ValueError(f'unsupported slice indexing: {indices}')


def validate(index: slice, bound: int) -> Tuple[int, int]:

    start, stop, step = index.indices(bound)

    if step != 1:

        raise ValueError("interval slicing is not supported")

    elif not stop or start >= stop:

        raise ValueError(f"asked to slice an empty subset")

    else:

        return start, stop


class CachedPrefix[T](Struct):

    # The wrapped / decorated `__getitem__` method.
    method: Callable[[T, SliceIndexing], T]

    # The underlying cached slices.
    cache: WeakKeyDictionary[T, Dict[GridKey, T]] = field(default_factory=WeakKeyDictionary)

    def slice_rows(self, instance: T, index: slice) -> T:

        start, stop = validate(index, total := (layout := instance.layout).rows)

        if start:

            return self.method(instance, slice(start, stop, 1))

        elif stop == total:

            return instance

        else:

            cache = self.cache.setdefault(instance, {})

            if (key := (stop, layout.cols)) not in cache:

                cache[key] = self.method(instance, slice(0, stop, 1))

            return cache[key]

    def slice_pair(self, instance: T, rows: slice, cols: slice) -> T:

        col_start, col_stop = validate(cols, width := (layout := instance.layout).cols)

        if col_start:

            row_start, row_stop = validate(rows, layout.rows)
            indexing = (slice(row_start, row_stop, 1), slice(col_start, col_stop, 1))

            return self.method(instance, indexing)

        elif col_stop == width:

            return self.slice_rows(instance, rows)

        else:

            row_start, row_stop = validate(rows, layout.rows)

            if row_start:

                indexing = (slice(row_start, row_stop, 1), slice(col_start, col_stop, 1))

                return self.method(instance, indexing)

            else:

                cache = self.cache.setdefault(instance, {})

                if (key := (row_stop, col_stop)) not in cache:

                    indexing = (slice(0, row_stop, 1), slice(0, col_stop, 1))

                    cache[key] = self.method(instance, indexing)

                return cache[key]

    def __call__(self, instance: T, indices: SliceIndexing) -> T:

        match indices:

            case (slice() as s, slice() as t):

                return self.slice_pair(instance, s, t)

            case slice() as s:

                return self.slice_rows(instance, s)

            case _:

                raise ValueError('only 1d or 2d slicing are supported')

    def __get__(self, instance: T | None, _) -> Callable[[T, SliceIndexing], T]:

        return partial(self, instance) if instance else self


SliceTypes = Tensor | Layout | int | float | bool


class SliceBuffer(Struct, kw_only=True, weakref=True):

    # The associated layout of this buffer.
    layout: Layout

    # The internalized address-based cached hash value.
    __cached_hash__: int = -1

    def __tensors__(self) -> Iterator[Tensor]:

        for key in self.__struct_fields__:

            if hasattr(value := getattr(self, key), 'data_ptr'):

                yield value

            elif hasattr(value, '__tensors__'):

                yield from value.__tensors__()

    def __post_init__(self) -> None:

        self.__cached_hash__ = hash(tuple(t.data_ptr() for t in self.__tensors__()))

    def __slice__[T: (SliceTypes, SliceBuffer)](self, key: str, indices: SliceIndexing) -> T:

        match (value := getattr(self, key), indices):

            case (int() | float() | bool(), _):

                return value

            case (PrefixContiguousTensor() | Layout(), (slice() as s, slice() as t)):

                return value[s.start:s.stop, t.start:t.stop]

            case (_, slice() as s) | (_, (slice() as s, _)):

                return value[s.start:s.stop]

            case _:

                raise ValueError(f'unsupported slice indexing: {indices}')

    @CachedPrefix
    def __getitem__(self, indices: SliceIndexing) -> Self:

        fields = self.__struct_fields__

        values = (self.__slice__(key, indices) for key in fields)
        kwargs = dict(zip(fields, values))

        return self.__class__(**kwargs)

    def bisect(self, pivot: int) -> Tuple[Self, Self]:

        if 0 < pivot < self.layout.rows:

            return self[:pivot], self[pivot:]

        else:

            raise ValueError("asked to bisect at boundaries resulting in an empty slice")

    def prepopulate(self, schedule: Schedule) -> None:

        match schedule:

            case ((list() | tuple()) as rows, (list() | tuple()) as cols):

                for r, c in product(sorted(rows), sorted(cols)):

                    _ = self[:r, :c]

            case [tuple(), *_]:

                for r, c in sorted(schedule):

                    _ = self[:r, :c]

            case list() | tuple():

                for r in sorted(schedule):

                    _ = self[:r]

            case _:

                raise ValueError(f'unsupported schedule: {schedule}')

    def __eq__(self, other) -> bool:

        return self is other

    def __hash__(self) -> int:

        return self.__cached_hash__

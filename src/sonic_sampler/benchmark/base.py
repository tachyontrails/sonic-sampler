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

from collections.abc import Iterator, Sequence
from hashlib import blake2b
from itertools import chain
from statistics import median
from typing import Final


# The register budget of a CTA and of a single thread, in 32-bit registers.
REGISTER_FILE: Final[int] = 1 << 16
THREAD_REGISTERS: Final[int] = 255

# The threads per warp.
WARP_SIZE: Final[int] = 32

# The bytes per derived seed, the width `torch.manual_seed` accepts.
SEED_BYTES: Final[int] = 8

# The longest error summary a record keeps.
ERROR_LENGTH: Final[int] = 200


def is_power_of_2(value: int) -> bool:

    return value > 0 and int.bit_count(value) == 1


def floor_power_of_2(value: int) -> int:

    return 1 << (value.bit_length() - 1)


def power_of_2_range(start: int, end: int, base: int = 1) -> Iterator[int]:

    return (base << i for i in range(start, end))


def power_of_2_ladder(end: int, ridge: int) -> Iterator[int]:

    return chain(power_of_2_range(0, end), power_of_2_range(ridge - 1, end - 2, 3))


def register_budget(warps: int) -> int:

    return min(REGISTER_FILE, THREAD_REGISTERS * WARP_SIZE * warps)


def nullify(value: str) -> str | None:

    return value or None


def relative_range(values: Sequence[float]) -> float:

    return (max(values) - min(values)) / median(values)


def derive_seed(*values: int) -> int:
    """
    A seed determined by the given integers alone, stable across processes and Python versions.

    """
    payload = b"".join(value.to_bytes(SEED_BYTES, "little", signed=True) for value in values)
    digest = blake2b(payload, digest_size=SEED_BYTES).digest()

    return int.from_bytes(digest, "little")


def describe(error: BaseException) -> str:

    summary = str(error).partition("\n")[0]

    return f"{type(error).__name__}: {summary}"[:ERROR_LENGTH]

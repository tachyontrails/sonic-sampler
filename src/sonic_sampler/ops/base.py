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

from triton import cdiv, jit, language as tl                        # noqa.
from triton.language.core import builtin                            # noqa.
from triton.language.extra.libdevice import fast_expf, fast_logf    # noqa.


MAX_K: int = 128


def next_power_of_2(value: int) -> int:

    active_bits = int.bit_count(value)

    return (1 << int.bit_length(value)) if active_bits > 1 else value


@jit
def gdc_wait():

    tl.inline_asm_elementwise(
        asm="griddepcontrol.wait;",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@jit
def gdc_launch_dependents():

    tl.inline_asm_elementwise(
        asm="griddepcontrol.launch_dependents;",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@jit
def ones_like(other):

    return tl.full(other.shape, 1, dtype=other.dtype)


@jit
def log_softmax(logits, maximum):

    scores = logits.to(tl.float32) - maximum
    exp_sum = tl.sum(fast_expf(scores), axis=0)

    return scores - fast_logf(exp_sum)


@jit
def nucleus_masking(p_ptr, batch_id, logits, maximum, max_k: tl.constexpr):

    scores = fast_expf(logits.to(tl.float32) - maximum)
    ladder = tl.cumsum(scores, axis=0)

    scale = (
        tl.gather(ladder, tl.full((1,), max_k - 1, dtype=tl.int32), axis=0)
        * tl.load(p_ptr + batch_id).to(tl.float32)
    )

    return ladder <= scale


@builtin
def to_scalar(value, _semantic=None, _generator=None):

    return _semantic.unsplat(value)


@jit
def cumulative_masking(
    k_ptr,
    p_ptr,
    n_ptr,
    batch_id,
    indicator,
    values,
    maximum,
    offsets,
    max_k: tl.constexpr,
):

    minima: tl.constexpr = -float("inf")

    if (indicator & 0x80) > 0:

        subset_mask = offsets < tl.load(k_ptr + batch_id)
        values = tl.where(subset_mask, values, minima)

    if (indicator & 0x100) > 0:

        nucleus_mask = (
            nucleus_masking(p_ptr, batch_id, values, maximum, max_k)
            | (offsets == 0)
        )

        values = tl.where(nucleus_mask, values, minima)

    if (indicator & 0x200) > 0:

        pivot = fast_logf(tl.load(n_ptr + batch_id).to(tl.float32)) + maximum
        values = tl.where(values >= pivot, values, minima)

    return values

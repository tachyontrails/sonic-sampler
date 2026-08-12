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

from triton import cdiv, jit, language as tl   # noqa.

from sonic_sampler.ops.base import gdc_wait
from sonic_sampler.ops.encoding import bitpack_block
from sonic_sampler.ops.topk.adaptive import radix_bitonic
from sonic_sampler.ops.topk.radix import radix_threshold
from sonic_sampler.ops.topk.selection import bitonic_selection, radix_selection


@jit
def top_k_reduction(
    y_ptr,
    row_id,
    col_id,
    logits,
    indices,
    max_k: tl.constexpr,
    scale: tl.constexpr,
    enable_adaptive: tl.constexpr,
    enable_radix: tl.constexpr,
    enable_pdl: tl.constexpr,
    stride_y: tl.constexpr,
    block_n: tl.constexpr,
):

    if enable_adaptive:

        radix_bitonic(
            y_ptr,
            row_id,
            col_id,
            logits,
            indices,
            max_k,
            scale,
            enable_pdl,
            stride_y,
            block_n,
        )

    elif enable_radix:

        # Compute the threshold and marginal via radix filtering.

        encoded, threshold, margin = radix_threshold(logits, max_k)

        # Packed radix top-k reduction.

        radix_selection(
            y_ptr,
            row_id,
            col_id,
            encoded,
            threshold,
            margin,
            indices,
            max_k,
            enable_pdl,
            stride_y,
            block_n,
        )

    else:

        # Packed bitonic top-k reduction.

        packed = bitpack_block(logits, indices, block_n)

        bitonic_selection(
            y_ptr,
            row_id,
            col_id,
            packed,
            max_k,
            enable_pdl,
            stride_y,
            block_n,
        )


@jit
def bitpacked_reduction_kernel(
    # Input Data Pointer(s).
    x_ptr,                              # Logits -> [ L, V_d' / W ].
    # Output Data Pointer(s).
    y_ptr,                              # Scratch-Pad -> [ L, Z_v • M ].
    # Model Specific(s).
    vocab_size: tl.constexpr,           # Local Vocab Size.
    max_k: tl.constexpr,                # Maximum Top-K Value -> M.
    scale: tl.constexpr,                # Power-of-2 Integer Scale Factor.
    # Conditional Flag(s).
    enable_adaptive: tl.constexpr,      # Apply Adaptive Filtering & Reduction.
    enable_radix: tl.constexpr,         # Apply Radix Filtering & Reduction.
    enable_pdl: tl.constexpr,           # PDL Enablement Flag.
    # Stride(s).
    stride_x: tl.constexpr,             # Batch Stride of Logits.
    stride_y: tl.constexpr,             # Batch Stride of Scratch-Pad.
    # Kernel Specific(s).
    block_n: tl.constexpr,              # Block Size of Vocab Block(s).
) -> None:

    # Load and extract batch and block id(s) from program id(s).

    row_id = tl.program_id(axis=0)
    col_id = tl.program_id(axis=1)

    # Compute the column indices and corresponding mask.

    shift = col_id * block_n
    steps = tl.arange(0, block_n)

    cols = shift + steps
    mask = cols < vocab_size

    # Compute the block offsets and slot id (if has slot indirection).

    offsets = (row_id * stride_x) + cols

    if enable_pdl:

        gdc_wait()

    # Load the logits and the bit-indicator(s).

    logits = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))

    top_k_reduction(
        y_ptr,
        row_id,
        col_id,
        logits,
        steps,
        max_k,
        scale,
        enable_adaptive,
        enable_radix,
        enable_pdl,
        stride_y,
        block_n,
    )

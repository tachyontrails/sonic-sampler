from __future__ import annotations

import torch

from functools import lru_cache
from typing import Final

from torch import Tensor

from triton import cdiv, jit, language as tl        # noqa.


I64: torch.dtype = torch.int64
U16: torch.dtype = torch.uint16


BLOCK_SIZE: Final[int] = 8_192
NUM_WARPS: Final[int] = 16


@jit
def indexed_copy_kernel(
    x_ptr,
    y_ptr,
    j_ptr,
    z_ptr,
    total_cols: tl.constexpr,
    block_n: tl.constexpr,
    stride_x: tl.constexpr,
) -> None:

    row_id = (
        tl.load(j_ptr + tl.program_id(axis=0))
        .to(tl.int32)
    )

    indicator = tl.load(z_ptr + row_id)

    if (indicator & 0x40) == 0:

        margin = row_id * stride_x
        shift = tl.program_id(axis=1) * block_n

        span = tl.max_contiguous(tl.multiple_of(shift + tl.arange(0, block_n), block_n), block_n)
        mask = span < total_cols

        offsets = margin + span
        source = tl.load(x_ptr + offsets, mask=mask)

        tl.store(y_ptr + offsets, source, mask=mask, cache_modifier=".cs")


@lru_cache
def total_sm(device: torch.device) -> int:

    return (
        torch.cuda.get_device_properties(device)
        .multi_processor_count
    )


def resolve_grid_block(indices: Tensor, cols: int) -> tuple[tuple[int, int], int]:
    """
    Resolves the tiling grid shape by conditionally halving the module-fixed block size
    should saturating the trivial single-wave SM occupancy be possible.

    Note(s):

        • This is largely optimizes for the low batch size and / or small lookahead
          setting(s) for up to ~10% latency improvement.

    """
    rows = indices.size(0)
    blocks = cdiv(cols, size := BLOCK_SIZE)

    if (blocks * rows * 2) <= total_sm(indices.device):

        size //= 2
        blocks = cdiv(cols, size)

    return (rows, blocks), size


def indexed_copy(source: Tensor, target: Tensor, indices: Tensor, indicator: Tensor) -> None:
    """
    Fused `index_select` from `source` at the given `indices` and `index_copy_` into `target`,
    only for `indicator` values at those positions that do not have the 0x40 (greedy) bit set.

    Note(s):

        • Default `num_warps` and `block_n` values are chosen to nominally perform well across
          batch sizes with larger transfer rows capable of reaching ~70% bandwidth utilization.

    """
    stride_x = source.stride(0)
    total_cols = source.flatten(1).size(1)

    grid, block_size = resolve_grid_block(indices, total_cols)

    indexed_copy_kernel[grid](
        source,
        target,
        indices,
        indicator,
        total_cols,
        block_size,
        stride_x,
        num_warps=NUM_WARPS,
    )


def warmup_indexed_copy(source: Tensor, target: Tensor) -> None:
    """
    Facilitates triggering the underlying kernel compilation for both the minimum and maximum
    batch sizes to ensure coverage across the two block size configuration(s).

    """
    indices = torch.randperm(source.size(0), dtype=I64)
    indicator = torch.zeros_like(indices, dtype=U16)

    # Warm-up single batch size case to possibly trigger block size halving.

    indexed_copy(source, target, indices[:1], indicator[:1])

    # Warm-up largest batch size case for the default block size.

    indexed_copy(source, target, indices, indicator)

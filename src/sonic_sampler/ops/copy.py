from __future__ import annotations

from typing import Final

from torch import Tensor

from triton import cdiv, jit, language as tl        # noqa.


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


def indexed_copy(source: Tensor, target: Tensor, indices: Tensor, indicator: Tensor) -> None:
    """
    Fused `index_select` from `source` at the given `indices` and `index_copy_` into `target`,
    only when `indicator` at those positions do not have the 0x40 (greedy) bit set.

    Note(s):

        • Default `num_warps` and `block_n` values are chosen to nominally perform well across
          batch sizes with larger transfer rows capable of reaching ~70% bandwidth utilization.

    """
    stride_x = source.stride(0)
    total_cols = source.flatten(1).size(1)

    grid = (indices.size(0), cdiv(total_cols, BLOCK_SIZE))

    indexed_copy_kernel[grid](
        source,
        target,
        indices,
        indicator,
        total_cols,
        BLOCK_SIZE,
        stride_x,
        num_warps=NUM_WARPS,
    )

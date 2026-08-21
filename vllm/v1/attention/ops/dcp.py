# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA DCP collective selection and direct symmetric-memory implementations."""

from __future__ import annotations

import functools
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any, Protocol

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import get_dcp_group
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.ops.cp_common import (
    DirectCPWorkspace,
    direct_cp_enabled,
    direct_cp_multicast_enabled,
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from vllm.distributed.parallel_state import GroupCoordinator


# LSE/output combine


def mask_dcp_empty_shards_(
    lse: torch.Tensor,
    seq_lens: torch.Tensor | None,
    query_start_loc: torch.Tensor | None,
) -> None:
    if seq_lens is None and query_start_loc is None:
        return
    if seq_lens is None or query_start_loc is None:
        raise ValueError("seq_lens and query_start_loc must be provided together")
    if (
        seq_lens.ndim != 1
        or query_start_loc.ndim != 1
        or query_start_loc.shape[0] != seq_lens.shape[0] + 1
    ):
        raise ValueError("query_start_loc must contain one boundary per sequence")

    row_indices = torch.arange(
        lse.shape[0], device=lse.device, dtype=query_start_loc.dtype
    )
    sequence_indices = torch.searchsorted(
        query_start_loc[1:], row_indices, right=True
    ).clamp_max(seq_lens.shape[0] - 1)
    empty_rows = (row_indices >= query_start_loc[-1]) | (
        seq_lens[sequence_indices] == 0
    )
    lse.masked_fill_(empty_rows[:, None], float("-inf"))


# AG + RS/AR implementation


@triton.jit
def _correct_attn_cp_out_kernel(
    outputs_ptr,
    new_output_ptr,
    lses_ptr,
    vlse_ptr,
    outputs_stride_B,
    outputs_stride_H,
    outputs_stride_D,
    lses_stride_N,
    lses_stride_B,
    lses_stride_H,
    lse_idx,
    HEAD_DIM: tl.constexpr,
    N: tl.constexpr,
    N_ROUNDED: tl.constexpr,
    IS_BASE_E: tl.constexpr,
):
    """
    Apply the all-gathered lses to correct each local rank's attention
    output. we still need perform a cross-rank reduction to obtain the
    final attention output.

    Args:
        outputs_ptr (triton.PointerType):
            Pointer to input tensor of shape [ B, H, D ]
        lses_ptr (triton.PointerType):
            Pointer to input tensor of shape [ N, B, H ]
        new_output_ptr (triton.PointerType):
            Pointer to output tensor of shape [ B, H, D ]
        vlse_ptr (triton.PointerType):
            Pointer to output tensor of shape [ B, H ]
    """
    batch_idx = tl.program_id(axis=0).to(tl.int64)
    head_idx = tl.program_id(axis=1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)
    num_n_offsets = tl.arange(0, N_ROUNDED)
    valid_n_offsets = num_n_offsets < N

    # shape = [N]
    lse_offsets = (
        num_n_offsets * lses_stride_N
        + batch_idx * lses_stride_B
        + head_idx * lses_stride_H
    )

    # calc final lse
    lse = tl.load(
        lses_ptr + lse_offsets,
        mask=valid_n_offsets,
        other=-float("inf"),
    ).to(tl.float32)
    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0, lse_max)
    lse -= lse_max
    if IS_BASE_E:
        lse_exp = tl.exp(lse)
        lse_acc = tl.sum(lse_exp, axis=0)
        lse = tl.log(lse_acc)
    else:
        lse_exp = tl.exp2(lse)
        lse_acc = tl.sum(lse_exp, axis=0)
        lse = tl.log2(lse_acc)
    lse += lse_max

    lse_offsets = batch_idx * lses_stride_B + head_idx * lses_stride_H
    tl.store(vlse_ptr + lse_offsets, lse)

    # shape = [D]
    output_offsets = (
        batch_idx * outputs_stride_B
        + head_idx * outputs_stride_H
        + d_offsets * outputs_stride_D
    )

    # correct output
    lse_offset = (
        lse_idx * lses_stride_N + batch_idx * lses_stride_B + head_idx * lses_stride_H
    )
    lse_tmp = tl.load(lses_ptr + lse_offset).to(tl.float32)
    lse_finally = lse_tmp - lse
    lse_finally = tl.where(
        (lse_finally != lse_finally) | (lse_finally == float("inf")),
        -float("inf"),
        lse_finally,
    )
    factor = tl.exp(lse_finally) if IS_BASE_E else tl.exp2(lse_finally)
    output = tl.load(outputs_ptr + output_offsets)
    output = output * factor
    output = tl.where(factor == 0.0, 0.0, output)

    tl.store(new_output_ptr + output_offsets, output)


class CPTritonContext:
    """The CPTritonContext is used to avoid recompilation of the Triton JIT."""

    def __init__(self):
        self.inner_kernel = None

    def call_kernel(self, kernel, grid, *regular_args, **const_args):
        if self.inner_kernel is None:
            self.inner_kernel = kernel[grid](*regular_args, **const_args)
        else:
            self.inner_kernel[grid](*regular_args)


def correct_attn_out(
    out: torch.Tensor,
    lses: torch.Tensor,
    cp_rank: int,
    ctx: CPTritonContext,
    is_lse_base_on_e: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Correct the attention output using the all-gathered lses.

    Args:
        out: Tensor of shape [ B, H, D ]
        lses: Tensor of shape [ N, B, H ]
        cp_rank: Current rank in the context-parallel group
        ctx: Triton context to avoid recompilation

    Returns:
        Tuple of (out, lse) with corrected attention and final log-sum-exp.
    """
    if ctx is None:
        ctx = CPTritonContext()

    # --- Normalize to 3D views ---
    if out.ndim == 4 and out.shape[1] == 1:
        out = out.squeeze(1)
    assert out.ndim == 3, f"expected out [B,H,D] or [B,1,H,D], got {tuple(out.shape)}"

    if lses.ndim == 4 and lses.shape[-1] == 1:
        lses = lses.squeeze(-1)
    if lses.ndim == 4 and lses.shape[1] == 1:
        lses = lses.squeeze(1)
    assert lses.ndim == 3, (
        f"expected lses [N,B,H] (optionally with a 1-sized extra dim), "
        f"got {tuple(lses.shape)}"
    )

    B, H, D = out.shape
    N = lses.shape[0]

    # Strides after we normalized shapes to 3-D views.  The kernel computes
    # offsets for `vlse_ptr` using lses_stride_B/H, so the output buffer must
    # have the same B/H stride layout as a slice of `lses`.
    o_sB, o_sH, o_sD = out.stride()
    l_sN, l_sB, l_sH = lses.stride()

    # Allocate LSE with the same B/H strides as `lses` so writes land correctly
    # even when `lses` is a non-contiguous view (e.g., 4-D to 3-D squeeze).
    lse = torch.empty_strided(
        (B, H), (l_sB, l_sH), device=lses.device, dtype=lses.dtype
    )

    # Kernel launch config
    grid = (B, H, 1)

    regular_args = (
        out,
        out,
        lses,
        lse,
        o_sB,
        o_sH,
        o_sD,
        l_sN,
        l_sB,
        l_sH,
        cp_rank,
    )
    const_args = {
        "HEAD_DIM": D,
        "N": N,
        "N_ROUNDED": next_power_of_2(N),
        "IS_BASE_E": is_lse_base_on_e,
    }
    ctx.call_kernel(_correct_attn_cp_out_kernel, grid, *regular_args, **const_args)
    return out, lse


def _cp_lse_common(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    is_lse_base_on_e=True,
    seq_lens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
):
    """
    cp_attn_out: [ B, H, D ]
    cp_attn_lse: [ B, H ]
    """
    if cp_group.world_size == 1:
        return cp_attn_out

    if ctx is None:
        ctx = CPTritonContext()

    cp_attn_lse = cp_attn_lse.contiguous()
    mask_dcp_empty_shards_(cp_attn_lse, seq_lens, query_start_loc)
    lses = cp_group.all_gather(cp_attn_lse, dim=0).reshape(
        (cp_group.world_size,) + cp_attn_lse.shape
    )
    out, lse = correct_attn_out(
        cp_attn_out,
        lses,
        cp_group.rank_in_group,
        ctx,
        is_lse_base_on_e=is_lse_base_on_e,
    )
    return out, lse


def cp_lse_ag_out_rs(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e=True,
    seq_lens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    head_major_output: bool = False,
):
    """
    cp_attn_out: [ B, H, D ]
    cp_attn_lse: [ B, H ]
    """
    out, lse = _cp_lse_common(
        cp_attn_out,
        cp_attn_lse,
        cp_group,
        ctx=ctx,
        is_lse_base_on_e=is_lse_base_on_e,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
    )
    if head_major_output:
        out = cp_group.reduce_scatter_head_major(out, dim=1)
    else:
        out = cp_group.reduce_scatter(out, dim=1)

    if return_lse:
        cp_num_heads = lse.shape[1] // cp_group.world_size
        cp_rank = cp_group.rank_in_group
        lse = lse[:, cp_num_heads * cp_rank : cp_num_heads * (cp_rank + 1)]
        return out, lse
    return out


def cp_lse_ag_out_rs_into(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    output_provider: Callable[[torch.Tensor], torch.Tensor],
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
):
    """Correct DCP partials and reduce-scatter into borrowed output storage."""
    if cp_group.world_size <= 1:
        raise RuntimeError("cp_lse_ag_out_rs_into requires DCP world size > 1")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("cp_lse_ag_out_rs_into is eager-only")

    out, lse = _cp_lse_common(
        cp_attn_out,
        cp_attn_lse,
        cp_group,
        ctx=ctx,
        is_lse_base_on_e=is_lse_base_on_e,
    )
    output = output_provider(out)
    if not isinstance(output, torch.Tensor):
        raise TypeError("DCP output provider must return a tensor")
    out = cp_group.reduce_scatter_into(out, output, dim=1)

    if return_lse:
        cp_num_heads = lse.shape[1] // cp_group.world_size
        cp_rank = cp_group.rank_in_group
        lse = lse[:, cp_num_heads * cp_rank : cp_num_heads * (cp_rank + 1)]
        return out, lse
    return out


def cp_lse_ag_out_ar(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e=True,
    seq_lens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
):
    """
    cp_attn_out: [ B, H, D ]
    cp_attn_lse: [ B, H ]
    """
    out, lse = _cp_lse_common(
        cp_attn_out,
        cp_attn_lse,
        cp_group,
        ctx=ctx,
        is_lse_base_on_e=is_lse_base_on_e,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
    )
    out = cp_group.all_reduce(out)

    if return_lse:
        return out, lse
    return out


# B12X PCIe A2A implementation

_B12X_DCP_A2A_POOLS: dict[tuple[int, int, int, int, int, int], Any] = {}
_B12X_DCP_A2A_DISABLED: set[tuple[int, int, int, int, int, int]] = set()
# DCP likewise has one stable eager scheduler owner.  Graph target/draft/encoder
# identities are supplied separately by their GraphCaptureContext.
_B12X_DCP_EAGER_CHANNEL_ID = "vllm:eager:dcp"
_B12X_DCP_MAX_CONCURRENT_CHANNELS = 2
_DCP_A2A_GRAPH_BUFFERS: dict[
    tuple[tuple[int, ...], torch.device, torch.dtype],
    tuple[torch.Tensor, torch.Tensor],
] = {}


def _is_supported_bhd_layout(tensor: torch.Tensor) -> bool:
    """Accept packed token-major or capacity-strided head-major BHD views."""
    if tensor.ndim != 3 or int(tensor.stride(2)) != 1:
        return False
    batch, heads, head_dim = (int(value) for value in tensor.shape)
    stride_batch, stride_head, _ = (int(value) for value in tensor.stride())
    packed_token_major = stride_batch == heads * head_dim and stride_head == head_dim
    capacity_strided_head_major = (
        stride_batch == head_dim and stride_head >= batch * head_dim
    )
    return packed_token_major or capacity_strided_head_major


@functools.lru_cache(maxsize=1)
def _load_b12x_dcp_a2a_pool() -> Any | None:
    try:
        from b12x.comm.pcie import DcpAllToAllPool as PCIeDCPA2APool
    except Exception:
        return None
    return PCIeDCPA2APool


def _b12x_dcp_init_failed(
    cp_group: GroupCoordinator,
    device: torch.device,
    init_error: Exception | None,
) -> bool:
    """Reach consensus on pool initialization through its exchange group."""
    failed = torch.tensor(
        [int(init_error is not None)], dtype=torch.int32, device=device
    )
    dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=cp_group.device_group)
    return bool(failed.item())


def _get_b12x_dcp_a2a_pool(
    cp_group: GroupCoordinator,
    *,
    device: torch.device,
    total_heads: int,
    head_dim: int,
    query_head_dim: int,
    max_batch_size: int,
) -> Any | None:
    device_index = device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    key = (
        id(cp_group.device_group),
        int(device_index),
        int(total_heads),
        int(head_dim),
        int(query_head_dim),
        int(max_batch_size),
    )
    if key in _B12X_DCP_A2A_DISABLED:
        return None

    pool = _B12X_DCP_A2A_POOLS.get(key)
    if pool is not None:
        return pool

    # IPC allocation and handle exchange are not capture-safe. Dedicated
    # kernel warmup normally initializes this channel before graph capture.
    if torch.cuda.is_current_stream_capturing():
        return None
    pool_cls = _load_b12x_dcp_a2a_pool()
    if pool_cls is None:
        _B12X_DCP_A2A_DISABLED.add(key)
        return None

    init_error: Exception | None = None
    try:
        pool = pool_cls.from_exchange_group(
            exchange_group=cp_group.device_group,
            device=device,
            max_batch_size=max_batch_size,
            total_heads=total_heads,
            head_dim=head_dim,
            query_head_dim=query_head_dim,
            single_channel=False,
            max_concurrent_channels=_B12X_DCP_MAX_CONCURRENT_CHANNELS,
        )
        pool.prepare_channels((_B12X_DCP_EAGER_CHANNEL_ID,))
        pool.for_stream(channel_id=_B12X_DCP_EAGER_CHANNEL_ID)
    except Exception as exc:
        init_error = exc

    # Keep the status collective ordered with the NCCL group used above for
    # IPC-handle exchange. A separate Gloo group can be at a different startup
    # sequence when vLLM initializes overlapping TP/DCP coordinators.
    any_failed = _b12x_dcp_init_failed(cp_group, device, init_error)

    if any_failed:
        if pool is not None:
            pool.close()
        _B12X_DCP_A2A_DISABLED.add(key)
        if init_error is not None:
            logger.warning(
                "B12X PCIe DCP collective initialization failed; falling "
                "back to NCCL: %s",
                init_error,
            )
        return None

    assert pool is not None
    _B12X_DCP_A2A_POOLS[key] = pool
    logger.info(
        "Using B12X PCIe DCP collectives "
        "(world_size=%d, max_batch_size=%d, heads=%d, "
        "query_head_dim=%d, output_head_dim=%d).",
        cp_group.world_size,
        max_batch_size,
        total_heads,
        query_head_dim,
        head_dim,
    )
    return pool


@contextmanager
def capture_b12x_dcp_a2a(
    cp_group: GroupCoordinator,
    stream: torch.cuda.Stream | None = None,
    *,
    channel_id: str | None = None,
):
    """Bind registered B12X DCP pools to the graph's owning stream.

    Each graph capture receives independent channels; reusing channels across
    target and draft graphs would make one graph depend on another's lifetime.

    Args:
        cp_group: DCP group whose registered pools should enter capture.
        stream: CUDA stream owned by the enclosing graph capture.
        channel_id: Rank-stable semantic identity for the captured graph.
    """
    group_id = id(cp_group.device_group)
    matching_pools = sorted(
        (
            (key, pool)
            for key, pool in _B12X_DCP_A2A_POOLS.items()
            if key[0] == group_id
        ),
        key=lambda item: item[0][1:],
    )
    if matching_pools and channel_id is None:
        raise RuntimeError(
            "distributed PCIe DCP graph capture requires an explicit semantic "
            "channel_id"
        )
    with ExitStack() as stack:
        for _, pool in matching_pools:
            stack.enter_context(pool.capture(stream=stream, channel_id=channel_id))
        yield


def checkpoint_b12x_dcp_a2a_channels(
    cp_group: GroupCoordinator,
) -> tuple[int, dict[Any, tuple[Any, Any]]]:
    """Snapshot B12X DCP pools before a disposable graph capture."""
    group_id = id(cp_group.device_group)
    checkpoints = {
        key: (pool, pool.checkpoint_channels())
        for key, pool in _B12X_DCP_A2A_POOLS.items()
        if key[0] == group_id
    }
    return group_id, checkpoints


def rollback_b12x_dcp_a2a_channels(
    checkpoint: tuple[int, dict[Any, tuple[Any, Any]]],
) -> None:
    """Restore DCP pools after their disposable graphs are destroyed."""
    group_id, checkpoints = checkpoint
    for key, pool in list(_B12X_DCP_A2A_POOLS.items()):
        if key[0] != group_id:
            continue
        saved = checkpoints.get(key)
        if saved is None:
            pool.close()
            del _B12X_DCP_A2A_POOLS[key]
            continue
        saved_pool, channel_checkpoint = saved
        if pool is not saved_pool:
            pool.close()
            _B12X_DCP_A2A_POOLS[key] = saved_pool
        saved_pool.rollback_channels(channel_checkpoint)


def _try_b12x_dcp_lse_reduce(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    *,
    return_lse: bool,
    is_lse_base_on_e: bool,
    max_batch_size: int | None,
    query_head_dim: int | None,
) -> torch.Tensor | None:
    """Use the low-latency B12X PCIe path when its contract is satisfied."""
    world_size = cp_group.world_size
    if (
        return_lse
        or not cp_attn_out.is_cuda
        or cp_attn_out.dtype not in (torch.float16, torch.bfloat16)
        or cp_attn_lse.dtype != torch.float32
        or world_size not in (2, 4, 8)
        or cp_attn_out.ndim != 3
        or cp_attn_lse.shape != cp_attn_out.shape[:2]
    ):
        return None

    batch, total_heads, head_dim = cp_attn_out.shape
    if total_heads % world_size != 0 or head_dim % 8 != 0:
        return None

    if max_batch_size is None:
        max_batch_size = batch
    max_batch_size = int(max_batch_size)
    token_cap = envs.VLLM_DCP_A2A_MAX_TOKENS
    if token_cap > 0:
        # Deliberate hybrid dispatch: batches above the cap take a pipelined
        # NCCL collective instead, and the staging pool shrinks to the cap.
        if batch > token_cap:
            return None
        max_batch_size = min(max_batch_size, token_cap)
    if max_batch_size < 1:
        return None
    if query_head_dim is None:
        query_head_dim = head_dim
    query_head_dim = int(query_head_dim)
    if query_head_dim <= 0 or query_head_dim % 8 != 0:
        return None

    pool = _get_b12x_dcp_a2a_pool(
        cp_group,
        device=cp_attn_out.device,
        total_heads=total_heads,
        head_dim=head_dim,
        query_head_dim=query_head_dim,
        max_batch_size=max_batch_size,
    )
    if pool is None:
        return None

    if batch > max_batch_size:
        logger.warning_once(
            "B12X PCIe DCP A2A received batch=%d beyond its configured "
            "max_batch_size=%d; falling back to NCCL.",
            batch,
            max_batch_size,
        )
        return None

    # The channel accepts packed token-major input and the capacity-strided
    # head-major layout produced by B12X sparse MLA. Preserve either layout;
    # only legacy padded-head slices need materialization.
    if not _is_supported_bhd_layout(cp_attn_out):
        cp_attn_out = cp_attn_out.contiguous()
    if not cp_attn_lse.is_contiguous():
        cp_attn_lse = cp_attn_lse.contiguous()

    reduced_storage = torch.empty(
        (total_heads // world_size, batch, head_dim),
        device=cp_attn_out.device,
        dtype=cp_attn_out.dtype,
    )
    reduced = reduced_storage.transpose(0, 1)
    return pool.lse_reduce_scatter(
        cp_attn_out,
        cp_attn_lse,
        out=reduced,
        is_lse_base_on_e=is_lse_base_on_e,
        channel_id=_B12X_DCP_EAGER_CHANNEL_ID,
    )


def _try_b12x_dcp_all_gather_heads(
    local_input: torch.Tensor,
    cp_group: GroupCoordinator,
    *,
    max_batch_size: int | None,
    output_head_dim: int | None,
) -> torch.Tensor | None:
    """Gather rank-local query heads with the B12X PCIe channel."""
    world_size = cp_group.world_size
    if (
        not local_input.is_cuda
        or local_input.dtype not in (torch.float16, torch.bfloat16)
        or world_size not in (2, 4, 8)
        or local_input.ndim != 3
        or not local_input.is_contiguous()
    ):
        return None

    batch, local_heads, head_dim = local_input.shape
    if local_heads <= 0 or head_dim % 8 != 0:
        return None
    if max_batch_size is None:
        max_batch_size = batch
    max_batch_size = int(max_batch_size)
    token_cap = envs.VLLM_DCP_A2A_MAX_TOKENS
    if token_cap > 0:
        if batch > token_cap:
            return None
        max_batch_size = min(max_batch_size, token_cap)
    if max_batch_size < 1 or batch > max_batch_size:
        return None
    if output_head_dim is None:
        output_head_dim = head_dim
    output_head_dim = int(output_head_dim)
    if output_head_dim <= 0 or output_head_dim % 8 != 0:
        return None

    pool = _get_b12x_dcp_a2a_pool(
        cp_group,
        device=local_input.device,
        total_heads=local_heads * world_size,
        head_dim=output_head_dim,
        query_head_dim=head_dim,
        max_batch_size=max_batch_size,
    )
    if pool is None:
        return None
    return pool.all_gather_heads(
        local_input,
        channel_id=_B12X_DCP_EAGER_CHANNEL_ID,
    )


def dcp_b12x_all_gather_heads(
    local_input: torch.Tensor,
    cp_group: GroupCoordinator,
    *,
    max_batch_size: int | None = None,
    output_head_dim: int | None = None,
) -> torch.Tensor:
    """Gather query heads with B12X, falling back to the group backend."""
    local_input = local_input.contiguous()
    if envs.VLLM_USE_B12X_DCP_A2A:
        result = _try_b12x_dcp_all_gather_heads(
            local_input,
            cp_group,
            max_batch_size=max_batch_size,
            output_head_dim=output_head_dim,
        )
        if result is not None:
            return result
    return cp_group.all_gather(local_input, dim=1)


def warmup_b12x_dcp_a2a(
    cp_group: GroupCoordinator,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_batch_size: int,
    total_heads: int,
    head_dim: int,
    query_head_dim: int | None = None,
) -> None:
    """Create and exercise the B12X DCP channel before CUDA graph capture."""
    if not envs.VLLM_USE_B12X_DCP_A2A:
        return
    if cp_group.world_size not in (2, 4, 8):
        # The PCIe channel only exists for these world sizes. The runtime
        # dispatchers already fall back to NCCL collectives per call, so an
        # unsupported DCP size (e.g. TP6 with DCP3/DCP6) must not fail boot.
        logger.warning_once(
            "B12X PCIe DCP collectives support world sizes 2/4/8; "
            "DCP world size %d uses NCCL collectives instead.",
            cp_group.world_size,
        )
        return
    if query_head_dim is None:
        query_head_dim = head_dim
    local_query = torch.empty(
        (1, total_heads // cp_group.world_size, query_head_dim),
        device=device,
        dtype=dtype,
    )
    gathered_query = _try_b12x_dcp_all_gather_heads(
        local_query,
        cp_group,
        max_batch_size=max_batch_size,
        output_head_dim=head_dim,
    )
    if gathered_query is None:
        raise RuntimeError(
            "B12X PCIe DCP query all-gather is unavailable for the configured "
            "attention geometry"
        )
    partial_output = torch.empty(
        (1, total_heads, head_dim),
        device=device,
        dtype=dtype,
    )
    partial_lse = torch.zeros(
        (1, total_heads),
        device=device,
        dtype=torch.float32,
    )
    result = _try_b12x_dcp_lse_reduce(
        partial_output,
        partial_lse,
        cp_group,
        return_lse=False,
        is_lse_base_on_e=True,
        max_batch_size=max_batch_size,
        query_head_dim=query_head_dim,
    )
    if result is None:
        raise RuntimeError(
            "B12X PCIe DCP output reduction is unavailable for the configured "
            "attention geometry"
        )


# Standard A2A implementation


def _lse_weighted_combine(
    outputs: torch.Tensor,
    lses: torch.Tensor,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    CPU reference implementation for LSE-weighted combination.

    This is a pure PyTorch implementation used for testing and validation.

    Args:
        outputs: Partial attention outputs [N, B, H, D]
                 N = number of KV shards (ranks)
                 B = batch size (num_tokens)
                 H = number of heads per rank
                 D = head dimension
        lses: Log-sum-exp values [N, B, H]
        return_lse: If True, also return the global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2

    Returns:
        Combined output [B, H, D], and optionally global LSE [B, H]
    """
    N, B, H, D = outputs.shape

    # Handle NaN and inf in LSEs
    lses = torch.where(
        torch.isnan(lses) | torch.isinf(lses),
        torch.tensor(float("-inf"), device=lses.device, dtype=lses.dtype),
        lses,
    )

    # Compute max LSE for numerical stability
    lse_max, _ = lses.max(dim=0)  # [B, H]
    lse_max = torch.where(
        lse_max == float("-inf"),
        torch.zeros_like(lse_max),
        lse_max,
    )

    # Compute weights: softmax over the N dimension
    if is_lse_base_on_e:
        weights = torch.exp(lses - lse_max.unsqueeze(0))  # [N, B, H]
    else:
        weights = torch.pow(2.0, lses - lse_max.unsqueeze(0))  # [N, B, H]

    # Handle NaN weights
    weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)

    # Normalize weights
    weight_sum = weights.sum(dim=0, keepdim=True)  # [1, B, H]
    weights = weights / weight_sum.clamp(min=1e-10)  # [N, B, H]

    # Weighted combination: sum over N dimension
    weights = weights.unsqueeze(-1)
    outputs = torch.where(weights == 0, torch.zeros_like(outputs), outputs)
    result = (outputs * weights).sum(dim=0)  # [B, H, D]

    if return_lse:
        if is_lse_base_on_e:
            global_lse = torch.log(weight_sum.squeeze(0)) + lse_max  # [B, H]
        else:
            global_lse = torch.log2(weight_sum.squeeze(0)) + lse_max  # [B, H]
        return result, global_lse

    return result


def _dcp_a2a_lse_pack_dim(output_dtype: torch.dtype) -> int:
    bits = torch.finfo(output_dtype).bits
    if bits == 16:
        return 2
    if bits == 32:
        return 1
    raise ValueError(f"Cannot pack fp32 LSE into output dtype {output_dtype}.")


def _validate_dcp_valid_counts(
    valid_counts: torch.Tensor | None,
    cp_attn_out: torch.Tensor,
) -> None:
    if valid_counts is None:
        return
    expected_shape = (cp_attn_out.shape[0],)
    if valid_counts.shape != expected_shape:
        raise ValueError(
            f"valid_counts must have shape {expected_shape}, got "
            f"{tuple(valid_counts.shape)}."
        )
    if valid_counts.dtype != torch.int32:
        raise TypeError(
            f"valid_counts must have dtype torch.int32, got {valid_counts.dtype}."
        )
    if valid_counts.device != cp_attn_out.device:
        raise ValueError(
            "valid_counts and attention output must be on the same device, got "
            f"{valid_counts.device} and {cp_attn_out.device}."
        )


@triton.jit
def _sanitize_dcp_empty_rows_kernel(
    out_ptr,
    lse_ptr,
    valid_counts_ptr,
    out_stride_B,
    out_stride_H,
    out_stride_D,
    lse_stride_B,
    lse_stride_H,
    valid_counts_stride_B,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < HEAD_DIM
    row_is_valid = tl.load(valid_counts_ptr + batch_idx * valid_counts_stride_B) > 0
    out_offsets = (
        batch_idx * out_stride_B + head_idx * out_stride_H + d_offsets * out_stride_D
    )
    values = tl.load(out_ptr + out_offsets, mask=d_mask)
    tl.store(out_ptr + out_offsets, tl.where(row_is_valid, values, 0.0), mask=d_mask)
    lse_offset = batch_idx * lse_stride_B + head_idx * lse_stride_H
    lse = tl.load(lse_ptr + lse_offset)
    tl.store(lse_ptr + lse_offset, tl.where(row_is_valid, lse, -float("inf")))


def sanitize_dcp_attn_empty_rows(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    valid_counts: torch.Tensor | None,
) -> None:
    """Force locally empty DCP rows to the neutral ``(0, -inf)`` state."""
    _validate_dcp_valid_counts(valid_counts, cp_attn_out)
    if valid_counts is None or cp_attn_out.shape[0] == 0:
        return
    if cp_attn_out.ndim != 3 or cp_attn_lse.shape != cp_attn_out.shape[:2]:
        raise ValueError(
            "Expected attention output [B,H,D] and LSE [B,H], got "
            f"{tuple(cp_attn_out.shape)} and {tuple(cp_attn_lse.shape)}."
        )
    head_dim = cp_attn_out.shape[2]
    grid = (cp_attn_out.shape[0], cp_attn_out.shape[1], 1)
    _sanitize_dcp_empty_rows_kernel[grid](
        cp_attn_out,
        cp_attn_lse,
        valid_counts,
        cp_attn_out.stride(0),
        cp_attn_out.stride(1),
        cp_attn_out.stride(2),
        cp_attn_lse.stride(0),
        cp_attn_lse.stride(1),
        valid_counts.stride(0),
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )


def _dcp_a2a_send_recv_buffers(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Don't use the shared WorkspaceManager here. A FULL cudagraph bakes in the
    # buffer address at capture, but a larger eager batch can grow that workspace
    # and free the captured address. Eager calls therefore use ordinary temporary
    # tensors, while captured calls retain fixed-size owners below.
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        # FULL graphs share a global graph pool. Without a live Python owner,
        # a larger descriptor's staging allocation can be recycled while a
        # smaller descriptor is captured, leaving both NCCL graph nodes bound
        # to the same address. Keep one exact-shape pair alive per device so
        # descriptors cannot alias each other's A2A staging storage. Layers of
        # one graph may reuse the pair because all operations are stream-ordered.
        key = (shape, device, dtype)
        buffers = _DCP_A2A_GRAPH_BUFFERS.get(key)
        if buffers is None:
            buffers = (
                torch.empty(shape, device=device, dtype=dtype),
                torch.empty(shape, device=device, dtype=dtype),
            )
            _DCP_A2A_GRAPH_BUFFERS[key] = buffers
        return buffers

    return (
        torch.empty(shape, device=device, dtype=dtype),
        torch.empty(shape, device=device, dtype=dtype),
    )


@triton.jit
def _dcp_a2a_pack_send_kernel(
    out_ptr,
    lse_ptr,
    send_ptr,
    out_stride_B,
    out_stride_H,
    out_stride_D,
    lse_stride_B,
    lse_stride_H,
    send_stride_N,
    send_stride_B,
    send_stride_H,
    send_stride_D,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    H_PER_RANK: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    local_head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)

    for rank_idx in tl.static_range(N):
        src_head_idx = rank_idx * H_PER_RANK + local_head_idx
        send_base = (
            rank_idx * send_stride_N
            + batch_idx * send_stride_B
            + local_head_idx * send_stride_H
        )

        out_offsets = (
            batch_idx * out_stride_B
            + src_head_idx * out_stride_H
            + d_offsets * out_stride_D
        )
        tl.store(
            send_ptr + send_base + d_offsets * send_stride_D,
            tl.load(out_ptr + out_offsets),
        )

        lse_val = tl.load(
            lse_ptr + batch_idx * lse_stride_B + src_head_idx * lse_stride_H
        ).to(tl.float32)
        if LSE_PACK_DIM == 1:
            tl.store(
                send_ptr + send_base + HEAD_DIM * send_stride_D,
                lse_val.to(send_ptr.dtype.element_ty),
            )
        else:
            lse_bits = lse_val.to(tl.uint32, bitcast=True)
            lo = (lse_bits & 0xFFFF).to(tl.uint16)
            hi = ((lse_bits >> 16) & 0xFFFF).to(tl.uint16)
            tl.store(
                send_ptr + send_base + HEAD_DIM * send_stride_D,
                lo.to(send_ptr.dtype.element_ty, bitcast=True),
            )
            tl.store(
                send_ptr + send_base + (HEAD_DIM + 1) * send_stride_D,
                hi.to(send_ptr.dtype.element_ty, bitcast=True),
            )


@triton.jit
def _dcp_a2a_unpack_combine_kernel(
    recv_ptr,
    out_ptr,
    out_lse_ptr,
    recv_stride_N,
    recv_stride_B,
    recv_stride_H,
    recv_stride_D,
    out_stride_B,
    out_stride_H,
    out_stride_D,
    out_lse_stride_B,
    out_lse_stride_H,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)

    lse_max = -float("inf")
    for rank_idx in tl.static_range(N):
        recv_base = (
            rank_idx * recv_stride_N
            + batch_idx * recv_stride_B
            + head_idx * recv_stride_H
        )
        if LSE_PACK_DIM == 1:
            lse_val = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D).to(
                tl.float32
            )
        else:
            lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
            hi_raw = tl.load(recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D)
            lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        lse_max = tl.maximum(lse_max, lse_val)

    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)

    lse_sum = 0.0
    for rank_idx in tl.static_range(N):
        recv_base = (
            rank_idx * recv_stride_N
            + batch_idx * recv_stride_B
            + head_idx * recv_stride_H
        )
        if LSE_PACK_DIM == 1:
            lse_val = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D).to(
                tl.float32
            )
        else:
            lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
            hi_raw = tl.load(recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D)
            lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        if IS_BASE_E:
            lse_sum += tl.exp(lse_val - lse_max)
        else:
            lse_sum += tl.exp2(lse_val - lse_max)

    if IS_BASE_E:  # noqa: SIM108
        global_lse = tl.log(lse_sum) + lse_max
    else:
        global_lse = tl.log2(lse_sum) + lse_max

    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for rank_idx in tl.static_range(N):
        recv_base = (
            rank_idx * recv_stride_N
            + batch_idx * recv_stride_B
            + head_idx * recv_stride_H
        )
        if LSE_PACK_DIM == 1:
            lse_val = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D).to(
                tl.float32
            )
        else:
            lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
            hi_raw = tl.load(recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D)
            lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        if IS_BASE_E:
            weight = tl.exp(lse_val - global_lse)
        else:
            weight = tl.exp2(lse_val - global_lse)
        weight = tl.where(weight != weight, 0.0, weight)
        partial = tl.load(recv_ptr + recv_base + d_offsets * recv_stride_D).to(
            tl.float32
        )
        partial = tl.where(weight == 0.0, 0.0, partial)
        acc += partial * weight

    final_offsets = (
        batch_idx * out_stride_B + head_idx * out_stride_H + d_offsets * out_stride_D
    )
    tl.store(out_ptr + final_offsets, acc)

    if RETURN_LSE:
        out_lse_offset = batch_idx * out_lse_stride_B + head_idx * out_lse_stride_H
        tl.store(out_lse_ptr + out_lse_offset, global_lse)


def _dcp_a2a_pack_send(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    send_buffer: torch.Tensor,
    world_size: int,
    h_per_rank: int,
    head_dim: int,
    lse_pack_dim: int,
    seq_lens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
) -> None:
    mask_dcp_empty_shards_(cp_attn_lse, seq_lens, query_start_loc)
    grid = (cp_attn_out.shape[0], h_per_rank, 1)
    _dcp_a2a_pack_send_kernel[grid](
        cp_attn_out,
        cp_attn_lse,
        send_buffer,
        cp_attn_out.stride(0),
        cp_attn_out.stride(1),
        cp_attn_out.stride(2),
        cp_attn_lse.stride(0),
        cp_attn_lse.stride(1),
        send_buffer.stride(0),
        send_buffer.stride(1),
        send_buffer.stride(2),
        send_buffer.stride(3),
        N=world_size,
        HEAD_DIM=head_dim,
        H_PER_RANK=h_per_rank,
        LSE_PACK_DIM=lse_pack_dim,
    )


def _dcp_a2a_unpack_combine(
    recv_buffer: torch.Tensor,
    head_dim: int,
    lse_pack_dim: int,
    return_lse: bool,
    is_lse_base_on_e: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    world_size, num_tokens, h_per_rank, _ = recv_buffer.shape
    out_storage = torch.empty(
        (h_per_rank, num_tokens, head_dim),
        device=recv_buffer.device,
        dtype=recv_buffer.dtype,
    )
    out = out_storage.transpose(0, 1)
    out_lse = torch.empty(
        (num_tokens, h_per_rank) if return_lse else (1, 1),
        device=recv_buffer.device,
        dtype=torch.float32 if return_lse else recv_buffer.dtype,
    )
    grid = (num_tokens, h_per_rank, 1)
    _dcp_a2a_unpack_combine_kernel[grid](
        recv_buffer,
        out,
        out_lse,
        recv_buffer.stride(0),
        recv_buffer.stride(1),
        recv_buffer.stride(2),
        recv_buffer.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out_lse.stride(0),
        out_lse.stride(1),
        N=world_size,
        HEAD_DIM=head_dim,
        IS_BASE_E=is_lse_base_on_e,
        RETURN_LSE=return_lse,
        LSE_PACK_DIM=lse_pack_dim,
    )
    if return_lse:
        return out, out_lse
    return out


def dcp_a2a_lse_reduce(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
    seq_lens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    use_b12x: bool = False,
    b12x_max_batch_size: int | None = None,
    b12x_query_head_dim: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Combine partial attention outputs across DCP ranks using All-to-All.

    The output and LSE are packed into a single output-dtype buffer, sent
    with one All-to-All, then unpacked and combined with exact LSE weighting.

    Args:
        cp_attn_out: [B, H, D] where B=num_tokens, H=total_heads, D=head_dim
        cp_attn_lse: [B, H] floating-point log-sum-exp values
        cp_group: GroupCoordinator for DCP communication
        ctx: CPTritonContext (unused, for signature compatibility)
        return_lse: If True, also return the combined global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2
        seq_lens: Local KV lengths. Empty shards contribute zero weight.
        query_start_loc: Cumulative query-token offsets for each request.
        use_b12x: Try the low-latency B12X PCIe path before NCCL A2A
        b12x_max_batch_size: Configured token capacity for B12X staging
        b12x_query_head_dim: Query width when it differs from output width

    Returns:
        Combined output [B, H/N, D] (head-scattered)
        If return_lse=True, also returns global_lse [B, H/N]
    """
    world_size = cp_group.world_size

    if world_size == 1:
        if return_lse:
            return cp_attn_out, cp_attn_lse
        return cp_attn_out

    if use_b12x and envs.VLLM_USE_B12X_DCP_A2A:
        b12x_result = _try_b12x_dcp_lse_reduce(
            cp_attn_out,
            cp_attn_lse,
            cp_group,
            return_lse=return_lse,
            is_lse_base_on_e=is_lse_base_on_e,
            max_batch_size=b12x_max_batch_size,
            query_head_dim=b12x_query_head_dim,
        )
        if b12x_result is not None:
            return b12x_result

    B, H, D = cp_attn_out.shape
    if H % world_size != 0:
        raise ValueError(f"H={H} must be divisible by DCP world size {world_size}.")
    H_per_rank = H // world_size
    lse_pack_dim = _dcp_a2a_lse_pack_dim(cp_attn_out.dtype)

    send_buffer, recv_buffer = _dcp_a2a_send_recv_buffers(
        (world_size, B, H_per_rank, D + lse_pack_dim),
        device=cp_attn_out.device,
        dtype=cp_attn_out.dtype,
    )

    _dcp_a2a_pack_send(
        cp_attn_out,
        cp_attn_lse,
        send_buffer,
        world_size,
        H_per_rank,
        D,
        lse_pack_dim,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
    )

    work = dist.all_to_all_single(
        recv_buffer.view(-1),
        send_buffer.view(-1),
        group=cp_group.device_group,
        async_op=True,
    )
    work.wait()

    return _dcp_a2a_unpack_combine(
        recv_buffer, D, lse_pack_dim, return_lse, is_lse_base_on_e
    )


def get_dcp_workspace_max_num_tokens(vllm_config: VllmConfig) -> int:
    scheduler_config = vllm_config.scheduler_config
    speculative_config = vllm_config.speculative_config
    speculative_tokens = vllm_config.num_speculative_tokens
    tokens_per_seq = (
        1
        + (
            2
            if speculative_config is not None and speculative_config.parallel_drafting
            else 1
        )
        * speculative_tokens
    )
    return min(
        scheduler_config.max_num_batched_tokens,
        max(
            scheduler_config.max_num_seqs * tokens_per_seq,
            vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        ),
    )


def reserve_query_head_storage(
    query: torch.Tensor, padded_num_heads: int
) -> torch.Tensor:
    """Reserve backing storage for fixed-head decode kernels."""
    assert query.ndim == 3
    assert query.shape[1] <= padded_num_heads
    padded = query.new_empty((query.shape[0], padded_num_heads, query.shape[2]))
    padded.resize_(query.shape)
    padded.copy_(query)
    return padded


# Symmetric-memory A2A implementation


_A2A_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


class DirectDCPA2AWorkspace(DirectCPWorkspace):
    """Persistent symmetric buffers for direct DCP output exchange."""

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_num_tokens: int,
        heads_per_rank: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        num_ubatches: int = 1,
    ) -> None:
        if dtype not in _A2A_SUPPORTED_DTYPES:
            raise ValueError(f"Direct DCP A2A does not support {dtype}")
        if num_ubatches < 1:
            raise ValueError(
                f"Direct DCP A2A requires at least one ubatch slot, got {num_ubatches}"
            )
        super().__init__(group, device, num_ubatches)
        self.max_num_tokens = max_num_tokens
        self.heads_per_rank = heads_per_rank
        self.head_dim = head_dim

        output_shape = (
            num_ubatches,
            2,
            self.world_size,
            max_num_tokens,
            heads_per_rank,
            head_dim,
        )
        lse_shape = (
            num_ubatches,
            2,
            self.world_size,
            max_num_tokens,
            heads_per_rank,
        )
        signal_shape = (num_ubatches, 2, self.world_size)
        self.received_output, self.peer_output_ptrs = self._allocate(
            output_shape, dtype
        )
        self.received_lse, self.peer_lse_ptrs = self._allocate(lse_shape, torch.float32)
        self.received_signal, self.peer_signal_ptrs = self._allocate(
            signal_shape, torch.int32
        )

    def lse_reduce(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        is_lse_base_on_e: bool,
        seq_lens: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ubatch = dbo_current_ubatch_id()
        num_tokens = partial_output.shape[0]
        output = partial_output.new_empty(
            (num_tokens, self.heads_per_rank, self.head_dim)
        )
        torch.ops._C.direct_dcp_a2a_lse_reduce(
            partial_output,
            partial_lse,
            seq_lens,
            query_start_loc,
            self.peer_output_ptrs[ubatch],
            self.peer_lse_ptrs[ubatch],
            self.peer_signal_ptrs[ubatch],
            self.received_output[ubatch],
            self.received_lse[ubatch],
            self.received_signal[ubatch],
            self.epoch[ubatch : ubatch + 1],
            output,
            self.world_size,
            self.rank,
            self.max_num_tokens,
            is_lse_base_on_e,
        )
        return output


@functools.cache
def get_direct_dcp_a2a_workspace(
    group: GroupCoordinator,
    device: torch.device,
    max_num_tokens: int,
    heads_per_rank: int,
    head_dim: int,
    dtype: torch.dtype,
    num_ubatches: int,
) -> DirectDCPA2AWorkspace | None:
    if not direct_cp_enabled(
        group, dtype, envs.VLLM_USE_DIRECT_DCP_A2A, _A2A_SUPPORTED_DTYPES
    ):
        return None
    return DirectDCPA2AWorkspace(
        group.device_group,
        device,
        max_num_tokens,
        heads_per_rank,
        head_dim,
        dtype,
        num_ubatches,
    )


# Q gather

# Symmetric-memory implementation


def _q_gather_layout_supported(
    world_size: int,
    heads_per_rank: int,
    head_dim: int,
    dtype: torch.dtype,
    padded_num_heads: int | None,
) -> bool:
    element_size = torch.empty((), dtype=dtype).element_size()
    gathered_num_heads = world_size * heads_per_rank
    storage_num_heads = (
        gathered_num_heads if padded_num_heads is None else padded_num_heads
    )
    return (
        heads_per_rank * head_dim * element_size % 16 == 0
        and storage_num_heads * head_dim * element_size % 16 == 0
    )


class DirectDCPQGatherWorkspace(DirectCPWorkspace):
    """Publish query shards directly into the consumer-final symmetric buffer.

    The final buffer is reusable after the downstream DCP output combine. That
    combine orders all ranks after attention has consumed the gathered query.
    """

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_num_tokens: int,
        heads_per_rank: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        num_ubatches: int = 1,
        padded_num_heads: int | None = None,
    ) -> None:
        if num_ubatches < 1:
            raise ValueError(
                "Direct DCP q-gather requires at least one ubatch slot, "
                f"got {num_ubatches}"
            )
        if max_num_tokens < 1 or heads_per_rank < 1 or head_dim < 1:
            raise ValueError(
                "Direct DCP q-gather dimensions must be positive, got "
                f"T={max_num_tokens}, H={heads_per_rank}, D={head_dim}"
            )
        gathered_num_heads = group.size() * heads_per_rank
        if not _q_gather_layout_supported(
            group.size(), heads_per_rank, head_dim, dtype, padded_num_heads
        ):
            raise ValueError("Direct DCP q-gather requires 16-byte-aligned query rows.")
        super().__init__(group, device, num_ubatches)
        if self.world_size <= 1:
            raise ValueError("Direct DCP q-gather requires at least two ranks")
        self.max_num_tokens = max_num_tokens
        self.heads_per_rank = heads_per_rank
        self.gathered_num_heads = gathered_num_heads
        self.padded_num_heads = (
            self.gathered_num_heads if padded_num_heads is None else padded_num_heads
        )
        if self.padded_num_heads < self.gathered_num_heads:
            raise ValueError(
                "Direct DCP q-gather padded heads must cover gathered heads: "
                f"{self.padded_num_heads} < {self.gathered_num_heads}"
            )
        self.head_dim = head_dim

        query_shape = (
            num_ubatches,
            max_num_tokens,
            self.padded_num_heads,
            head_dim,
        )
        signal_shape = (num_ubatches, 2, self.world_size)
        self.final_query, _ = self._allocate(query_shape, dtype)
        self.received_signal, _ = self._allocate(signal_shape, torch.int32)
        query_multicast_ptrs = self._multicast_ptrs(self.final_query)
        signal_multicast_ptrs = self._multicast_ptrs(self.received_signal)
        self.multicast_ptrs = list(
            zip(query_multicast_ptrs, signal_multicast_ptrs, strict=True)
        )
        if not all(
            query_ptr and signal_ptr for query_ptr, signal_ptr in self.multicast_ptrs
        ):
            raise RuntimeError(
                "Direct DCP q-gather requires NVLS symmetric-memory multicast."
            )
        self.completion = self.received_signal.new_zeros((num_ubatches, 1))
        torch.accelerator.synchronize()

    def gather(self, local_query: torch.Tensor) -> torch.Tensor:
        ubatch = dbo_current_ubatch_id()
        if not 0 <= ubatch < self.num_ubatches:
            raise ValueError(
                f"DCP q-gather ubatch {ubatch} exceeds {self.num_ubatches} slots"
            )
        if local_query.ndim == 3 and local_query.shape[1] != self.heads_per_rank:
            raise ValueError(
                f"DCP q-gather expected {self.heads_per_rank} local query heads, "
                f"got {local_query.shape[1]}"
            )

        num_tokens = local_query.shape[0]
        output = torch.as_strided(
            self.final_query[ubatch],
            size=(num_tokens, self.gathered_num_heads, self.head_dim),
            stride=(
                self.gathered_num_heads * self.head_dim,
                self.head_dim,
                1,
            ),
        )
        query_multicast_ptr, signal_multicast_ptr = self.multicast_ptrs[ubatch]
        torch.ops._C.direct_dcp_q_gather(
            local_query,
            output,
            self.received_signal[ubatch],
            self.completion[ubatch],
            self.epoch[ubatch : ubatch + 1],
            self.world_size,
            self.rank,
            self.max_num_tokens,
            self.padded_num_heads,
            query_multicast_ptr,
            signal_multicast_ptr,
        )
        return output


@functools.cache
def get_direct_dcp_q_gather_workspace(
    group: GroupCoordinator,
    device: torch.device,
    max_num_tokens: int,
    heads_per_rank: int,
    head_dim: int,
    dtype: torch.dtype,
    num_ubatches: int,
    padded_num_heads: int | None = None,
) -> DirectDCPQGatherWorkspace | None:
    if not direct_cp_multicast_enabled(group, dtype, envs.VLLM_USE_DIRECT_DCP_Q_GATHER):
        return None
    if not _q_gather_layout_supported(
        group.world_size, heads_per_rank, head_dim, dtype, padded_num_heads
    ):
        return None
    return DirectDCPQGatherWorkspace(
        group.device_group,
        device,
        max_num_tokens,
        heads_per_rank,
        head_dim,
        dtype,
        num_ubatches,
        padded_num_heads,
    )


# KV gather

# Symmetric-memory implementation


_KV_GATHER_SUPPORTED_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
)


def _kv_gather_layout_supported(token_dim: int, dtype: torch.dtype) -> bool:
    return token_dim * torch.empty((), dtype=dtype).element_size() % 16 == 0


class DirectDCPKVGatherWorkspace(DirectCPWorkspace):
    """Persistent symmetric buffers for direct DCP KV gather."""

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_gathered_tokens: int,
        token_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        num_ubatches: int = 1,
    ) -> None:
        if dtype not in _KV_GATHER_SUPPORTED_DTYPES:
            raise ValueError(f"Direct DCP kv-gather does not support {dtype}")
        if num_ubatches < 1:
            raise ValueError(
                "Direct DCP kv-gather requires at least one ubatch slot, "
                f"got {num_ubatches}"
            )
        if max_gathered_tokens < 1 or token_dim < 1:
            raise ValueError(
                "Direct DCP kv-gather dimensions must be positive, got "
                f"T={max_gathered_tokens}, D={token_dim}"
            )
        if not _kv_gather_layout_supported(token_dim, dtype):
            raise ValueError("Direct DCP kv-gather requires 16-byte-aligned KV rows.")
        super().__init__(group, device, num_ubatches)
        if self.world_size <= 1:
            raise ValueError("Direct DCP kv-gather requires at least two ranks")
        if max_gathered_tokens % self.world_size != 0:
            raise ValueError(
                "Direct DCP kv-gather capacity must divide evenly across "
                f"ranks: {max_gathered_tokens} % {self.world_size} != 0"
            )
        self.max_gathered_tokens = max_gathered_tokens

        kv_shape = (num_ubatches, 2, max_gathered_tokens, token_dim)
        signal_shape = (num_ubatches, 2, self.world_size)
        self.received_kv, _ = self._allocate(kv_shape, dtype)
        self.received_signal, _ = self._allocate(signal_shape, torch.int32)
        kv_multicast_ptrs = self._multicast_ptrs(self.received_kv)
        signal_multicast_ptrs = self._multicast_ptrs(self.received_signal)
        self.multicast_ptrs = list(
            zip(kv_multicast_ptrs, signal_multicast_ptrs, strict=True)
        )
        if not all(kv_ptr and signal_ptr for kv_ptr, signal_ptr in self.multicast_ptrs):
            raise RuntimeError(
                "Direct DCP kv-gather requires NVLS symmetric-memory multicast."
            )
        self.completion = self.received_signal.new_zeros((num_ubatches, 2))
        torch.accelerator.synchronize()

    def gather(self, gathered_kv: torch.Tensor, local_kv: torch.Tensor) -> None:
        ubatch = dbo_current_ubatch_id()
        if not 0 <= ubatch < self.num_ubatches:
            raise ValueError(
                f"DCP kv-gather ubatch {ubatch} exceeds {self.num_ubatches} slots"
            )
        kv_multicast_ptr, signal_multicast_ptr = self.multicast_ptrs[ubatch]
        torch.ops._C.direct_dcp_kv_gather(
            local_kv,
            self.received_kv[ubatch],
            self.received_signal[ubatch],
            self.completion[ubatch],
            self.epoch[ubatch : ubatch + 1],
            gathered_kv,
            self.world_size,
            self.rank,
            self.max_gathered_tokens,
            kv_multicast_ptr,
            signal_multicast_ptr,
        )


@functools.cache
def get_direct_dcp_kv_gather_workspace(
    group: GroupCoordinator,
    device: torch.device,
    max_gathered_tokens: int,
    token_dim: int,
    dtype: torch.dtype,
    num_ubatches: int,
) -> DirectDCPKVGatherWorkspace | None:
    if not direct_cp_multicast_enabled(
        group,
        dtype,
        envs.VLLM_USE_DIRECT_DCP_KV_GATHER,
        _KV_GATHER_SUPPORTED_DTYPES,
    ):
        return None
    if not _kv_gather_layout_supported(token_dim, dtype):
        return None
    return DirectDCPKVGatherWorkspace(
        group.device_group,
        device,
        max_gathered_tokens,
        token_dim,
        dtype,
        num_ubatches,
    )


# MLA DCP backend selection


class DCPCombine(Protocol):
    def __call__(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        *,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> torch.Tensor: ...


class MLADCPManager:
    """Select and own layer-level collective implementations for MLA DCP."""

    _kv_gather: Callable[[torch.Tensor, torch.Tensor], object]

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        num_heads: int,
        query_head_dim: int,
        output_head_dim: int,
        query_dtype: torch.dtype,
        output_dtype: torch.dtype,
        padded_num_heads: int | None,
        is_lse_base_on_e: bool,
        use_pcp: bool,
    ) -> None:
        parallel_config = vllm_config.parallel_config
        self.group = get_dcp_group()
        self.device = torch.device(device)
        self.num_ubatches = max(parallel_config.num_ubatches, 1)
        self.max_num_tokens = get_dcp_workspace_max_num_tokens(vllm_config)
        self.use_a2a = parallel_config.dcp_comm_backend == "a2a"
        self.padded_num_heads = padded_num_heads

        self.combine = self._init_combine(
            num_heads,
            output_head_dim,
            output_dtype,
            is_lse_base_on_e,
            use_pcp,
        )
        self.query_gather = (
            None
            if use_pcp
            else self._init_query_gather(
                num_heads,
                query_head_dim,
                query_dtype,
            )
        )

    def _init_combine(
        self,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        is_lse_base_on_e: bool,
        use_pcp: bool,
    ) -> DCPCombine:
        direct_workspace = None
        if self.use_a2a:
            direct_workspace = get_direct_dcp_a2a_workspace(
                self.group,
                self.device,
                self.max_num_tokens,
                num_heads,
                head_dim,
                dtype,
                self.num_ubatches,
            )
        if direct_workspace is not None:
            logger.info_once("Using direct symmetric-memory DCP A2A for MLA.")
            return functools.partial(
                direct_workspace.lse_reduce,
                is_lse_base_on_e=is_lse_base_on_e,
            )

        combine_fn = (
            dcp_a2a_lse_reduce
            if self.use_a2a
            else cp_lse_ag_out_ar
            if use_pcp
            else cp_lse_ag_out_rs
        )
        return functools.partial(
            combine_fn,
            cp_group=self.group,
            is_lse_base_on_e=is_lse_base_on_e,
        )

    def _init_query_gather(
        self,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        direct_workspace = get_direct_dcp_q_gather_workspace(
            self.group,
            self.device,
            self.max_num_tokens,
            num_heads,
            head_dim,
            dtype,
            self.num_ubatches,
            self.padded_num_heads,
        )
        if direct_workspace is not None:
            logger.info_once("Using direct symmetric-memory DCP query gather for MLA.")
            return direct_workspace.gather
        return self._gather_query

    def _gather_query(self, query: torch.Tensor) -> torch.Tensor:
        query = self.group.all_gather(query, dim=1)
        if self.padded_num_heads is not None:
            query = reserve_query_head_storage(query, self.padded_num_heads)
        return query

    def init_kv_gather(
        self,
        workspace: torch.Tensor,
        max_gathered_tokens: int,
    ) -> None:
        world_size = self.group.world_size
        assert max_gathered_tokens > 0
        assert max_gathered_tokens % world_size == 0
        assert workspace.ndim == 2
        assert workspace.is_contiguous()
        assert workspace.shape[0] == (
            max_gathered_tokens + max_gathered_tokens // world_size
        )
        assert workspace.shape[1] > 0

        direct_workspace = get_direct_dcp_kv_gather_workspace(
            self.group,
            workspace.device,
            max_gathered_tokens,
            workspace.shape[1],
            workspace.dtype,
            self.num_ubatches,
        )
        if direct_workspace is not None:
            logger.info_once(
                "Using direct symmetric-memory DCP chunked-context KV gather for MLA."
            )
            self._kv_gather = direct_workspace.gather
        else:
            self._kv_gather = functools.partial(
                torch.distributed.all_gather_into_tensor,
                group=self.group.device_group,
            )

    def kv_gather(
        self,
        gathered_kv: torch.Tensor,
        local_kv: torch.Tensor,
    ) -> object:
        return self._kv_gather(gathered_kv, local_kv)

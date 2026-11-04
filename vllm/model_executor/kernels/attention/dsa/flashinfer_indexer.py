# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer's fused FP8 MQA-logits + exact top-k indexer (SM120 family).

``flashinfer.dsa_indexer.dsa_indexer_det_topk(..., backend="sm120")`` scores
bit-identically to DeepGEMM's ``fp8_fp4_mqa_logits`` /
``fp8_fp4_paged_mqa_logits`` on SM120 and selects each row's exact top-k
without materializing the logits for prefill.
"""

import functools
import importlib.util

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

logger = init_logger(__name__)

FLASHINFER_INDEXER_HEADS = (32, 64)
FLASHINFER_INDEXER_HEAD_DIM = 128
FLASHINFER_INDEXER_MAX_TOPK = 4096
FLASHINFER_INDEXER_MAX_NEXT_N = 16


@functools.cache
def flashinfer_indexer_available() -> bool:
    """True when FlashInfer ships the SM120 DSA indexer and the GPU is SM120-family."""
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and has_flashinfer()
        and importlib.util.find_spec("flashinfer.experimental.sm120_dsa_indexer")
        is not None
    )


def use_flashinfer_indexer(
    *,
    query_shape: tuple[int, ...],
    topk: int,
    fp4_cache: bool,
    full_context: bool,
    candidate_selection: bool,
) -> bool:
    """Whether an indexer layer ranks its rows with FlashInfer's fused kernel.

    query_shape: one token's indexer queries, (heads, head_dim);
    fp4_cache: the MXFP4 indexer cache keeps the DeepGEMM path;
    full_context: every rank ranks its rows over the whole context (no DCP or
    PCP); candidate_selection: v4.1 two-level selection needs the logits.
    Disabled with VLLM_USE_FUSED_MQA_TOPK=0, like the SM100 fused kernel.
    """
    heads, head_dim = tuple(query_shape)
    return (
        envs.VLLM_USE_FUSED_MQA_TOPK
        and not fp4_cache
        and full_context
        and not candidate_selection
        and heads in FLASHINFER_INDEXER_HEADS
        and head_dim == FLASHINFER_INDEXER_HEAD_DIM
        and 0 < topk <= FLASHINFER_INDEXER_MAX_TOPK
        and flashinfer_indexer_available()
    )


def _contiguous_output(out: torch.Tensor) -> torch.Tensor:
    return out if out.is_contiguous() else torch.empty_like(out)


def flashinfer_indexer_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_offsets: torch.Tensor,
    compress_ratio: int,
    max_seqlen_k: int,
    output: torch.Tensor,
) -> None:
    """Fused replacement of ``fp8_fp4_mqa_logits`` + ``top_k_per_row_prefill``.

    q: [rows, H, 128] fp8; k: [keys, 128] fp8 holding the requests' keys back
    to back as ``cu_seqlens_k`` lists them; k_scale: [keys] fp32; weights:
    [rows, H] fp32; cu_seqlens_q: int32 [reqs + 1] the requests' rows;
    q_offsets: int64 [reqs] the context position before each request's first
    row: row ``u`` of request ``s`` sees its first
    ``(q_offsets[s] + u + 1) // compress_ratio`` keys. output: [rows, k] int32,
    written like ``top_k_per_row_prefill``: request-local key ids, ``-1``
    padded.
    """
    from flashinfer.dsa_indexer import dsa_indexer_det_topk

    logger.info_once(
        "Using FlashInfer's fused SM120 indexer for prefill (top-k %d).",
        output.shape[1],
    )
    indices = _contiguous_output(output)
    dsa_indexer_det_topk(
        q,
        k,
        weights,
        cu_seqlens_q,
        cu_seqlens_k,
        top_k=output.shape[1],
        q_causal_offsets=q_offsets,
        ratio=compress_ratio,
        max_seqlen_k=max_seqlen_k,
        indices=indices,
        return_scores=False,
        backend="sm120",
        k_scale=k_scale,
    )
    if indices is not output:
        output.copy_(indices)


def flashinfer_indexer_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Fused replacement of ``fp8_fp4_paged_mqa_logits`` + the decode top-k.

    q: [B, next_n, H, 128] fp8 (next_n <= 16); kv_cache: the DeepGEMM-layout
    indexer cache [num_blocks, block_size, 1, 132] uint8; weights:
    [B * next_n, H] fp32; context_lens: int32 [B, next_n]; output:
    [B * next_n, k] int32 request-local positions, ``-1`` padded. The launches
    depend only on the shapes, so a captured CUDA graph replays for any
    lengths.
    """
    from flashinfer.dsa_indexer import dsa_indexer_det_topk

    logger.info_once(
        "Using FlashInfer's fused SM120 indexer for decode (top-k %d).",
        output.shape[1],
    )
    indices = _contiguous_output(output)
    dsa_indexer_det_topk(
        q,
        kv_cache,
        weights,
        None,
        None,
        top_k=output.shape[1],
        indices=indices,
        return_scores=False,
        backend="sm120",
        block_table=block_table,
        context_lens=context_lens,
    )
    if indices is not output:
        output.copy_(indices)

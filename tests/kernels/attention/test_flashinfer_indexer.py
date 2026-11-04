# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer's fused SM120 indexer vs the DeepGEMM logits + top-k path,
through ``sparse_attn_indexer``."""

from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
import vllm.model_executor.layers.sparse_attn_indexer as indexer_layer
from vllm import _custom_ops as ops
from vllm.config import CUDAGraphMode
from vllm.model_executor.kernels.attention.dsa.flashinfer_indexer import (
    flashinfer_indexer_available,
)
from vllm.utils.deep_gemm import get_num_sms, get_paged_mqa_logits_metadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerPrefillMetadata,
    build_prefill_chunk_metadata,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not flashinfer_indexer_available(),
    reason="Requires an SM120 GPU and FlashInfer's SM120 DSA indexer",
)

HEADS = 64
TOPK = 2048
BLOCK_SIZE = 64


def _run(monkeypatch, metadata, q, weights, kv_cache, rows, fused, workspace=None):
    monkeypatch.setattr(
        indexer_layer,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={"indexer": metadata},
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        ),
    )
    if workspace is not None:
        monkeypatch.setattr(
            indexer_layer,
            "current_workspace_manager",
            lambda: SimpleNamespace(get_simultaneous=lambda *specs: workspace),
        )
    monkeypatch.setattr(envs, "VLLM_USE_FUSED_MQA_TOPK", fused)
    out = torch.full((rows, TOPK), -1, device="cuda", dtype=torch.int32)
    indexer_layer.sparse_attn_indexer(
        hidden_states=torch.empty(rows, 1, device="cuda"),
        k_cache_prefix="indexer",
        kv_cache=kv_cache,
        q_quant=q,
        q_scale=None,
        k=None,
        weights=weights,
        quant_block_size=128,
        scale_fmt="ue8m0",
        topk_tokens=TOPK,
        head_dim=128,
        max_model_len=1 << 16,
        total_seq_lens=kv_cache.shape[0] if workspace is None else 0,
        topk_indices_buffer=out,
        skip_k_cache_insert=True,
        use_pcp=False,
        pcp_shard_decode_requests=False,
        dense_mha_metadata_layer_name="",
    )
    return out


def _scores(ids, q, k, k_scale, weights):
    """fp32 scores of the selected keys (row-local ids into ``k``)."""
    out = torch.full(ids.shape, -torch.inf, device="cuda")
    for i in range(0, ids.shape[0], 32):
        sel = ids[i : i + 32]
        keys = k[sel.clamp(min=0).long()].float()  # [r, topk, 128]
        logits = torch.einsum("rhd,rkd->rhk", q[i : i + 32].float(), keys)
        s = (weights[i : i + 32, :, None] * logits.relu()).sum(1)
        s = s * k_scale[sel.clamp(min=0).long()]
        out[i : i + 32] = s.masked_fill(sel < 0, -torch.inf)
    return out


def _assert_same_topk(fused, dense, scores_fn):
    """Same valid counts and the same selected scores (ties may pick other
    keys)."""
    assert ((fused >= 0).sum(1) == (dense >= 0).sum(1)).all()
    a = scores_fn(fused).sort(dim=1, descending=True).values
    b = scores_fn(dense).sort(dim=1, descending=True).values
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("compress_ratio", [1, 4])
@pytest.mark.parametrize("window", ["all", "query_slice", "row_shard"])
def test_prefill_matches_dense(monkeypatch, compress_ratio, window):
    torch.manual_seed(0)
    # Context length and new rows per request; the last is a fresh prompt.
    seq_lens = [9000, 30000, 300]
    query_lens = [500, 300, 300]
    keys_per_req = [n // compress_ratio for n in seq_lens]
    i32 = dict(device="cuda", dtype=torch.int32)
    query_start_loc = torch.tensor([0, 500, 800, 1100], **i32)
    query_slice = slice(200, 1000) if window == "query_slice" else None
    chunk = build_prefill_chunk_metadata(
        0,
        3,
        query_start_loc,
        query_start_loc.cpu(),
        torch.tensor(seq_lens, **i32),
        torch.tensor(keys_per_req, **i32),
        torch.tensor(keys_per_req, dtype=torch.int32),
        torch.zeros(3, 1, **i32),
        compress_ratio,
        query_slice=query_slice,
        skip_kv_gather=True,  # the keys are already in the gather workspace
    )
    assert chunk is not None and chunk.req_query_start is not None
    rows = sum(query_lens)
    keys = sum(keys_per_req)
    q = torch.randn(rows, HEADS, 128, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn(keys, 128, device="cuda").to(torch.float8_e4m3fn)
    k_scale = 2.0 ** torch.randint(-3, 3, (keys,), device="cuda").float()
    weights = torch.randn(rows, HEADS, device="cuda") / HEADS
    sizes = [450, rows - 450] if window == "row_shard" else None
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=torch.tensor(seq_lens, **i32),
        max_seq_len=max(seq_lens),
        slot_mapping=torch.zeros(rows, device="cuda", dtype=torch.int64),
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=3,
        num_prefill_tokens=rows,
        prefill=DeepseekV32IndexerPrefillMetadata(
            chunks=[chunk], max_prefill_seq_len=max(seq_lens), row_shard_sizes=sizes
        ),
    )
    if sizes is not None:
        # Rank 1 of two scores rows [450, rows); the exchange needs a TP group.
        monkeypatch.setattr(indexer_layer, "get_tensor_model_parallel_rank", lambda: 1)
        monkeypatch.setattr(
            indexer_layer.PrefillRowShard, "exchange_topk", lambda *a, **kw: None
        )
    workspace = (k, k_scale.view(torch.uint8).view(keys, 4))

    def run(fused):
        out = _run(monkeypatch, metadata, q, weights, k, rows, fused, workspace)
        return out[chunk.token_start : chunk.token_end][
            450 - chunk.token_start if sizes else 0 :
        ]

    fused, dense = run(True), run(False)
    first = 450 if sizes else chunk.token_start
    # Row r of request s starts at its key segment: absolute key ids.
    seg = torch.tensor([0, *torch.tensor(keys_per_req).cumsum(0).tolist()])
    req = (
        torch.searchsorted(
            query_start_loc.cpu(),
            torch.arange(first, first + fused.shape[0]),
            right=True,
        )
        - 1
    )
    base = seg[req].cuda()[:, None]

    def scores(ids):
        absolute = torch.where(ids >= 0, ids + base, ids)
        return _scores(
            absolute, q[first : first + ids.shape[0]], k, k_scale, weights[first:]
        )

    _assert_same_topk(fused, dense, scores)


def _decode_inputs(batch, next_n, max_len=40000, seed=1):
    torch.manual_seed(seed)
    blocks_per_req = (max_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = batch * blocks_per_req
    kv_cache = torch.zeros(
        num_blocks, BLOCK_SIZE, 128 + 4, device="cuda", dtype=torch.uint8
    )
    block_table = (
        torch.randperm(num_blocks, device="cuda", dtype=torch.int32)
        .view(batch, blocks_per_req)
        .contiguous()
    )
    slots = (
        block_table[:, :, None] * BLOCK_SIZE
        + torch.arange(BLOCK_SIZE, device="cuda", dtype=torch.int32)
    ).view(-1)
    k = torch.randn(slots.numel(), 128, device="cuda", dtype=torch.bfloat16)
    ops.indexer_k_quant_and_cache(k, kv_cache, slots.long(), 128, "ue8m0")
    rows = batch * next_n
    q = torch.randn(batch, next_n, HEADS, 128, device="cuda").to(torch.float8_e4m3fn)
    weights = torch.randn(rows, HEADS, device="cuda") / HEADS
    return kv_cache, block_table, q, weights


def _decode_metadata(block_table, lens, next_n):
    i32 = dict(device="cuda", dtype=torch.int32)
    batch = block_table.shape[0]
    seq_lens = (lens[:, None] - next_n + 1 + torch.arange(next_n, **i32)).to(
        torch.int32
    )
    decode = DeepSeekV32IndexerDecodeMetadata(
        block_table=block_table,
        seq_lens=seq_lens,
        decode_lens=torch.full((batch,), next_n, **i32),
        requires_padding=False,
        schedule_metadata=get_paged_mqa_logits_metadata(
            seq_lens, BLOCK_SIZE, get_num_sms()
        ),
    )
    return DeepseekV32IndexerMetadata(
        seq_lens=lens,
        max_seq_len=int(lens.max()),
        slot_mapping=torch.zeros(batch * next_n, device="cuda", dtype=torch.int64),
        num_decodes=batch,
        num_decode_tokens=batch * next_n,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=decode,
    )


def _decode_scores(kv_cache, block_table, q, weights, next_n):
    """Scores of request-local positions, read back from the paged cache."""
    flat = kv_cache.view(kv_cache.shape[0], -1)
    values = flat[:, : BLOCK_SIZE * 128].view(-1, BLOCK_SIZE, 128)
    scales = flat[:, BLOCK_SIZE * 128 :].contiguous().view(torch.float32)

    def scores(ids):
        req = torch.arange(ids.shape[0], device="cuda") // next_n
        pos = ids.clamp(min=0).long()
        blk = block_table[req[:, None], pos // BLOCK_SIZE].long()
        k = values.view(torch.float8_e4m3fn)[blk, pos % BLOCK_SIZE]  # [r, k, 128]
        logits = torch.einsum("rhd,rkd->rhk", q.view(-1, HEADS, 128).float(), k.float())
        s = (weights[:, :, None] * logits.relu()).sum(1) * scales[blk, pos % BLOCK_SIZE]
        return s.masked_fill(ids < 0, -torch.inf)

    return scores


@pytest.mark.parametrize("next_n", [1, 2, 4])
def test_decode_matches_dense(monkeypatch, workspace_init, next_n):
    batch = 4
    kv_cache, block_table, q, weights = _decode_inputs(batch, next_n)
    lens = torch.tensor([1500, 39000, 2048, 20000], device="cuda", dtype=torch.int32)
    metadata = _decode_metadata(block_table, lens, next_n)
    rows = batch * next_n
    q_rows = q.view(rows, HEADS, 128)
    fused = _run(monkeypatch, metadata, q_rows, weights, kv_cache, rows, True)
    dense = _run(monkeypatch, metadata, q_rows, weights, kv_cache, rows, False)
    _assert_same_topk(
        fused, dense, _decode_scores(kv_cache, block_table, q, weights, next_n)
    )


def test_decode_graph_replays_with_new_lengths(monkeypatch):
    """Captured once, the decode path ranks the replayed step's lengths."""
    batch, next_n = 4, 2
    kv_cache, block_table, q, weights = _decode_inputs(batch, next_n)
    lens = torch.tensor([3000, 5000, 7000, 9000], device="cuda", dtype=torch.int32)
    metadata = _decode_metadata(block_table, lens, next_n)
    rows = batch * next_n
    q_rows = q.view(rows, HEADS, 128)
    out = torch.full((rows, TOPK), -1, device="cuda", dtype=torch.int32)

    def step():
        out.copy_(_run(monkeypatch, metadata, q_rows, weights, kv_cache, rows, True))

    step()  # FlashInfer sets up its decode state outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    new_lens = torch.tensor([39000, 100, 2048, 25000], device="cuda", dtype=torch.int32)
    fresh = _decode_metadata(block_table, new_lens, next_n)
    metadata.decode.seq_lens.copy_(fresh.decode.seq_lens)
    graph.replay()
    torch.accelerator.synchronize()
    expected = _run(monkeypatch, fresh, q_rows, weights, kv_cache, rows, True)
    assert torch.equal(out.sort(dim=1).values, expected.sort(dim=1).values)

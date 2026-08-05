# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.v1.attention.backends import flashinfer as flashinfer_backend


def test_decode_cudagraph_wrapper_cache_includes_query_length(monkeypatch):
    builder = object.__new__(flashinfer_backend.FlashInferMetadataBuilder)
    builder._decode_wrappers_cudagraph = {}
    builder.paged_kv_indptr = SimpleNamespace(gpu=torch.empty(9, dtype=torch.int32))
    builder.paged_kv_indices = SimpleNamespace(gpu=torch.empty(16, dtype=torch.int32))
    builder.paged_kv_last_page_len = SimpleNamespace(
        gpu=torch.empty(8, dtype=torch.int32)
    )
    builder.is_kvcache_nvfp4 = False

    wrapper_factory = Mock(side_effect=lambda *args, **kwargs: object())
    monkeypatch.setattr(
        flashinfer_backend, "BatchDecodeWithPagedKVCacheWrapper", wrapper_factory
    )
    monkeypatch.setattr(flashinfer_backend, "get_kv_cache_layout", lambda: "NHD")
    monkeypatch.setattr(builder, "_get_workspace_buffer", lambda: object())

    single_token = builder._get_decode_wrapper(8, True, q_len_per_req=1)
    speculative = builder._get_decode_wrapper(8, True, q_len_per_req=4)

    assert speculative is not single_token
    assert builder._get_decode_wrapper(8, True, q_len_per_req=1) is single_token
    assert wrapper_factory.call_count == 2

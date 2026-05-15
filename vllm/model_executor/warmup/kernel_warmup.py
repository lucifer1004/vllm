# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

# Backends that use the DSv4 sparse-MLA path on SM12x. The cold-JIT IMA we
# fix here is specific to these backends' input-prep kernels.
_DEEPSEEK_V4_SPARSE_MLA_BACKENDS = frozenset(
    {
        "FLASHMLA_SPARSE",
        "DEEPSEEK_SPARSE_SWA",
    }
)

# Mixed prefill+decode shape used to JIT the attention-side input-prep
# kernels (e.g. `_compute_swa_indices_and_lens_kernel`).
_DEEPSEEK_V4_SPARSE_MLA_MIXED_WARMUP_TOKENS = 16

# Single-chunk prefill at the scheduler's max batched-token budget.
# Covers the canonical SM12x serve config (max_num_batched_tokens=8192);
# `_clamp_warmup_tokens` clamps down for smaller caps.
_DEEPSEEK_V4_SPARSE_MLA_PREFILL_WARMUP_TOKENS = 8192

# Slot-mapping kernel `_compute_slot_mapping_kernel` JIT-specializes on
# num_tokens; warm a fan of sizes so the first real decode request never
# hits a fresh JIT compile (which on SM12x can produce non-deterministic
# codegen that writes wrong slot_mapping → KV corruption → IMA).
_DEEPSEEK_V4_SLOT_MAPPING_WARMUP_TOKENS = tuple(range(1, 17)) + (
    32,
    64,
    128,
    256,
    512,
)


def _attention_backend_name(backend: object) -> str | None:
    get_name = getattr(backend, "get_name", None)
    if get_name is None:
        return None
    try:
        return get_name()
    except NotImplementedError:
        return None


def _has_deepseek_v4_sparse_mla_backend(runner: "GPUModelRunner") -> bool:
    for groups in getattr(runner, "attn_groups", []) or ():
        for group in groups:
            name = _attention_backend_name(getattr(group, "backend", None))
            if name in _DEEPSEEK_V4_SPARSE_MLA_BACKENDS:
                return True
    return False


def _clamp_warmup_tokens(num_tokens: int, max_tokens: int) -> int:
    return max(0, min(num_tokens, max_tokens))


def _deepseek_v4_slot_mapping_warmup(runner: "GPUModelRunner") -> None:
    """Warm `block_table.compute_slot_mapping` across a fan of token counts.

    The underlying Triton `_compute_slot_mapping_kernel` JIT-specializes on
    num_tokens; on SM12x with non-deterministic codegen, the first compile
    can produce a kernel that writes wrong slot_mapping → KV cache
    corruption → downstream sparse-MLA reads OOB → IMA at
    `flash_mla_sparse_fwd`. Pre-JIT all the shapes the scheduler will ever
    issue during steady-state decode so the first real request runs the
    cached kernel.
    """
    max_tokens = getattr(runner, "max_num_tokens", 1)
    block_table = runner.input_batch.block_table

    # Snapshot the runner buffers we mutate so warmup never leaks state into
    # the first real request.
    saved_query_start_loc_np = None
    saved_query_start_loc_gpu = None
    if hasattr(runner, "query_start_loc"):
        saved_query_start_loc_np = runner.query_start_loc.np[:2].copy()
        saved_query_start_loc_gpu = runner.query_start_loc.gpu[:2].clone()

    try:
        for requested_tokens in _DEEPSEEK_V4_SLOT_MAPPING_WARMUP_TOKENS:
            num_tokens = _clamp_warmup_tokens(requested_tokens, max_tokens)
            if num_tokens <= 0:
                continue

            positions_source = torch.arange(
                num_tokens, dtype=torch.int64, device=runner.device
            )
            if hasattr(runner, "query_start_loc"):
                runner.query_start_loc.np[0] = 0
                runner.query_start_loc.np[1] = num_tokens
                runner.query_start_loc.copy_to_gpu(2)
                query_start_loc = runner.query_start_loc.gpu[:2]
            else:
                query_start_loc = torch.tensor(
                    [0, num_tokens], dtype=torch.int32, device=runner.device
                )

            if hasattr(runner, "positions"):
                saved_positions = runner.positions[:num_tokens].clone()
                runner.positions[:num_tokens].copy_(positions_source)
                positions = runner.positions[:num_tokens]
            else:
                saved_positions = None
                positions = positions_source

            try:
                block_table.commit_block_table(1)
                block_table.compute_slot_mapping(1, query_start_loc, positions)
            finally:
                if saved_positions is not None:
                    runner.positions[:num_tokens].copy_(saved_positions)
    finally:
        if saved_query_start_loc_np is not None:
            runner.query_start_loc.np[:2] = saved_query_start_loc_np
            assert saved_query_start_loc_gpu is not None
            runner.query_start_loc.gpu[:2].copy_(saved_query_start_loc_gpu)


@torch.inference_mode()
def _deepseek_v4_request_prep_warmup(worker: "Worker") -> None:
    """Pre-JIT the slot-mapping kernel before the first real request."""
    if not envs.VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP:
        return

    runner = worker.model_runner
    if runner.is_pooling_model or not _has_deepseek_v4_sparse_mla_backend(runner):
        return
    if not current_platform.is_cuda_alike():
        return

    logger.info("Warming up DeepSeek V4 request preparation kernels.")
    _deepseek_v4_slot_mapping_warmup(runner)
    torch.accelerator.synchronize()


def _deepseek_v4_sparse_mla_attention_warmup(worker: "Worker") -> None:
    """Warm sparse-MLA attention shapes via `_dummy_run`.

    Runs the model forward with synthetic batches matching the canonical
    serve shapes:
    1. Mixed prefill+decode (16 tokens) — warms shared attention paths
       (`_compute_swa_indices_and_lens_kernel`, indexer KV-insert, ...)
    2. Single max-chunk prefill (8K tokens) — warms prefill-only kernels
       (`_compute_prefill_metadata_kernel`, `_save_partial_states_kernel`,
       `_fused_kv_compress_norm_rope_insert_*`)
    3. 2nd-chunk prefill with prior context (8K curr + 8K prior) — warms
       the alt-shape `_build_prefill_chunk_metadata_kernel` codepath the
       indexer hits when a chunked prefill is not the first chunk.

    Both prefill calls use `create_single_prefill=True` so `_dummy_run`
    builds a single-request batch with `max_query_len == num_tokens`
    (mixed-batch warmup uses a different `max_query_len` and so misses
    these kernel specializations).
    """
    if not envs.VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP:
        return

    runner = worker.model_runner
    if runner.is_pooling_model or not _has_deepseek_v4_sparse_mla_backend(runner):
        return

    max_tokens = worker.scheduler_config.max_num_batched_tokens
    mixed_tokens = _clamp_warmup_tokens(
        _DEEPSEEK_V4_SPARSE_MLA_MIXED_WARMUP_TOKENS, max_tokens
    )
    prefill_tokens = _clamp_warmup_tokens(
        _DEEPSEEK_V4_SPARSE_MLA_PREFILL_WARMUP_TOKENS, max_tokens
    )
    if mixed_tokens <= 0 and prefill_tokens <= 0:
        return

    logger.info(
        "Warming up DeepSeek V4 sparse MLA attention "
        "for mixed tokens=%s and prefill tokens=%s.",
        mixed_tokens,
        prefill_tokens,
    )
    if mixed_tokens > 0:
        runner._dummy_run(
            num_tokens=mixed_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )
    if prefill_tokens > 0:
        runner._dummy_run(
            num_tokens=prefill_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_single_prefill=True,
        )
        # Simulate the second-and-later chunk of a chunked prefill so the
        # alt-shape `_build_prefill_chunk_metadata_kernel` codepath that
        # fires when the indexer sees prior context gets JIT-compiled
        # here, not on the first user request that exceeds
        # `max_num_batched_tokens`.
        runner._dummy_run(
            num_tokens=prefill_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_single_prefill=True,
            profile_seq_lens=prefill_tokens * 2,
        )


def kernel_warmup(worker: "Worker"):
    # DeepSeek V4 sparse-MLA warmup runs first so the slot-mapping kernel
    # and attention input-prep kernels are JIT'd against the runner's
    # pristine state, before DG/FlashInfer warmup mutate it.
    _deepseek_v4_sparse_mla_attention_warmup(worker)
    _deepseek_v4_request_prep_warmup(worker)

    # Deep GEMM warmup
    do_deep_gemm_warmup = (
        envs.VLLM_USE_DEEP_GEMM
        and is_deep_gemm_supported()
        and envs.VLLM_DEEP_GEMM_WARMUP != "skip"
    )
    if do_deep_gemm_warmup:
        model = worker.get_model()
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        deep_gemm_warmup(model, max_tokens)

    enable_flashinfer_autotune = (
        worker.vllm_config.kernel_config.enable_flashinfer_autotune
    )
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if enable_flashinfer_autotune is False:
        logger.info("Skipping FlashInfer autotune because it is disabled.")
    elif has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(worker.model_runner)

    # FlashInfer attention warmup
    # Only warmup if the model has FlashInfer attention groups
    # and is not a pooling model
    def _is_flashinfer_backend(backend):
        try:
            return backend.get_name() == "FLASHINFER"
        except NotImplementedError:
            return False

    if (
        not worker.model_runner.is_pooling_model
        and worker.model_runner.attn_groups
        # NOTE: This should be `any` instead of `all` but other hybrid attention
        # backends don't support this dummy run. Once we remove
        # `build_for_cudagraph_capture`, we can change it to `any`.
        and all(
            _is_flashinfer_backend(group.backend)
            for groups in worker.model_runner.attn_groups
            for group in groups
        )
    ):
        logger.info("Warming up FlashInfer attention.")
        # Warmup with mixed batch containing both prefill and decode tokens
        # This is to warm up both prefill and decode attention kernels
        worker.model_runner._dummy_run(
            num_tokens=16,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )


def flashinfer_autotune(runner: "GPUModelRunner") -> None:
    """
    Autotune FlashInfer operations.
    FlashInfer have many implementations for the same operation,
    autotuning runs benchmarks for each implementation and stores
    the results. The results are cached transparently and
    future calls to FlashInfer will use the best implementation.
    Without autotuning, FlashInfer will rely on heuristics, which may
    be significantly slower.
    """
    import vllm.utils.flashinfer as fi_utils

    with torch.inference_mode(), fi_utils.autotune():
        # Certain FlashInfer kernels (e.g. nvfp4 routed moe) are
        # incompatible with autotuning. This state is used to skip
        # those kernels during the autotuning process.
        fi_utils._is_fi_autotuning = True

        # We skip EPLB here since we don't want to record dummy metrics
        # When autotuning with number of tokens m, flashinfer will autotune
        # operations for all number of tokens up to m.
        # So we only need to run with the max number of tokens.
        runner._dummy_run(
            runner.scheduler_config.max_num_batched_tokens,
            skip_eplb=True,
            is_profile=True,
        )

        fi_utils._is_fi_autotuning = False

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the max_num_reqs gate on the V2 mixed prefill+decode warmup."""

from types import SimpleNamespace

import pytest

from vllm.v1.worker.gpu import warmup
from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup


def _fail(*args, **kwargs):
    raise AssertionError("worker callback must not run when warmup is skipped")


@pytest.mark.parametrize("max_num_reqs", [1, 0])
def test_mixed_warmup_skipped_for_single_seq(max_num_reqs):
    """A mixed prefill+decode step needs >=2 requests; with max_num_reqs < 2
    the warmup must be skipped without touching the worker callbacks."""
    runner = SimpleNamespace(is_pooling_model=False, max_num_reqs=max_num_reqs)

    assert (
        run_mixed_prefill_decode_warmup(
            runner,
            worker_execute_model=_fail,
            worker_sample_tokens=_fail,
            num_tokens=128,
        )
        is False
    )


@pytest.mark.parametrize("fail_warmup", [False, True])
def test_kernel_warmup_restores_uncalibrated_adaptive_manager(monkeypatch, fail_warmup):
    """Startup must warm fixed drafts before calibration and retain its manager."""
    manager = SimpleNamespace(cost_tables=None)
    rejection_sampler = SimpleNamespace(enable_adaptive_verification=True)
    runner = SimpleNamespace(
        adaptive_verification=manager,
        rejection_sampler=rejection_sampler,
    )

    def run_steps(model_runner, execute, sample):
        assert model_runner.adaptive_verification is None
        assert not model_runner.rejection_sampler.enable_adaptive_verification
        if fail_warmup:
            raise RuntimeError("warmup failed")

    monkeypatch.setattr(warmup, "_warmup_kernels", run_steps)
    if fail_warmup:
        with pytest.raises(RuntimeError, match="warmup failed"):
            warmup.warmup_kernels(runner, _fail, _fail)
    else:
        warmup.warmup_kernels(runner, _fail, _fail)
    assert runner.adaptive_verification is manager
    assert manager.cost_tables is None
    assert rejection_sampler.enable_adaptive_verification


@pytest.mark.parametrize("is_last_rank", [False, True])
@pytest.mark.parametrize("has_pp", [False, True])
@pytest.mark.parametrize("enable_jit_warmup", [False, True])
def test_worker_preloads_pp_kernels_before_single_warmup(
    monkeypatch, is_last_rank, has_pp, enable_jit_warmup
):
    """Preload PP bookkeeping before any warmup can post feedback."""
    from contextlib import nullcontext
    from unittest.mock import Mock

    from vllm.config.compilation import CompilationMode
    from vllm.v1.worker import gpu_worker

    events = []
    runner = Mock(
        pp_handler=SimpleNamespace() if has_pp else None,
        is_last_pp_rank=is_last_rank,
        warmup_pp_decode_update=Mock(side_effect=lambda: events.append("pp_preload")),
    )
    worker = gpu_worker.Worker.__new__(gpu_worker.Worker)
    worker.model_runner = runner
    worker.use_v2_model_runner = True
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        kernel_config=SimpleNamespace(enable_jit_warmup=enable_jit_warmup),
    )
    worker.model_config = SimpleNamespace(enforce_eager=False)
    worker._get_cudagraph_capture_context = nullcontext

    class CaptureReached(Exception):
        pass

    runner.capture_model.side_effect = CaptureReached
    monkeypatch.setattr(
        gpu_worker, "kernel_warmup", lambda worker: events.append("kernel_warmup")
    )
    monkeypatch.setattr(
        gpu_worker,
        "warmup_kernels",
        lambda *args: events.append("warmup"),
    )
    with pytest.raises(CaptureReached):
        worker.compile_or_warm_up_model()

    expected = ["pp_preload"] if has_pp and not is_last_rank else []
    expected.append("kernel_warmup")
    if enable_jit_warmup:
        expected.append("warmup")
    assert events == expected

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCIe-safe replacement for torch symmetric memory's group barrier.

torch's ``_SymmetricMemory.barrier(channel)`` synchronizes the group with
CAS exchanges on peer-mapped signal pads. On platforms without native P2P
atomics (``cudaDevP2PAttrNativeAtomicSupported == 0``, e.g. PCIe-only
multi-GPU boxes) those CAS ops are not atomic: barrier tokens get lost or
duplicated under PCIe load and the group wedges permanently.

This module swaps the barrier for a protocol that only needs primitives
such platforms do support:

  * sender:   ``cuStreamWriteValue32`` of the instance's sequence number
              into a **per-instance ring slot** of the receiver's signal
              pad — a plain posted P2P write (with the default preceding
              memory barrier, giving release semantics for prior P2P
              traffic);
  * receiver: ``cuStreamWaitValue32(EQ)`` polling the matching slot of
              its *own* pad — local memory only.

No remote read-modify-write anywhere. Slot index is
``(channel, seq % RING, sender)``, so concurrent same-channel instances
issued from different streams (the pipelined fused ops interleave a
compute and a backend stream) touch different slots and cannot corrupt
each other regardless of device-side execution order; EQ matching makes
a slot's stale previous-lap value simply not match. This also removes the
multi-stream instance-interleaving hazard of the CAS design
(pytorch/pytorch#189228). Ring-slot reuse is safe for ``RING >= 2``: a
rank can only issue instance ``K+RING`` after completing ``K+RING-1``,
which requires every peer to have written ``K+RING-1`` and therefore to
have passed its wait for instance ``K``.

The patch is process-global and a drop-in barrier for every handle; the
pipelined fused ops (``fused_matmul_reduce_scatter``,
``fused_all_gather_matmul`` and their scaled variants) synchronize
exclusively through ``barrier(channel)`` plus stream-ordered plain
writes, so patching the barrier makes the whole family PCIe-safe.

Known constraints:

  * **No CUDA graph capture.** The sequence number is baked into the
    recorded write, so a replayed barrier would trivially satisfy its own
    wait and synchronize nothing (the stock CAS barrier replays
    correctly). Calling the patched barrier during capture raises. The
    fused ops only run at sequence-parallel sizes, which sit above vLLM's
    cudagraph capture ceiling, so this path is structurally unreachable
    today.
  * **Sequence state is keyed by the handle's signal-pad address.** If an
    allocation is freed and a new symm-mem handle reuses the same VA with
    a zeroed pad, call :func:`reset_pcie_barrier_state`. torch's fused
    ops DO re-allocate their workspace when an op requests more space —
    unsafely (see the workspace-guard notes below); the installed guard
    floors the first allocation and keeps retired workspaces alive, so
    VA reuse does not occur.
"""

import os
import threading
from collections import deque
from collections.abc import Callable
from typing import Any

import torch
from torch._C._distributed_c10d import _SymmetricMemory

from vllm.logger import init_logger

logger = init_logger(__name__)

# torch's get_symm_mem_workspace() re-allocates and re-rendezvous's the
# fused-op workspace whenever an op requests more than the current size,
# dropping the old workspace tensor with NO synchronization. Device-side
# work still targeting the old workspace (parked barrier waits on its
# signal pads, in-flight P2P chunk copies from peers) then polls freed /
# recyclable memory and can park its stream forever. Two defenses:
#   * floor the first allocation high enough that growth never happens
#     in practice;
#   * if growth does happen, drain the device and keep the old tensor
#     alive forever (bounded leak, rare) so parked ops keep polling
#     stable memory.
_WORKSPACE_FLOOR = (
    int(os.getenv("VLLM_SYMM_MEM_WORKSPACE_FLOOR_MB", "256")) * 1024 * 1024
)
_workspace_graveyard: list[torch.Tensor] = []
_orig_get_workspace: Callable[[str, int], torch.Tensor] | None = None

# Debug sidecar (env-gated, not for commit): when barriers stop
# completing, dump the host-side sequence counters and the handle's own
# signal pad so the wedge can be attributed (missing peer writes vs
# mismatched waits vs op-order divergence). All bookkeeping is gated on
# _dbg_enabled and O(1) per call.
_dbg_enabled = False
_dbg_handles: dict[int, _SymmetricMemory] = {}  # own-pad VA -> latest handle
_dbg_calls = 0

# Pipeline event bracketing (env VLLM_SYMM_MEM_PIPELINE_TRACE): record a
# CUDA event after every fused-pipeline stage; at a stall the sidecar
# host-queries them and names the first stage that never completed —
# no debugger involved (the SM120 debugger backend kills wedged
# targets). Entries: (label, stream_id, event).
_pipe_enabled = False
_pipe_trace: deque[tuple[str, int, torch.cuda.Event]] | None = None
_pipe_lock = threading.Lock()
_orig_pipelined = None


def _pipe_mark(label: str) -> None:
    ev = torch.cuda.Event()
    ev.record()
    with _pipe_lock:
        assert _pipe_trace is not None
        _pipe_trace.append(
            (label, torch.cuda.current_stream().cuda_stream & 0xFFFF, ev)
        )


_MAX_CHANNELS = 8
_RING = 4
_lock = threading.Lock()
# own-pad VA -> {channel: last sequence number}
_seq_state: dict[int, dict[int, int]] = {}
_installed = False
_orig_barrier = None
_drv: Any = None


def _pcie_safe_barrier(
    self: _SymmetricMemory, channel: int = 0, timeout_ms: int = 0
) -> None:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "The PCIe-safe symm-mem barrier cannot be captured in a CUDA "
            "graph: its baked sequence number would satisfy its own wait "
            "on replay without synchronizing the group."
        )
    world = self.world_size
    rank = self.rank
    pad_ptrs = self.signal_pad_ptrs
    if channel >= _MAX_CHANNELS:
        raise ValueError(f"channel {channel} >= {_MAX_CHANNELS}")
    if self.signal_pad_size < _MAX_CHANNELS * _RING * world * 4:
        raise RuntimeError(
            f"signal pad too small: {self.signal_pad_size} < "
            f"{_MAX_CHANNELS * _RING * world * 4}"
        )

    own_key = pad_ptrs[rank]
    global _dbg_calls
    with _lock:
        chans = _seq_state.setdefault(own_key, {})
        seq = chans.get(channel, 0) + 1
        chans[channel] = seq
        if _dbg_enabled:
            # keyed by own-pad VA: handle wrappers are not identity-stable
            # across calls (an unconditional identity-scan list here grew
            # unboundedly and cost O(list) per barrier call — the fake "E
            # perf regression" of 2026-07-08)
            _dbg_calls += 1
            _dbg_handles[own_key] = self

    stream = _drv.CUstream(torch.cuda.current_stream().cuda_stream)
    base = 4 * world * (channel * _RING + seq % _RING)
    eq = int(_drv.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ)
    for peer in range(world):
        (err,) = _drv.cuStreamWriteValue32(
            stream,
            _drv.CUdeviceptr(pad_ptrs[peer] + base + 4 * rank),
            seq,
            0,
        )
        if int(err) != 0:
            raise RuntimeError(f"cuStreamWriteValue32 failed: {err}")
    if _pipe_enabled:
        _pipe_mark(f"bar(ch{channel},#{seq}).writes")
    for peer in range(world):
        (err,) = _drv.cuStreamWaitValue32(
            stream,
            _drv.CUdeviceptr(pad_ptrs[rank] + base + 4 * peer),
            seq,
            eq,
        )
        if int(err) != 0:
            raise RuntimeError(f"cuStreamWaitValue32 failed: {err}")
    if _pipe_enabled:
        _pipe_mark(f"bar(ch{channel},#{seq}).waits")


def reset_pcie_barrier_state() -> None:
    """Forget all per-handle sequence numbers (see module docstring)."""
    with _lock:
        _seq_state.clear()


def _guarded_get_workspace(group_name, min_size):
    import torch.distributed._symmetric_memory as tsm

    tensor = tsm._group_name_to_workspace_tensor.get(group_name)
    size = tensor.numel() * tensor.element_size() if tensor is not None else 0
    need = max(min_size, _WORKSPACE_FLOOR)
    if tensor is not None and size < need:
        logger.warning(
            "symm-mem workspace grows %d -> %d bytes; draining the device "
            "and retiring the old workspace without freeing it",
            size,
            need,
        )
        torch.accelerator.synchronize()
        _workspace_graveyard.append(tensor)
    get_workspace = _orig_get_workspace
    assert get_workspace is not None
    return get_workspace(group_name, need)


def _install_workspace_guard() -> None:
    global _orig_get_workspace
    import torch.distributed._symmetric_memory as tsm

    if _orig_get_workspace is not None:
        return
    _orig_get_workspace = tsm.get_symm_mem_workspace
    tsm.get_symm_mem_workspace = _guarded_get_workspace


# Fused-op pipelines, re-issued for PCIe: torch's stock micro-pipelines
# enqueue peer-memory operations (signal-pad memops, P2P copies)
# concurrently from two streams per process and secure the required
# kernel scheduling order with a sleep-kernel nudge the source itself
# only calls "almost guarantee[d]". On PCIe platforms we replace both
# pipelines with comm-stream variants: every peer-memory operation is
# issued on ONE stream per process, while producers/consumers (pure
# local kernels) ride a second stream ordered by explicit CUDA events
# in both directions (producer-done -> comm may start; peers-done ->
# buffer may be overwritten). Compute/comm overlap is retained; the
# only overlap given up is between P2P copies of different streams,
# which torch's own comments note cannot overlap anyway. Measured on
# 4x SM120 PCIe (8192x4096 @ 4096x4096 bf16 vs unfused NCCL):
# matmul-reduce-scatter 1.78x (stock dual-stream: 1.56x — the explicit
# ordering also removes the stock path's scheduling-miss degradation),
# all-gather-matmul 1.69x (stock: 1.78x). The rarely-used *_last_dim
# all-gather variant is left stock.
_orig_produce_a2a = None
_orig_multi_ag = None


def _comm_stream_produce_and_all2all(
    chunk_producer, output, group_name, out_chunk_dim=0
):
    import torch.distributed._symmetric_memory as tsm
    import torch.distributed.distributed_c10d as c10d

    out_chunks = output.chunk(
        c10d._get_group_size_by_name(group_name), dim=out_chunk_dim
    )
    p2p_workspace_size_req = out_chunks[0].numel() * out_chunks[0].element_size() * 2
    symm_mem = tsm.get_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)
    group_size = symm_mem.world_size
    rank = symm_mem.rank

    comm = torch.cuda.current_stream()
    prod = tsm._get_backend_stream()

    symm_mem.barrier(channel=0)  # entry fence, on comm stream
    prod.wait_stream(comm)

    def get_p2p_buf(r: int, idx: int) -> torch.Tensor:
        offset = 0 if idx == 0 else out_chunks[0].numel()
        return symm_mem.get_buffer(r, out_chunks[0].shape, out_chunks[0].dtype, offset)

    # reuse-fence events: producer of step k+2 must wait until peers
    # finished reading the same buffer at step k (second barrier done)
    reuse_evt: list = [None, None]

    for step in range(1, group_size):
        remote_rank = (rank - step) % group_size
        buf_id = 1 if step % 2 == 0 else 0

        if reuse_evt[buf_id] is not None:
            prod.wait_event(reuse_evt[buf_id])
        with torch.cuda.stream(prod):
            chunk_producer((rank + step) % group_size, get_p2p_buf(rank, buf_id))
        done = torch.cuda.Event()
        done.record(prod)

        comm.wait_event(done)
        # all peer-memory ops stay on the comm stream:
        symm_mem.barrier(channel=step % 2)
        out_chunks[remote_rank].copy_(get_p2p_buf(remote_rank, buf_id))
        symm_mem.barrier(channel=step % 2)
        ev = torch.cuda.Event()
        ev.record(comm)
        reuse_evt[buf_id] = ev

    with torch.cuda.stream(prod):
        chunk_producer(rank, out_chunks[rank])
    comm.wait_stream(prod)
    symm_mem.barrier(channel=0)


def _comm_stream_multi_all_gather_and_consume(
    shard, shard_consumer, ag_out, group_name, ag_out_needed=True
):
    import torch.distributed._symmetric_memory as tsm

    p2p_workspace_size_req = 0
    for x in shard:
        p2p_workspace_size_req += x.numel() * x.element_size()
    symm_mem = tsm.get_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)
    group_size = symm_mem.world_size
    rank = symm_mem.rank

    for x, y in zip(shard, ag_out):
        assert x.is_contiguous()
        assert y.is_contiguous()
        assert x.shape[0] * group_size == y.shape[0]
        assert x.shape[1:] == y.shape[1:]

    comm = torch.cuda.current_stream()
    cons = tsm._get_backend_stream()

    symm_mem.barrier(channel=0)

    def copy_shard(dst, src):
        for d, s in zip(dst, src):
            d.copy_(s)

    def get_p2p_bufs(remote_rank):
        offset_bytes = 0
        bufs = []
        for x in shard:
            buf = symm_mem.get_buffer(
                remote_rank,
                x.shape,
                x.dtype,
                storage_offset=offset_bytes // x.element_size(),
            )
            bufs.append(buf)
            offset_bytes += buf.numel() * buf.element_size()
        return bufs

    shards: list[list[torch.Tensor]] = [[] for _ in range(group_size)]
    for x in ag_out:
        for i, y in enumerate(x.chunk(group_size)):
            shards[i].append(y)

    copy_shard(get_p2p_bufs(rank), shard)  # local write, comm stream
    symm_mem.barrier(channel=1)

    # own shard consumes the input directly; overlap it with peer copies
    cons.wait_stream(comm)
    with torch.cuda.stream(cons):
        shard_consumer(shard, rank)

    for step in range(1, group_size):
        remote_rank = (step + rank) % group_size
        # peer read on the comm stream
        copy_shard(shards[remote_rank], get_p2p_bufs(remote_rank))
        ev = torch.cuda.Event()
        ev.record(comm)
        cons.wait_event(ev)
        with torch.cuda.stream(cons):
            shard_consumer(shards[remote_rank], remote_rank)

    if ag_out_needed:
        copy_shard(shards[rank], shard)  # local copy, comm stream

    comm.wait_stream(cons)
    symm_mem.barrier(channel=0)


def _install_comm_stream_pipelines() -> None:
    """Replace both fused micro-pipelines process-wide. Idempotent."""
    global _orig_produce_a2a, _orig_multi_ag
    import torch.distributed._symmetric_memory as tsm

    if _orig_produce_a2a is not None:
        return
    _orig_produce_a2a = tsm._pipelined_produce_and_all2all
    _orig_multi_ag = tsm._pipelined_multi_all_gather_and_consume
    tsm._pipelined_produce_and_all2all = _comm_stream_produce_and_all2all
    tsm._pipelined_multi_all_gather_and_consume = (
        _comm_stream_multi_all_gather_and_consume
    )
    logger.info_once(
        "torch fused-op micro-pipelines replaced with comm-stream "
        "variants: peer-memory ops on one stream, producers "
        "event-ordered on a second."
    )


def _install_pipeline_trace() -> None:
    """Wrap the fused-op pipeline so every stage records a CUDA event."""
    global _orig_pipelined, _pipe_trace, _pipe_enabled

    import torch.distributed._symmetric_memory as tsm

    if _orig_pipelined is not None:
        return
    _pipe_trace = deque(maxlen=512)
    _orig_pipelined = tsm._pipelined_produce_and_all2all
    op_counter = [0]

    def traced(chunk_producer, output, group_name, out_chunk_dim=0):
        op_counter[0] += 1
        op = op_counter[0]

        def wrapped_producer(dst_rank, buf):
            chunk_producer(dst_rank, buf)
            _pipe_mark(f"op{op}.producer->{dst_rank}")

        _pipe_mark(f"op{op}.begin")
        res = _orig_pipelined(wrapped_producer, output, group_name, out_chunk_dim)
        _pipe_mark(f"op{op}.host-return")
        return res

    tsm._pipelined_produce_and_all2all = traced
    _pipe_enabled = True
    logger.info_once("symm-mem pipeline event bracketing enabled")


# Single-stream fused pipelines (env VLLM_SYMM_MEM_SINGLE_STREAM_FUSED):
# sequential replacements for torch's dual-stream micro-pipelines. torch
# rides cross-rank barrier obligations on a backend stream and secures
# the required kernel scheduling order with a sleep-kernel nudge its own
# comments call "almost guarantee[d]". These variants keep the exact
# buffer/channel discipline but issue everything in order on the current
# stream: no backend stream, no cross-stream joins, no scheduling
# assumptions — trading compute/comm overlap for a concurrency shape
# with no intra-process concurrent peer enqueues. Used as a bisection
# instrument for the driver enqueue wedge and as a candidate fallback.
# (The rarely-hit *_last_dim all-gather variant is left stock.)


def _single_stream_produce_and_all2all(
    chunk_producer, output, group_name, out_chunk_dim=0
):
    import torch.distributed._symmetric_memory as tsm
    import torch.distributed.distributed_c10d as c10d

    out_chunks = output.chunk(
        c10d._get_group_size_by_name(group_name), dim=out_chunk_dim
    )
    p2p_workspace_size_req = out_chunks[0].numel() * out_chunks[0].element_size() * 2
    symm_mem = tsm.get_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)
    group_size = symm_mem.world_size
    rank = symm_mem.rank

    symm_mem.barrier(channel=0)

    def get_p2p_buf(r: int, idx: int) -> torch.Tensor:
        offset = 0 if idx == 0 else out_chunks[0].numel()
        return symm_mem.get_buffer(r, out_chunks[0].shape, out_chunks[0].dtype, offset)

    for step in range(1, group_size):
        remote_rank = (rank - step) % group_size
        # keep torch's step-parity -> buffer/channel pairing
        buf_id = 1 if step % 2 == 0 else 0
        chunk_producer((rank + step) % group_size, get_p2p_buf(rank, buf_id))
        symm_mem.barrier(channel=step % 2)
        out_chunks[remote_rank].copy_(get_p2p_buf(remote_rank, buf_id))
        symm_mem.barrier(channel=step % 2)

    chunk_producer(rank, out_chunks[rank])
    symm_mem.barrier(channel=0)


def _single_stream_multi_all_gather_and_consume(
    shard, shard_consumer, ag_out, group_name, ag_out_needed=True
):
    import torch.distributed._symmetric_memory as tsm

    p2p_workspace_size_req = 0
    for x in shard:
        p2p_workspace_size_req += x.numel() * x.element_size()
    symm_mem = tsm.get_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)
    group_size = symm_mem.world_size
    rank = symm_mem.rank

    for x, y in zip(shard, ag_out):
        assert x.is_contiguous()
        assert y.is_contiguous()
        assert x.shape[0] * group_size == y.shape[0]
        assert x.shape[1:] == y.shape[1:]

    symm_mem.barrier(channel=0)

    def copy_shard(dst, src):
        for d, s in zip(dst, src):
            d.copy_(s)

    def get_p2p_bufs(remote_rank):
        offset_bytes = 0
        bufs = []
        for x in shard:
            buf = symm_mem.get_buffer(
                remote_rank,
                x.shape,
                x.dtype,
                storage_offset=offset_bytes // x.element_size(),
            )
            bufs.append(buf)
            offset_bytes += buf.numel() * buf.element_size()
        return bufs

    shards: list[list[torch.Tensor]] = [[] for _ in range(group_size)]
    for x in ag_out:
        for i, y in enumerate(x.chunk(group_size)):
            shards[i].append(y)

    copy_shard(get_p2p_bufs(rank), shard)
    symm_mem.barrier(channel=1)

    shard_consumer(shard, rank)
    for step in range(1, group_size):
        remote_rank = (step + rank) % group_size
        copy_shard(shards[remote_rank], get_p2p_bufs(remote_rank))
        shard_consumer(shards[remote_rank], remote_rank)

    if ag_out_needed:
        copy_shard(shards[rank], shard)

    symm_mem.barrier(channel=0)


def _install_single_stream_fused() -> None:
    """Replace both fused micro-pipelines process-wide. Idempotent."""
    global _orig_produce_a2a, _orig_multi_ag
    import torch.distributed._symmetric_memory as tsm

    if _orig_produce_a2a is not None:
        return
    _orig_produce_a2a = tsm._pipelined_produce_and_all2all
    _orig_multi_ag = tsm._pipelined_multi_all_gather_and_consume
    tsm._pipelined_produce_and_all2all = _single_stream_produce_and_all2all
    tsm._pipelined_multi_all_gather_and_consume = (
        _single_stream_multi_all_gather_and_consume
    )
    logger.info_once(
        "torch fused-op micro-pipelines replaced with single-stream "
        "sequential variants (VLLM_SYMM_MEM_SINGLE_STREAM_FUSED)."
    )


def install_pcie_safe_barrier() -> None:
    """Replace ``_SymmetricMemory.barrier`` process-wide. Idempotent."""
    global _installed, _orig_barrier, _drv
    if _installed:
        return
    from cuda.bindings import driver as drv

    _drv = drv
    _orig_barrier = _SymmetricMemory.barrier
    _SymmetricMemory.barrier = _pcie_safe_barrier
    _install_workspace_guard()
    if os.getenv("VLLM_SYMM_MEM_STOCK_PIPELINES"):
        logger.info_once(
            "fused-op pipelines left stock "
            "(VLLM_SYMM_MEM_STOCK_PIPELINES, dev override)."
        )
    elif os.getenv("VLLM_SYMM_MEM_SINGLE_STREAM_FUSED"):
        _install_single_stream_fused()
    else:
        _install_comm_stream_pipelines()
    _installed = True
    workspace_floor_mib = _WORKSPACE_FLOOR // (1024 * 1024)
    logger.info_once(
        "torch symm-mem group barrier replaced with the PCIe-safe "
        "stream-memops protocol (no native P2P atomics on this platform); "
        f"fused-op workspace floored at {workspace_floor_mib} MiB with growth guard."
    )
    if os.getenv("VLLM_SYMM_MEM_PIPELINE_TRACE"):
        _install_pipeline_trace()
    if os.getenv("VLLM_SYMM_MEM_BARRIER_DEBUG"):
        global _dbg_enabled
        _dbg_enabled = True
        _start_debug_sidecar()


def _start_debug_sidecar() -> None:
    import time

    def _watch():
        # This thread issues D2H copies while the main thread may hold a
        # global-mode cudagraph capture; relaxed mode makes those legal
        # instead of poisoning the capture (run-8 lesson).
        (err, _) = _drv.cuThreadExchangeStreamCaptureMode(
            _drv.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED
        )
        if int(err) != 0:
            logger.error("BARRIER DEBUG: relaxed capture mode failed: %s", err)
            return
        prev = -1
        frozen = 0
        while True:
            time.sleep(20.0)
            calls = _dbg_calls
            # capture/decode phases legitimately pause fused ops; require
            # a sustained freeze before dumping to cut false positives
            frozen = frozen + 1 if (calls == prev and calls > 0) else 0
            if frozen >= 3 and _dbg_handles:
                try:
                    dumps = []
                    with _lock:
                        handles = list(_dbg_handles.values())
                        seqs = {hex(k): dict(v) for k, v in _seq_state.items()}
                    for h in handles:
                        pad_words = h.signal_pad_size // 4
                        pad = h.get_signal_pad(h.rank, [pad_words], torch.int32)
                        host = pad.cpu()
                        nz = host.nonzero().flatten().tolist()
                        vals = {i: int(host[i]) for i in nz[:120]}
                        dumps.append((hex(h.signal_pad_ptrs[h.rank]), vals))
                    logger.error(
                        "BARRIER STALLED rank=%d calls=%d seqs=%s pads=%s",
                        handles[0].rank,
                        calls,
                        seqs,
                        dumps,
                    )
                    if _pipe_enabled and _pipe_trace:
                        with _pipe_lock:
                            entries = list(_pipe_trace)
                        done_by_stream = {}
                        pending = []
                        for label, sid, ev in entries:
                            if ev.query():
                                done_by_stream[sid] = label
                            else:
                                pending.append(f"s{sid}:{label}")
                        logger.error(
                            "PIPELINE TRACE rank=%d last-complete=%s "
                            "first-pending=%s (pending total %d)",
                            handles[0].rank,
                            done_by_stream,
                            pending[:6],
                            len(pending),
                        )
                except Exception as e:
                    logger.error("BARRIER DEBUG failed: %r", e)
            prev = calls

    threading.Thread(target=_watch, daemon=True, name="pcie-barrier-dbg").start()

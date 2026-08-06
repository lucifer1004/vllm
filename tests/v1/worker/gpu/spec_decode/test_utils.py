# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.worker.gpu.spec_decode.utils import draft_gumbel_pos


def test_draft_gumbel_pos_uses_disjoint_counter_range() -> None:
    """Draft proposals must not reuse a Philox counter the verifier keys on.

    The rejection sampler derives both its acceptance uniform and its recovery
    Gumbel noise from the token position, so an overlap between proposal and
    verification noise would bias the sampler.
    """
    positions = torch.tensor([0, 1, 17, 2**20], dtype=torch.int64)

    actual = draft_gumbel_pos(positions)

    torch.testing.assert_close(actual, positions + (1 << 30))
    # The whole mapped range sits above every position the target can key on.
    assert actual.min() > positions.max()

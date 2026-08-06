# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.worker.gpu.spec_decode.utils import draft_gumbel_pos


def test_draft_gumbel_pos_uses_disjoint_counter_range() -> None:
    positions = torch.tensor([0, 1, 17, 2**20], dtype=torch.int64)

    actual = draft_gumbel_pos(positions)

    torch.testing.assert_close(actual, positions + 1 + (1 << 30))
    assert torch.all(actual > positions)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch


def join_vocab_shards(
    received: torch.Tensor, num_shards: int, local_rows: int
) -> torch.Tensor:
    """Turn source-rank-major all-to-all blocks into row-major logits."""
    width = received.size(-1)
    return (
        received.view(num_shards, local_rows, width)
        .transpose(0, 1)
        .reshape(local_rows, num_shards * width)
    )

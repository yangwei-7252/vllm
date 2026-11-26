# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from vllm.config import (
    DeviceConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import parallel_state
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.platforms import current_platform


def _worker(rank, world, tp, embed_size, head_size, rendezvous):
    dp = world // tp
    device = torch.device(f"cuda:{rank}" if current_platform.is_cuda_alike() else "cpu")
    backend = "nccl" if device.type == "cuda" else "gloo"
    if device.type == "cuda":
        torch.cuda.set_device(device)
    config = VllmConfig(
        parallel_config=ParallelConfig(
            data_parallel_size=dp,
            data_parallel_rank=rank // tp,
            tensor_parallel_size=tp,
            embedding_tensor_parallel_size=embed_size,
            lmhead_tensor_parallel_size=head_size,
            distributed_executor_backend="mp",
        ),
        device_config=DeviceConfig(device=device.type),
        scheduler_config=SchedulerConfig(
            max_model_len=8,
            is_encoder_decoder=False,
            max_num_batched_tokens=8,
            max_num_seqs=4,
        ),
    )
    torch.distributed.init_process_group(
        backend,
        rank=rank,
        world_size=world,
        store=torch.distributed.FileStore(rendezvous, world),
        timeout=timedelta(seconds=60),
    )
    parallel_state.init_distributed_environment(
        world, rank, local_rank=rank, backend=backend
    )
    try:
        with (
            set_current_vllm_config(config),
            torch.inference_mode(),
            torch.device(device),
        ):
            parallel_state.initialize_model_parallel(tp, backend=backend)
            vocab, hidden = 67, 4
            weight = torch.arange(vocab * hidden, dtype=torch.float32).view(
                vocab, hidden
            )
            emb = VocabParallelEmbedding(vocab, hidden, prefix="model.embed_tokens")
            head = ParallelLMHead(vocab, hidden, bias=True, prefix="lm_head")
            processor = LogitsProcessor(vocab)
            bias = torch.arange(vocab, dtype=torch.float32)
            for layer, size in ((emb, embed_size), (head, head_size)):
                assert layer.tp_size == tp * size
                assert layer.tp_rank == rank % tp * size + (rank // tp) % size
                layer.weight_loader(layer.weight, weight)
            head.weight_loader(head.bias, bias)
            if embed_size == head_size:
                assert head.tie_weights(emb).weight is emb.weight
            elif min(embed_size, head_size) > 1:
                with pytest.raises(ValueError, match="matching vocabulary shards"):
                    head.tie_weights(emb)
            for active in (True, False, True):
                replica = rank // tp
                ids = torch.tensor(
                    [0, 33, 66] if replica == 0 else ([65] if active else []),
                    dtype=torch.long,
                )
                states = emb(ids)
                torch.testing.assert_close(states, weight[ids], rtol=0, atol=0)
                expected = states @ weight.T + bias
                logits = processor(head, states, head.bias)
                torch.testing.assert_close(logits, expected, rtol=0, atol=0)
                if head_size > 1:
                    indices, scores = processor.get_top_k_tokens(
                        head, states, 2, head.bias
                    )
                    values, tokens = expected.topk(2, dim=-1)
                    torch.testing.assert_close(indices, tokens)
                    torch.testing.assert_close(scores, values)
                    tokens = processor.get_top_tokens(head, states, head.bias)
                    torch.testing.assert_close(tokens, expected.argmax(dim=-1))

            if head_size > 1:
                extended = ParallelLMHead(vocab + 5, hidden, org_num_embeddings=vocab)
                extended.weight_loader(extended.weight, weight)
                torch.testing.assert_close(
                    processor(extended, states), states @ weight.T, rtol=0, atol=0
                )
    finally:
        parallel_state.destroy_model_parallel()
        parallel_state.destroy_distributed_environment()


@pytest.mark.parametrize(
    ("world", "tp", "embed_size", "head_size"),
    [
        (2, 1, 2, 2),
        (4, 2, 2, 2),
        (4, 1, 2, 4),
        (2, 1, 1, 2),
        (2, 1, 2, 1),
        (2, 1, 1, 1),
    ],
)
def test_vocab_tp_row_ownership(tmp_path: Path, world, tp, embed_size, head_size):
    mp.spawn(
        _worker,
        args=(world, tp, embed_size, head_size, str(tmp_path / "store")),
        nprocs=world,
    )

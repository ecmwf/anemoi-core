# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

import anemoi.models.distributed.random as random_utils
from anemoi.models.distributed.random import get_independent_torch_seed
from anemoi.models.distributed.random import get_synced_torch_seed
from anemoi.models.distributed.random import seed_torch_rng_sources
from anemoi.models.distributed.random import use_synced_torch_rng
from anemoi.models.layers.utils import maybe_checkpoint

_MAX_LIGHTNING_SEED = 2**32 - 1


def test_rank_default_seeds_differ_from_synced_seed() -> None:
    base_seed = 1234

    assert get_independent_torch_seed(base_seed, global_rank=0) != get_independent_torch_seed(
        base_seed,
        global_rank=1,
    )
    assert get_independent_torch_seed(base_seed, global_rank=0) != get_synced_torch_seed(base_seed)
    assert get_synced_torch_seed(base_seed, sync_group_id=0) != get_synced_torch_seed(
        base_seed,
        sync_group_id=1,
    )


def test_derived_seeds_fit_lightning_seed_range() -> None:
    base_seed = 2**32 - 1

    derived_seeds = [
        get_independent_torch_seed(base_seed, global_rank=0),
        get_independent_torch_seed(base_seed, global_rank=1024),
        get_synced_torch_seed(base_seed, sync_group_id=0),
        get_synced_torch_seed(base_seed, sync_group_id=1024),
    ]

    assert all(0 <= seed <= _MAX_LIGHTNING_SEED for seed in derived_seeds)


def test_synced_context_fails_when_not_initialized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(random_utils, "_synced_seed", None)
    monkeypatch.setattr(random_utils, "_synced_state", None)

    with pytest.raises(RuntimeError, match="Synchronized Torch RNG source has not been initialized"):
        with use_synced_torch_rng():
            pass


def test_synced_context_matches_across_independent_rank_sources() -> None:
    base_seed = 1234

    seed_torch_rng_sources(base_seed, global_rank=0, reset_synced=True)
    rank_0_default = torch.rand(4)
    with use_synced_torch_rng():
        rank_0_synced = torch.rand(4)

    seed_torch_rng_sources(base_seed, global_rank=1, reset_synced=True)
    rank_1_default = torch.rand(4)
    with use_synced_torch_rng():
        rank_1_synced = torch.rand(4)

    assert not torch.allclose(rank_0_default, rank_1_default)
    assert torch.allclose(rank_0_synced, rank_1_synced)


def test_synced_context_can_be_scoped_by_model_group() -> None:
    base_seed = 1234

    seed_torch_rng_sources(base_seed, global_rank=0, sync_group_id=0, reset_synced=True)
    with use_synced_torch_rng():
        group_0_synced = torch.rand(4)

    seed_torch_rng_sources(base_seed, global_rank=1, sync_group_id=1, reset_synced=True)
    with use_synced_torch_rng():
        group_1_synced = torch.rand(4)

    assert not torch.allclose(group_0_synced, group_1_synced)


def test_seed_default_seeds_default_stream_with_independent_seed() -> None:
    base_seed = 1234
    global_rank = 4

    independent_seed = seed_torch_rng_sources(
        base_seed,
        global_rank=global_rank,
        seed_default=True,
        reset_synced=True,
    )
    default_after_seed = torch.rand(4)

    torch.manual_seed(independent_seed)
    expected_default = torch.rand(4)

    torch.testing.assert_close(default_after_seed, expected_default)


def test_synced_context_restores_and_advances_separate_rng_state() -> None:
    base_seed = 4321

    seed_torch_rng_sources(base_seed, global_rank=3, reset_synced=True)
    torch.rand(3)
    with use_synced_torch_rng():
        first_synced = torch.rand(4)
    default_after_synced = torch.rand(4)

    seed_torch_rng_sources(base_seed, global_rank=3, reset_synced=True)
    torch.rand(3)
    expected_default = torch.rand(4)
    with use_synced_torch_rng():
        first_synced_again = torch.rand(4)
    with use_synced_torch_rng():
        second_synced = torch.rand(4)

    seed_torch_rng_sources(base_seed, global_rank=3, reset_synced=True)
    with use_synced_torch_rng():
        combined_synced = torch.rand(8)

    assert torch.allclose(default_after_synced, expected_default)
    assert torch.allclose(first_synced, first_synced_again)
    assert torch.allclose(first_synced, combined_synced[:4])
    assert torch.allclose(second_synced, combined_synced[4:])


def test_checkpointed_synced_context_preserves_recomputed_rng_state() -> None:
    base_seed = 2468
    x = torch.ones(8, requires_grad=True)

    def fn(input_tensor: torch.Tensor) -> torch.Tensor:
        with use_synced_torch_rng():
            noise = torch.rand_like(input_tensor)
        return input_tensor * noise

    seed_torch_rng_sources(base_seed, global_rank=0, reset_synced=True)
    output = maybe_checkpoint(fn, True, x)
    forward_noise = output.detach().clone()
    output.sum().backward()
    with use_synced_torch_rng():
        next_after_checkpoint = torch.rand_like(x)

    seed_torch_rng_sources(base_seed, global_rank=0, reset_synced=True)
    with use_synced_torch_rng():
        expected_forward_noise = torch.rand_like(x)
    with use_synced_torch_rng():
        expected_next = torch.rand_like(x)

    torch.testing.assert_close(forward_noise, expected_forward_noise)
    torch.testing.assert_close(x.grad, forward_noise)
    torch.testing.assert_close(next_after_checkpoint, expected_next)


def test_synced_context_fails_if_cuda_starts_inside_context(monkeypatch: pytest.MonkeyPatch) -> None:
    base_seed = 2468
    original_get_current_rng_state = random_utils._get_current_rng_state
    call_count = 0

    seed_torch_rng_sources(base_seed, global_rank=0, reset_synced=True)

    def get_current_rng_state_with_new_cuda_device() -> random_utils.TorchRNGState:
        nonlocal call_count
        call_count += 1
        state = original_get_current_rng_state()
        if call_count == 2:
            state.cuda[0] = torch.empty_like(state.cpu)
        return state

    monkeypatch.setattr(random_utils, "_get_current_rng_state", get_current_rng_state_with_new_cuda_device)

    with pytest.raises(RuntimeError, match="CUDA was initialized inside use_synced_torch_rng"):
        with use_synced_torch_rng():
            pass

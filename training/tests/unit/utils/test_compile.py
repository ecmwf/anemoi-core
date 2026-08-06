# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
from pathlib import Path

import pytest
import torch

from anemoi.training.utils.compile import init_compile_cache
from anemoi.training.utils.compile import load_compile_cache
from anemoi.training.utils.compile import save_compile_cache
from anemoi.training.utils.compile import subset_tensor


def _compile_cache_test_function(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x) * 2 + x


def test_subset_tensor_none_returns_input() -> None:
    x = torch.arange(24).reshape(2, 3, 4)

    out, subset_index, subset_dim = subset_tensor(x, None)

    assert out is x
    assert subset_index is None
    assert subset_dim is None


def test_subset_tensor_ellipsis_only_returns_input() -> None:
    x = torch.arange(24).reshape(2, 3, 4)

    out, subset_index, subset_dim = subset_tensor(x, (Ellipsis,))

    assert out is x
    assert subset_index is None
    assert subset_dim is None


def test_subset_tensor_tuple_indices_select_first_dimension() -> None:
    x = torch.arange(24).reshape(3, 2, 4)

    out, subset_index, subset_dim = subset_tensor(x, (2, 0))

    expected_index = torch.tensor([2, 0], dtype=torch.long)

    torch.testing.assert_close(out, torch.index_select(x, dim=0, index=expected_index))
    torch.testing.assert_close(subset_index, expected_index)
    assert subset_dim == 0


def test_subset_tensor_list_indices_select_first_dimension() -> None:
    x = torch.arange(24).reshape(3, 2, 4)

    out, subset_index, subset_dim = subset_tensor(x, [2, 0])
    expected_index = torch.tensor([2, 0], dtype=torch.long)

    torch.testing.assert_close(out, torch.index_select(x, dim=0, index=expected_index))
    torch.testing.assert_close(subset_index, expected_index)
    assert subset_dim == 0


def test_subset_tensor_ellipsis_selects_last_dimension() -> None:
    x = torch.arange(24).reshape(2, 3, 4)

    out, subset_index, subset_dim = subset_tensor(x, (Ellipsis, [3, 1]))
    expected_index = torch.tensor([3, 1], dtype=torch.long)

    torch.testing.assert_close(out, torch.index_select(x, dim=-1, index=expected_index))
    torch.testing.assert_close(subset_index, expected_index)
    assert subset_dim == -1


def test_subset_tensor_tensor_indices_are_moved_to_input_device_and_long_dtype() -> None:
    x = torch.arange(24).reshape(2, 3, 4)
    input_index = torch.tensor([3, 0], dtype=torch.int32)

    out, subset_index, subset_dim = subset_tensor(x, (Ellipsis, input_index))
    expected_index = input_index.to(device=x.device, dtype=torch.long)

    torch.testing.assert_close(out, torch.index_select(x, dim=-1, index=expected_index))
    assert subset_index is not None
    assert subset_index.device == x.device
    assert subset_index.dtype == torch.long
    torch.testing.assert_close(subset_index, expected_index)
    assert subset_dim == -1


def test_compile_cache_round_trip_between_inductor_cache_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Save compile artifacts from cache directory A and load them into directory B."""
    cache_root_a = tmp_path / "cache_a"
    cache_root_b = tmp_path / "cache_b"
    cache_file = tmp_path / "compile_cache.pt"
    expected_cache_dir_a = cache_root_a / "anemoi_compile_cache"
    expected_cache_dir_b = cache_root_b / "anemoi_compile_cache"
    x = torch.linspace(-1, 1, 128)
    expected = _compile_cache_test_function(x)

    torch._dynamo.reset()
    try:
        # Initialize Inductor's cache environment from TMPDIR before compiling so
        # generated artifacts are written under directory A.
        monkeypatch.setenv("TMPDIR", str(cache_root_a))
        init_compile_cache()
        assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(expected_cache_dir_a)

        compiled_from_a = torch.compile(
            _compile_cache_test_function,
            fullgraph=True,
            dynamic=False,
            mode="max-autotune",
        )
        torch.testing.assert_close(compiled_from_a(x), expected)
        save_compile_cache(str(cache_file))
        assert cache_file.is_file()

        # Clear the in-memory Dynamo cache so the next invocation needs a fresh
        # compilation. Loading configures a distinct Inductor cache directory B and
        # repopulates the relevant compiler caches from the saved artifact bundle.
        torch._dynamo.reset()
        monkeypatch.setenv("TMPDIR", str(cache_root_b))
        load_compile_cache(str(cache_file))
        assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(expected_cache_dir_b)

        compiled_from_b = torch.compile(
            _compile_cache_test_function,
            fullgraph=True,
            dynamic=False,
            mode="max-autotune",
        )
        torch.testing.assert_close(compiled_from_b(x), expected)
    finally:
        torch._dynamo.reset()

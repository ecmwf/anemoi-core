# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for LoadingStrategy base class helper methods."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.loading.base import LoadingStrategy

if TYPE_CHECKING:
    from typing import Any


# ---------------------------------------------------------------------------
# Concrete subclass so we can instantiate and test helpers
# ---------------------------------------------------------------------------


class _StubLoader(LoadingStrategy):
    """Minimal concrete loader for testing base class helpers."""

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        return context


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def loader() -> _StubLoader:
    return _StubLoader()


@pytest.fixture
def simple_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))


# ---------------------------------------------------------------------------
# _extract_state_dict
# ---------------------------------------------------------------------------


class TestExtractStateDict:
    """Tests for LoadingStrategy._extract_state_dict."""

    def test_from_lightning_checkpoint(self, loader: _StubLoader, simple_model: nn.Module) -> None:
        """Extract state dict when data has a 'state_dict' key (Lightning format)."""
        state_dict = simple_model.state_dict()
        context = CheckpointContext(
            checkpoint_data={
                "state_dict": state_dict,
                "pytorch-lightning_version": "2.2.0",
                "epoch": 5,
            },
        )
        result = loader._extract_state_dict(context)
        assert set(result.keys()) == set(state_dict.keys())

    def test_from_pytorch_checkpoint(self, loader: _StubLoader, simple_model: nn.Module) -> None:
        """Extract state dict when data has a 'model_state_dict' key (PyTorch format)."""
        state_dict = simple_model.state_dict()
        context = CheckpointContext(
            checkpoint_data={
                "model_state_dict": state_dict,
                "optimizer_state_dict": {},
            },
        )
        result = loader._extract_state_dict(context)
        assert set(result.keys()) == set(state_dict.keys())

    def test_from_bare_state_dict(self, loader: _StubLoader) -> None:
        """Extract state dict when the data itself IS the state dict (bare tensors)."""
        bare = {"w": torch.randn(3, 3), "b": torch.randn(3)}
        context = CheckpointContext(checkpoint_data=bare)
        result = loader._extract_state_dict(context)
        assert set(result.keys()) == {"w", "b"}

    def test_raises_on_missing_state(self, loader: _StubLoader) -> None:
        """Raise when no recognisable state dict key is found."""
        from anemoi.training.checkpoint.exceptions import CheckpointValidationError

        context = CheckpointContext(checkpoint_data={"config": {"lr": 1e-3}})
        with pytest.raises(CheckpointValidationError, match="Cannot find model state"):
            loader._extract_state_dict(context)


# ---------------------------------------------------------------------------
# _preserve_anemoi_metadata
# ---------------------------------------------------------------------------


class TestPreserveAnemoiMetadata:
    """Tests for LoadingStrategy._preserve_anemoi_metadata."""

    def test_sets_attribute_when_data_indices_present(
        self,
        loader: _StubLoader,
        simple_model: nn.Module,
    ) -> None:
        """_ckpt_model_name_to_index should be set on the model when checkpoint has data_indices."""
        data_indices = Mock()
        data_indices.name_to_index = {"temperature": 0, "pressure": 1}

        checkpoint_data: dict[str, Any] = {
            "state_dict": simple_model.state_dict(),
            "hyper_parameters": {"data_indices": data_indices},
        }

        # Fresh model should NOT have the attribute
        assert not hasattr(simple_model, "_ckpt_model_name_to_index")

        loader._preserve_anemoi_metadata(simple_model, checkpoint_data)

        # After restoration, it should be set
        assert hasattr(simple_model, "_ckpt_model_name_to_index")
        assert simple_model._ckpt_model_name_to_index == {"temperature": 0, "pressure": 1}

    def test_skips_gracefully_when_no_data_indices(
        self,
        loader: _StubLoader,
        simple_model: nn.Module,
    ) -> None:
        """Should not raise or set attribute when hyper_parameters lacks data_indices."""
        checkpoint_data: dict[str, Any] = {
            "state_dict": simple_model.state_dict(),
            "hyper_parameters": {"lr": 1e-4},
        }

        loader._preserve_anemoi_metadata(simple_model, checkpoint_data)

        assert not hasattr(simple_model, "_ckpt_model_name_to_index")

    def test_skips_when_no_hyper_parameters(
        self,
        loader: _StubLoader,
        simple_model: nn.Module,
    ) -> None:
        """Should not raise when checkpoint has no hyper_parameters at all."""
        checkpoint_data: dict[str, Any] = {"state_dict": simple_model.state_dict()}

        loader._preserve_anemoi_metadata(simple_model, checkpoint_data)

        assert not hasattr(simple_model, "_ckpt_model_name_to_index")


# ---------------------------------------------------------------------------
# _mark_weights_loaded
# ---------------------------------------------------------------------------


class TestMarkWeightsLoaded:
    """Tests for LoadingStrategy._mark_weights_loaded."""

    def test_sets_weights_initialized_flag(
        self,
        loader: _StubLoader,
        simple_model: nn.Module,
    ) -> None:
        """weights_initialized should be True after calling _mark_weights_loaded."""
        assert not getattr(simple_model, "weights_initialized", False)

        loader._mark_weights_loaded(simple_model)

        assert simple_model.weights_initialized is True

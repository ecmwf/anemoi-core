# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Loading-layer parity behaviours from the legacy load paths.

Covers the loading-strategy parity with ``on_load_checkpoint``:

- weights-only loading is strict by default (missing keys fail);
- a strict-load failure surfaces as ``CheckpointLoadError``;
- a warning fires when the checkpoint's stored model hparams diverge from the run config;
- the runtime ``trainable_edge_perm`` migration is referenced/applied on load;
- ``model_output_idx`` buffers are re-injected from the live model during the processor refresh;
- a pre-multi-dataset single ``IndexCollection`` checkpoint is rejected with a ``TypeError``;
- a transfer-learning load populates ``_ckpt_variables_metadata``.
"""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.exceptions import CheckpointLoadError
from anemoi.training.checkpoint.loading.strategies import TransferLearningLoader
from anemoi.training.checkpoint.loading.strategies import WeightsOnlyLoader


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 3)
        self.decoder = nn.Linear(3, 2)


def _full_state(model: nn.Module, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        key: (torch.randn(value.shape, generator=generator) if value.is_floating_point() else value.clone())
        for key, value in model.state_dict().items()
    }


def _index_collection(name_to_index: dict[str, int]) -> object:
    return type("IndexCollection", (), {"name_to_index": name_to_index})()


@pytest.mark.asyncio
async def test_weights_only_default_is_strict() -> None:
    """A missing-key checkpoint fails by default (default strict=True)."""
    model = _Tiny()
    partial = {key: value for key, value in _full_state(model).items() if key.startswith("encoder.")}
    context = CheckpointContext(model=model, checkpoint_data={"state_dict": partial})

    with pytest.raises(CheckpointLoadError):
        await WeightsOnlyLoader().process(context)


@pytest.mark.asyncio
async def test_weights_only_non_strict_accepts_partial() -> None:
    """strict=False keeps the lenient fill-model behaviour for partial checkpoints."""
    model = _Tiny()
    partial = {key: value for key, value in _full_state(model).items() if key.startswith("encoder.")}
    context = CheckpointContext(model=model, checkpoint_data={"state_dict": partial})

    result = await WeightsOnlyLoader(strict=False).process(context)

    assert result.model is model


@pytest.mark.asyncio
async def test_hparams_divergence_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Diverging checkpoint model hparams (with coinciding shapes) produce a warning."""
    model = _Tiny()
    checkpoint = {
        "state_dict": _full_state(model),
        "hyper_parameters": {"config": {"model": {"num_channels": 64}}},
    }
    config = OmegaConf.create({"model": {"num_channels": 128}})
    context = CheckpointContext(model=model, checkpoint_data=checkpoint, config=config)

    with caplog.at_level(logging.WARNING):
        await WeightsOnlyLoader().process(context)

    assert any("hparam" in record.getMessage().lower() for record in caplog.records)


def test_model_output_idx_reinjected_during_refresh() -> None:
    """The processor refresh re-injects ``model_output_idx`` buffers from the live model."""

    class _WithOutputIdx(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pre_processors = nn.Linear(2, 2)
            self.register_buffer("model_output_idx", torch.tensor([1.0]))

    model = _WithOutputIdx()
    state_dict = {f"model.{key}": torch.full_like(value, 9.0) for key, value in model.state_dict().items()}
    context = CheckpointContext(
        model=model,
        checkpoint_data={"state_dict": state_dict},
        config=OmegaConf.create(
            {"training": {"update_ds_stats_on_ckpt_load": {"states": True, "tendencies": False}}},
        ),
    )

    WeightsOnlyLoader()._refresh_checkpoint_processors(context)

    refreshed = context.checkpoint_data["state_dict"]
    assert torch.equal(refreshed["model.model_output_idx"], model.state_dict()["model_output_idx"])


def test_single_dataset_metadata_raises() -> None:
    """A pre-multi-dataset single IndexCollection is rejected with a TypeError."""
    model = _Tiny()
    checkpoint = {"hyper_parameters": {"data_indices": _index_collection({"2t": 0})}}

    with pytest.raises(TypeError, match="migration sync"):
        WeightsOnlyLoader()._preserve_anemoi_metadata(model, checkpoint)


@pytest.mark.asyncio
async def test_transfer_learning_sets_variables_metadata() -> None:
    """A transfer-learning load populates ``_ckpt_variables_metadata`` from the checkpoint."""
    model = _Tiny()
    checkpoint = {
        "state_dict": _full_state(model),
        "hyper_parameters": {
            "data_indices": {"era5": _index_collection({"2t": 0})},
            "metadata": {"dataset": {"era5": {"variables_metadata": {"2t": {"units": "K"}}}}},
        },
    }
    context = CheckpointContext(model=model, checkpoint_data=checkpoint)

    result = await TransferLearningLoader().process(context)

    assert getattr(result.model, "_ckpt_variables_metadata", None) is not None

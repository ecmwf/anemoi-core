# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Multi-domain (participant) wiring of the training module."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import torch
from torch_geometric.data import HeteroData

from anemoi.training.data.batch_meta import META_KEY
from anemoi.training.train.methods.base import DEFAULT_PARTICIPANT_KEY
from anemoi.training.train.methods.single import SingleTraining
from anemoi.training.utils.masks import Boolean1DMask
from anemoi.training.utils.masks import NoOutputMask
from anemoi.training.utils.masks import build_output_masks

PARTICIPANT_GRID_SIZES = {"west": 4, "north": 6}


def _participant_graph() -> HeteroData:
    graph = HeteroData()
    for participant, num_nodes in PARTICIPANT_GRID_SIZES.items():
        graph[f"data_{participant}"].x = torch.zeros(num_nodes, 2)
        graph[f"data_{participant}"].num_nodes = num_nodes
        graph[f"data_{participant}"]["cutout_mask"] = torch.ones(num_nodes, 1)
    return graph


def _make_module(participants: tuple[str, ...] = ("west", "north")) -> SingleTraining:
    """Build a `__new__`-based training module wired only with the participant state."""
    module = SingleTraining.__new__(SingleTraining)
    pl.LightningModule.__init__(module)

    module.dataset_names = ["data"]
    module.target_dataset_names = ["data"]
    module._fused_dataset_graph = bool(participants)
    module._participants = list(participants)
    module._active_participant = participants[0] if participants else None

    keys = list(participants) if participants else [None]
    module._output_mask = {p: {"data": NoOutputMask()} for p in keys}
    module._scalers = {p: {"data": {"scaler": p}} for p in keys}
    module._updating_scalars = {p: {"data": {}} for p in keys}
    module._grid_sizes = {p: {"data": PARTICIPANT_GRID_SIZES.get(p, 8)} for p in keys}
    module._shard_sizes = {p: {"data": [PARTICIPANT_GRID_SIZES.get(p, 8)]} for p in keys}
    module._loss = torch.nn.ModuleDict(
        {module._participant_key(p): torch.nn.ModuleDict({"data": torch.nn.Identity()}) for p in keys},
    )
    module._metrics = torch.nn.ModuleDict(
        {module._participant_key(p): torch.nn.ModuleDict({"data": torch.nn.ModuleDict()}) for p in keys},
    )
    module._select_participant_objects()

    module.model = SimpleNamespace(model=MagicMock(), pre_processors={"data": lambda x: x})
    return module


# ── build_output_masks ────────────────────────────────────────────────────────


def test_build_output_masks_resolves_participant_node_group() -> None:
    graph = _participant_graph()
    configs = {"data": {"_target_": "anemoi.training.utils.masks.Boolean1DMask", "attribute_name": "cutout_mask"}}

    for participant, num_nodes in PARTICIPANT_GRID_SIZES.items():
        masks = build_output_masks(configs, graph, participant=participant)
        assert isinstance(masks["data"], Boolean1DMask)
        assert len(masks["data"].mask) == num_nodes


def test_build_output_masks_reports_the_missing_participant_node_group() -> None:
    graph = _participant_graph()
    configs = {"data": {"_target_": "anemoi.training.utils.masks.Boolean1DMask", "attribute_name": "cutout_mask"}}

    with pytest.raises(AssertionError, match="data_east"):
        build_output_masks(configs, graph, participant="east")


# ── participant state ─────────────────────────────────────────────────────────


def test_single_domain_module_has_no_participants() -> None:
    module = _make_module(participants=())

    assert module.participants == []
    assert module.active_participant is None
    assert module.loss is module._loss[DEFAULT_PARTICIPANT_KEY]
    assert module.grid_sizes == {"data": 8}


def test_set_active_participant_is_a_noop_without_participants() -> None:
    module = _make_module(participants=())

    module.set_active_participant("west")

    assert module.active_participant is None
    module.model.model.set_active_participant.assert_not_called()


def test_module_defaults_to_the_first_participant() -> None:
    module = _make_module()

    assert module.participants == ["west", "north"]
    assert module.active_participant == "west"
    assert module.grid_sizes == {"data": 4}
    assert module.shard_sizes == {"data": [4]}
    assert module.scalers == {"data": {"scaler": "west"}}
    assert module.loss is module._loss["west"]
    assert module.metrics is module._metrics["west"]
    assert module.output_mask is module._output_mask["west"]


def test_set_active_participant_switches_every_grid_sized_object() -> None:
    module = _make_module()

    module.set_active_participant("north")

    assert module.active_participant == "north"
    assert module.grid_sizes == {"data": 6}
    assert module.shard_sizes == {"data": [6]}
    assert module.scalers == {"data": {"scaler": "north"}}
    assert module.updating_scalars is module._updating_scalars["north"]
    assert module.loss is module._loss["north"]
    assert module.metrics is module._metrics["north"]
    assert module.output_mask is module._output_mask["north"]
    module.model.model.set_active_participant.assert_called_once_with("north")


def test_set_active_participant_rejects_an_unknown_participant() -> None:
    module = _make_module()

    with pytest.raises(ValueError, match="Unknown participant 'south'"):
        module.set_active_participant("south")


def test_all_participants_keep_their_losses_registered() -> None:
    module = _make_module()

    registered = dict(module.named_modules())
    assert "_loss.west.data" in registered
    assert "_loss.north.data" in registered


def test_first_activation_of_every_participant_is_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="anemoi.training.train.methods.base"):
        module = _make_module()  # activates `west`
        module.set_active_participant("north")
        module.set_active_participant("west")
        module.set_active_participant("north")

    activated = [record.getMessage() for record in caplog.records if "activated for the first time" in record.message]
    assert len(activated) == 2, "one line per participant, however often they are switched between"
    assert "'west'" in activated[0]
    assert "{'data': 4}" in activated[0], "the grid sizes identify which participant's graph is in use"
    assert "'north'" in activated[1]
    assert "{'data': 6}" in activated[1]
    assert module._activated_participants == ("west", "north")


# ── graph node resolution ─────────────────────────────────────────────────────


def test_data_node_name_appends_the_participant() -> None:
    module = _make_module()

    assert module._data_node_name("data", "west") == "data_west"
    assert module._data_node_name("era5", "north") == "era5_north"


def test_data_node_name_without_participant() -> None:
    module = _make_module(participants=())

    module._fused_dataset_graph = True
    assert module._data_node_name("era5", None) == "era5"

    module._fused_dataset_graph = False
    assert module._data_node_name("era5", None) == "data"


# ── on_after_batch_transfer ───────────────────────────────────────────────────


def _make_batch_transfer_module() -> SingleTraining:
    module = _make_module()
    module.keep_batch_sharded = True
    module.model_comm_group_size = 2
    module.reader_group_rank = 0
    module.is_first_step = False
    module.update_scalers = MagicMock()
    module.allgather_batch = MagicMock(side_effect=lambda tensor, _name: tensor)
    return module


def test_on_after_batch_transfer_switches_participant_before_sharding() -> None:
    module = _make_batch_transfer_module()
    batch = {"data": torch.zeros(2, 1, 1, 6, 3), META_KEY: {"participant": ["north", "north"]}}

    module.on_after_batch_transfer(batch, 0)

    assert module.active_participant == "north"
    # the shard sizes follow the participant that was just selected
    assert module.grid_shard_sizes == {"data": [6]}
    assert module.grid_shard_slice == {"data": slice(0, 6)}


def test_on_after_batch_transfer_keeps_the_participant_without_meta() -> None:
    module = _make_batch_transfer_module()

    module.on_after_batch_transfer({"data": torch.zeros(2, 1, 1, 4, 3)}, 0)

    assert module.active_participant == "west"
    assert module.grid_shard_sizes == {"data": [4]}

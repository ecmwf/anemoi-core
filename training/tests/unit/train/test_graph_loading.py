# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.graphs.projection_helpers import DEFAULT_DATASET_NAME
from anemoi.training.train.train import AnemoiTrainer


def test_existing_graph_validation_detects_fused_graph_from_loaded_file(tmp_path: Path) -> None:
    graph = HeteroData()
    graph["era5"].num_nodes = 1
    graph["cerra"].num_nodes = 1

    graph_path = tmp_path / "fused_graph.pt"
    torch.save(graph, graph_path)

    trainer = AnemoiTrainer.__new__(AnemoiTrainer)
    trainer.config = OmegaConf.create(
        {
            "graph": {"overwrite": False},
            "system": {"input": {"graph": str(graph_path)}},
            "dataloader": {
                "training": {
                    "datasets": {
                        "era5": {"dataset_config": {"dataset": "unused"}},
                        "cerra": {"dataset_config": {"dataset": "unused"}},
                    },
                },
            },
        },
    )

    loaded_graph = trainer.graph_data

    assert set(loaded_graph.node_types) == {"era5", "cerra"}


def _build_trainer_config_with_dataset_config(dataset_config: dict) -> AnemoiTrainer:
    trainer = AnemoiTrainer.__new__(AnemoiTrainer)
    trainer.config = OmegaConf.create(
        {
            "graph": {
                "overwrite": True,
                "nodes": {
                    DEFAULT_DATASET_NAME: {
                        "node_builder": {
                            "_target_": "anemoi.graphs.nodes.AnemoiDatasetNodes",
                            "dataset": "placeholder",
                        },
                    },
                },
                "edges": [],
            },
            "system": {"input": {"graph": None}},
            "dataloader": {
                "training": {
                    "datasets": {
                        "data": {"dataset_config": dataset_config},
                    },
                },
            },
        },
    )
    return trainer


def test_graph_build_forwards_full_dataset_config_to_node_builder() -> None:
    """Extra keys in dataset_config (e.g. check_variables_compatibility) must reach the node builder.

    Regression test for a bug where only dataset_config["dataset"] was passed to the graph node
    builder, stripping options like check_variables_compatibility that open_dataset needs.
    """
    dataset_config = {
        "dataset": "/path/to/dataset.zarr",
        "check_variables_compatibility": {"ignore_type_of_level": ["sp"]},
    }

    trainer = _build_trainer_config_with_dataset_config(dataset_config)

    mock_creator = MagicMock()
    mock_creator.create.return_value = HeteroData()

    with patch("anemoi.training.train.train.GraphCreator", return_value=mock_creator) as mock_gc_cls:
        trainer.graph_data

    # GraphCreator is called with the modified graph_config as first positional arg
    graph_config_arg = mock_gc_cls.call_args[0][0]
    captured_dataset = OmegaConf.to_container(
        graph_config_arg.nodes[DEFAULT_DATASET_NAME].node_builder.dataset,
        resolve=True,
    )

    assert captured_dataset == dataset_config, (
        "Extra open_dataset kwargs (like check_variables_compatibility) must be forwarded "
        f"to the graph node builder alongside the dataset path, but got: {captured_dataset}"
    )


def test_graph_build_drops_schema_keys_from_node_builder() -> None:
    """Schema-managed keys must NOT be forwarded to the graph node builder.

    Keys defined in DatasetConfigSchema (frequency, select, drop, statistics,
    step_start, step_end, step_frequency) are training-only and should not
    reach the graph node builder, which only understands the open_dataset API.
    """
    dataset_config = {
        "dataset": "/path/to/dataset.zarr",
        "frequency": "6h",
        "select": ["2t", "10u"],
        "drop": ["tp"],
        "statistics": "/path/to/stats.zarr",
        "check_variables_compatibility": {"ignore_type_of_level": ["sp"]},
    }

    trainer = _build_trainer_config_with_dataset_config(dataset_config)

    mock_creator = MagicMock()
    mock_creator.create.return_value = HeteroData()

    with patch("anemoi.training.train.train.GraphCreator", return_value=mock_creator) as mock_gc_cls:
        trainer.graph_data

    graph_config_arg = mock_gc_cls.call_args[0][0]
    captured_dataset = OmegaConf.to_container(
        graph_config_arg.nodes[DEFAULT_DATASET_NAME].node_builder.dataset,
        resolve=True,
    )

    # Schema keys must be absent; only dataset path + extra kwargs should be present.
    schema_keys_present = {"frequency", "select", "drop", "statistics"} & set(captured_dataset)
    assert not schema_keys_present, (
        f"Schema-managed keys must be dropped before passing to the graph node builder, "
        f"but found: {schema_keys_present}"
    )
    assert captured_dataset["dataset"] == dataset_config["dataset"]
    assert captured_dataset["check_variables_compatibility"] == dataset_config["check_variables_compatibility"]


@pytest.mark.parametrize(
    ("statistics_from", "expected"),
    [("h2", "/path/h2.zarr"), (None, "/path/h1.zarr")],
)
def test_graph_build_uses_reference_participant(
    statistics_from: str | None,
    expected: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With participants, the single graph is built from the reference participant (statistics_from, else first)."""
    trainer = _build_trainer_config_with_dataset_config({"dataset": "unused"})
    trainer.config.dataloader.training.datasets.data = OmegaConf.create(
        {
            "statistics_from": statistics_from,
            "participants": {
                "h1": {"dataset_config": {"dataset": "/path/h1.zarr"}},
                "h2": {"dataset_config": {"dataset": "/path/h2.zarr"}},
            },
        },
    )

    mock_creator = MagicMock()
    mock_creator.create.return_value = HeteroData()

    with (
        patch("anemoi.training.train.train.GraphCreator", return_value=mock_creator) as mock_gc_cls,
        caplog.at_level("INFO", logger="anemoi.training.train.train"),
    ):
        trainer.graph_data

    graph_config_arg = mock_gc_cls.call_args[0][0]
    captured_dataset = OmegaConf.to_container(graph_config_arg.nodes[DEFAULT_DATASET_NAME].node_builder.dataset)
    assert captured_dataset == {"dataset": expected}
    assert f"building the graph from participant '{Path(expected).stem}'" in caplog.text


def _participant_graph_trainer(participants: list[str]) -> AnemoiTrainer:
    """Trainer whose graph config defines one graph per participant."""
    node_cfg = {
        "node_builder": {"_target_": "anemoi.graphs.nodes.AnemoiDatasetNodes", "dataset": "placeholder"},
    }
    trainer = AnemoiTrainer.__new__(AnemoiTrainer)
    trainer.config = OmegaConf.create(
        {
            "graph": {
                "overwrite": True,
                "participants": {
                    participant: {"nodes": {DEFAULT_DATASET_NAME: node_cfg, "hidden": node_cfg}, "edges": []}
                    for participant in participants
                },
            },
            "system": {"input": {"graph": None}},
            "dataloader": {
                "training": {
                    "datasets": {
                        "data": {
                            "participants": {
                                participant: {"dataset_config": {"dataset": f"/path/{participant}.zarr"}}
                                for participant in participants
                            },
                        },
                    },
                },
            },
        },
    )
    return trainer


def _single_participant_graph() -> HeteroData:
    graph = HeteroData()
    graph[DEFAULT_DATASET_NAME].num_nodes = 2
    graph["hidden"].num_nodes = 3
    graph[(DEFAULT_DATASET_NAME, "to", "hidden")].edge_index = torch.tensor([[0, 1], [1, 0]])
    return graph


def _patch_graph_creator(captured: list) -> MagicMock:
    def make_creator(config: object) -> MagicMock:
        creator = MagicMock()
        creator.create.return_value = _single_participant_graph()
        captured.append(config)
        return creator

    return patch("anemoi.training.train.train.GraphCreator", side_effect=make_creator)


def test_participant_graphs_are_fused_with_suffixed_node_groups() -> None:
    """Two participants -> one graph each, fused into a single graph with suffixed node groups."""
    trainer = _participant_graph_trainer(["west", "north"])
    captured: list = []

    with _patch_graph_creator(captured):
        graph = trainer.graph_data

    assert set(graph.node_types) == {"data_west", "hidden_west", "data_north", "hidden_north"}
    assert ("data_west", "to", "hidden_west") in graph.edge_types
    assert ("data_north", "to", "hidden_north") in graph.edge_types
    assert graph["data_west"].num_nodes == 2
    assert graph["hidden_north"].num_nodes == 3

    # each participant's own dataset reached its own graph config
    injected = [config.nodes[DEFAULT_DATASET_NAME].node_builder.dataset["dataset"] for config in captured]
    assert injected == ["/path/west.zarr", "/path/north.zarr"]


def test_single_participant_graph_keeps_plain_node_names() -> None:
    """A single participant must produce exactly the graph a single-domain config produces today."""
    trainer = _participant_graph_trainer(["west"])

    with _patch_graph_creator([]):
        graph = trainer.graph_data

    assert set(graph.node_types) == {DEFAULT_DATASET_NAME, "hidden"}
    assert (DEFAULT_DATASET_NAME, "to", "hidden") in graph.edge_types


def test_participant_graph_rejects_unknown_participant() -> None:
    """A graph participant that the dataset does not define is a config error."""
    trainer = _participant_graph_trainer(["west"])
    trainer.config.graph.participants.typo = trainer.config.graph.participants.west

    with _patch_graph_creator([]), pytest.raises(ValueError, match="not participants of dataset"):
        trainer.graph_data

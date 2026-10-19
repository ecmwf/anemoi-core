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

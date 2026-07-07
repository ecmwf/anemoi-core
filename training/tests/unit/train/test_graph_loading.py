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


def test_graph_build_forwards_full_dataset_config_to_node_builder() -> None:
    """Extra keys in dataset_config (e.g. check_variables_compatibility) must reach the node builder.

    Regression test for a bug where only dataset_config["dataset"] was passed to the graph node
    builder, stripping options like check_variables_compatibility that open_dataset needs.
    """
    dataset_config = {
        "dataset": "/path/to/dataset.zarr",
        "check_variables_compatibility": {"ignore_type_of_level": ["sp"]},
    }

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

    captured = {}

    def fake_create(_save_path: None = None, _overwrite: bool = False) -> HeteroData:
        # Capture what was set on the node builder's dataset before GraphCreator.create runs.
        captured["dataset"] = OmegaConf.to_container(
            trainer.config.graph.nodes[DEFAULT_DATASET_NAME].node_builder.dataset,
            resolve=True,
        )
        return HeteroData()

    mock_creator = MagicMock()
    mock_creator.create.side_effect = fake_create

    with patch("anemoi.training.train.train.GraphCreator", return_value=mock_creator):
        trainer.graph_data

    assert captured["dataset"] == dataset_config, (
        "The full dataset_config (including check_variables_compatibility) must be forwarded "
        f"to the graph node builder, but got: {captured['dataset']}"
    )

# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

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

# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.training.train.graph_data import TrainerGraphDataFactory


def build_graph_config(overrides: dict | None = None) -> DictConfig:
    config = OmegaConf.create(
        {
            "system": {"input": {"graph": None}},
            "graph": {
                "overwrite": True,
                "nodes": {
                    "data": {
                        "node_builder": {
                            "dataset": "placeholder",
                        },
                    },
                },
                "edges": [],
            },
            "dataloader": {
                "training": {
                    "dataset_config": {
                        "dataset": "datasets/single.zarr",
                    },
                },
            },
        },
    )
    if overrides is not None:
        config = OmegaConf.merge(config, OmegaConf.create(overrides))
    return config


def test_is_existing_graph_mode_requires_graph_file_and_empty_graph_definition() -> None:
    config = build_graph_config(
        {
            "system": {"input": {"graph": "graphs/existing.pt"}},
            "graph": {
                "overwrite": False,
                "nodes": None,
                "edges": None,
            },
        },
    )

    factory = TrainerGraphDataFactory(config)

    assert factory.is_existing_graph_mode() is True


def test_validate_loaded_graph_rejects_missing_dataset_node() -> None:
    graph = HeteroData()
    graph["era5"].num_nodes = 1

    with pytest.raises(ValueError, match="Missing \\['cerra'\\]"):
        TrainerGraphDataFactory.validate_loaded_graph(graph, ["era5", "cerra"])


def test_create_graph_for_dataset_updates_dataset_path_and_merges_projections() -> None:
    config = build_graph_config()
    factory = TrainerGraphDataFactory(config)
    created_graph = HeteroData()

    with (
        patch("anemoi.training.train.graph_data.merge_projection_and_graph_config") as merge_projection,
        patch("anemoi.training.train.graph_data.GraphCreator") as graph_creator_cls,
    ):
        graph_creator_cls.return_value.create.return_value = created_graph

        graph = factory.create_graph_for_dataset("datasets/era5.zarr", "era5")

    assert graph is created_graph
    merge_projection.assert_called_once()
    assert merge_projection.call_args.kwargs["dataset_names"] == ["era5"]

    graph_config = graph_creator_cls.call_args.kwargs["config"]
    assert graph_config.nodes.data.node_builder.dataset == "datasets/era5.zarr"
    graph_creator_cls.return_value.create.assert_called_once_with(save_path=None, overwrite=True)


def test_build_uses_fused_graph_for_multiple_datasets() -> None:
    config = build_graph_config(
        {
            "graph": {
                "nodes": {
                    "era5": {},
                    "cerra": {},
                    "hidden": {},
                },
            },
            "dataloader": {
                "training": {
                    "datasets": {
                        "era5": {"dataset_config": {"dataset": "datasets/era5.zarr"}},
                        "cerra": {"dataset_config": {"dataset": "datasets/cerra.zarr"}},
                    },
                },
            },
        },
    )
    factory = TrainerGraphDataFactory(config)
    fused_graph = HeteroData()

    with (
        patch.object(factory, "create_fused_graph", return_value=fused_graph) as create_fused_graph,
        patch.object(factory, "create_graph_for_dataset") as create_graph_for_dataset,
    ):
        graph = factory.build()

    assert graph is fused_graph
    create_fused_graph.assert_called_once_with(["era5", "cerra"])
    create_graph_for_dataset.assert_not_called()


def test_build_uses_single_dataset_source_from_dataset_config() -> None:
    config = build_graph_config()
    factory = TrainerGraphDataFactory(config)
    single_graph = HeteroData()

    with patch.object(factory, "create_graph_for_dataset", return_value=single_graph) as create_graph_for_dataset:
        graph = factory.build()

    assert graph is single_graph
    create_graph_for_dataset.assert_called_once_with("datasets/single.zarr", "data")


def test_build_requires_dataset_key_in_dataset_config_mapping() -> None:
    config = build_graph_config()
    config.dataloader.training.dataset_config = OmegaConf.create({"source": "datasets/single.zarr"})
    factory = TrainerGraphDataFactory(config)

    with pytest.raises(ValueError, match="missing 'dataset' key"):
        factory.build()


def test_existing_graph_path_raises_when_file_is_missing(tmp_path: Path) -> None:
    config = build_graph_config(
        {
            "system": {"input": {"graph": tmp_path / "missing.pt"}},
            "graph": {
                "overwrite": False,
                "nodes": None,
                "edges": None,
            },
        },
    )
    factory = TrainerGraphDataFactory(config)

    with pytest.raises(FileNotFoundError, match="Existing graph file not found"):
        factory.existing_graph_path()

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


def test_create_graph_updates_dataset_path_and_merges_projections() -> None:
    config = build_graph_config()
    factory = TrainerGraphDataFactory(config)
    created_graph = HeteroData()

    with (
        patch("anemoi.training.train.graph_data.merge_projection_and_graph_config") as merge_projection,
        patch("anemoi.training.train.graph_data.GraphCreator") as graph_creator_cls,
    ):
        graph_creator_cls.return_value.create.return_value = created_graph

        graph = factory.create_graph(dataset_names=["era5"], dataset_path="datasets/era5.zarr")

    assert graph is created_graph
    merge_projection.assert_called_once()
    assert merge_projection.call_args.kwargs["dataset_names"] == ["era5"]

    graph_config = graph_creator_cls.call_args.kwargs["config"]
    assert graph_config.nodes.data.node_builder.dataset == "datasets/era5.zarr"
    graph_creator_cls.return_value.create.assert_called_once_with(save_path=None, overwrite=True)


def test_create_graph_reuses_configured_graph_path_without_suffix(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.pt"
    config = build_graph_config(
        {
            "system": {"input": {"graph": graph_path}},
            "graph": {"overwrite": False},
        },
    )
    factory = TrainerGraphDataFactory(config)
    loaded_graph = HeteroData()
    loaded_graph["data"].num_nodes = 1
    graph_path.write_text("fake graph")

    with patch.object(factory, "load_graph_from_file", return_value=loaded_graph) as load_graph_from_file:
        graph = factory.create_graph(dataset_names=["era5"], dataset_path="datasets/era5.zarr")

    assert graph is loaded_graph
    load_graph_from_file.assert_called_once_with(graph_path)


def test_create_graph_validates_reused_graph_against_generic_data_node(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.pt"
    config = build_graph_config(
        {
            "system": {"input": {"graph": graph_path}},
            "graph": {"overwrite": False},
        },
    )
    factory = TrainerGraphDataFactory(config)
    loaded_graph = HeteroData()
    graph_path.write_text("fake graph")

    with (
        patch.object(factory, "load_graph_from_file", return_value=loaded_graph) as load_graph_from_file,
        patch.object(factory, "validate_loaded_graph") as validate_loaded_graph,
    ):
        graph = factory.create_graph(dataset_names=["era5"], dataset_path="datasets/era5.zarr")

    assert graph is loaded_graph
    load_graph_from_file.assert_called_once_with(graph_path)
    validate_loaded_graph.assert_called_once_with(loaded_graph, ["data"])


def test_create_graph_validates_reused_graph_against_dataset_nodes_for_fused_graphs(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.pt"
    config = build_graph_config(
        {
            "system": {"input": {"graph": graph_path}},
            "graph": {
                "overwrite": False,
                "nodes": {
                    "era5": {},
                    "cerra": {},
                    "hidden": {},
                },
            },
        },
    )
    factory = TrainerGraphDataFactory(config)
    loaded_graph = HeteroData()
    graph_path.write_text("fake graph")

    with (
        patch.object(factory, "load_graph_from_file", return_value=loaded_graph) as load_graph_from_file,
        patch.object(factory, "validate_loaded_graph") as validate_loaded_graph,
    ):
        graph = factory.create_graph(dataset_names=["era5", "cerra"])

    assert graph is loaded_graph
    load_graph_from_file.assert_called_once_with(graph_path)
    validate_loaded_graph.assert_called_once_with(loaded_graph, ["era5", "cerra"])


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

    with patch.object(factory, "create_graph", return_value=fused_graph) as create_graph:
        graph = factory.build()

    assert graph is fused_graph
    create_graph.assert_called_once_with(dataset_names=["era5", "cerra"])


def test_build_validates_existing_multi_dataset_graph_against_dataset_names(tmp_path: Path) -> None:
    graph_path = tmp_path / "existing.pt"
    graph_path.write_text("fake graph")
    config = build_graph_config(
        {
            "system": {"input": {"graph": graph_path}},
            "graph": {
                "overwrite": False,
                "nodes": None,
                "edges": None,
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
    loaded_graph = HeteroData()

    with (
        patch.object(factory, "load_graph_from_file", return_value=loaded_graph) as load_graph_from_file,
        patch.object(factory, "validate_loaded_graph") as validate_loaded_graph,
    ):
        graph = factory.build()

    assert graph is loaded_graph
    load_graph_from_file.assert_called_once_with(graph_path)
    validate_loaded_graph.assert_called_once_with(loaded_graph, ["era5", "cerra"])


def test_build_uses_single_dataset_source_from_dataset_config() -> None:
    config = build_graph_config()
    factory = TrainerGraphDataFactory(config)
    single_graph = HeteroData()

    with patch.object(factory, "create_graph", return_value=single_graph) as create_graph:
        graph = factory.build()

    assert graph is single_graph
    create_graph.assert_called_once_with(dataset_names=["data"], dataset_path="datasets/single.zarr")


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

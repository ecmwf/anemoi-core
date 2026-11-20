# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import tempfile
from pathlib import Path
from unittest import mock

import pytest
import torch
from hydra import compose
from hydra import initialize_config_module
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData
from torch_geometric.utils import is_undirected

from anemoi.graphs.create import GraphCreator
from anemoi.training.commands.config import ConfigGenerator
from anemoi.training.schemas.base_schema import BaseSchema


@pytest.fixture
def config_generator() -> ConfigGenerator:
    return ConfigGenerator()


def test_dump_config_composes_from_config_path_directory(config_generator: ConfigGenerator) -> None:
    """`dump` composes the named config from a ``--config-path`` directory and writes the merged YAML."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        config_dir = Path(tmpdirname) / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "mycfg.yaml").write_text("foo: bar")

        output_path = Path(tmpdirname) / "output.yaml"
        config_generator.dump_config(config_dir, "mycfg", output_path)

        assert output_path.exists()
        assert OmegaConf.load(output_path) == {"foo": "bar"}


def test_dump_config_sort(config_generator: ConfigGenerator) -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        config_path = Path(tmpdirname) / "config"
        config_path.mkdir(parents=True, exist_ok=True)

        unsorted_cfg = OmegaConf.create({"zebra": 1, "alpha": 2})

        with mock.patch("anemoi.training.commands.config.compose", return_value=unsorted_cfg):
            unsorted_output = Path(tmpdirname) / "unsorted.yaml"
            config_generator.dump_config(config_path, "test", unsorted_output, sort=False)
            assert unsorted_output.read_text().splitlines() == ["zebra: 1", "alpha: 2"]

            sorted_output = Path(tmpdirname) / "sorted.yaml"
            config_generator.dump_config(config_path, "test", sorted_output, sort=True)
            assert sorted_output.read_text().splitlines() == ["alpha: 2", "zebra: 1"]


def test_validate_config_uses_package_path(config_generator: ConfigGenerator) -> None:
    """Test that validate_config works correctly with package configs.

    This test verifies the fix for issue #570: the AnemoiSearchPathPlugin
    adds 'pkg://anemoi.training/config' to the search path, enabling
    discovery of package configs like 'training/default'.

    Note: The package path is added by the plugin, not by the initialize() call,
    so we just verify the method executes successfully.
    """
    with (
        mock.patch("anemoi.training.commands.config.initialize"),
        mock.patch(
            "anemoi.training.commands.config.compose",
            return_value=OmegaConf.create({"test": "value"}),
        ),
        mock.patch("anemoi.training.commands.config.BaseSchema"),
    ):
        # Call validate_config - it should succeed
        # The AnemoiSearchPathPlugin will add the package path automatically
        config_generator.validate_config("test-config", mask_env_vars=False)

        # If we get here without exception, the validation worked


def test_validate_config_with_mask_env_vars(config_generator: ConfigGenerator) -> None:
    """Test that validate_config works with mask_env_vars option."""
    with (
        mock.patch("anemoi.training.commands.config.initialize"),
        mock.patch(
            "anemoi.training.commands.config.compose",
            return_value=OmegaConf.create({"test": "value"}),
        ),
        mock.patch("anemoi.training.commands.config.BaseSchema"),
        mock.patch.object(
            config_generator,
            "_mask_slurm_env_variables",
            return_value=OmegaConf.create({"test": "masked"}),
        ) as mock_mask,
    ):
        # Call validate_config with mask_env_vars=True
        config_generator.validate_config("test-config", mask_env_vars=True)

        # Verify that _mask_slurm_env_variables was called
        mock_mask.assert_called_once()


def test_validate_config_composes_from_config_path_directory(config_generator: ConfigGenerator) -> None:
    """`validate` can be pointed at a config that lives in an arbitrary directory."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        config_dir = Path(tmpdirname)
        (config_dir / "standalone.yaml").write_text("foo: bar")

        with mock.patch("anemoi.training.commands.config.BaseSchema") as mock_schema:
            config_generator.validate_config("standalone", mask_env_vars=False, config_path=config_dir)

        mock_schema.assert_called_once_with(foo="bar")


def test_optimizer_config_group_can_be_overridden() -> None:
    with initialize_config_module(version_base=None, config_module="anemoi.training.config"):
        cfg = compose(config_name="config", overrides=["training/optimization/optimizer=zero"])

    assert cfg.training.optimization.optimizer._target_ == "torch.distributed.optim.ZeroRedundancyOptimizer"


def test_da_corrector_example_composes_and_builds_bidirectional_edges() -> None:
    """The shared preset composes with instrument overrides and a usable graph."""
    with initialize_config_module(version_base=None, config_module="anemoi.training.config"):
        cfg = compose(
            config_name="da_corrector",
            overrides=[
                "system.input.dataset=/tmp/observations.zarr",
                "system.input.graph=/tmp/corrector.pt",
                "system.output.root=/tmp/corrector-run",
                "training.corrector.processor.num_layers=2",
            ],
        )

    OmegaConf.resolve(cfg)
    corrector = BaseSchema(**cfg).training.corrector
    assert corrector.type == "processor"
    assert corrector.processor.num_layers == 2
    assert {name for group in corrector.instrument_groups.values() for name in group.corrector_variables} == set(
        cfg.data.datasets.data.corrector,
    )
    assert [(edge.source_name, edge.target_name) for edge in cfg.graph.edges] == [
        ("data", "hidden"),
        ("hidden", "hidden"),
        ("hidden", "data"),
        ("data", "data"),
    ]

    # Exercise the configured data edges on existing nodes without opening a dataset.
    generator = torch.Generator().manual_seed(23)
    graph = HeteroData()
    graph["data"].x = torch.rand(32, 2, generator=generator)
    edge_config = OmegaConf.to_container(cfg.graph.edges[-1], resolve=True)
    graph = GraphCreator(OmegaConf.create({"edges": [edge_config]})).update_graph(graph)
    edges = graph["data", "to", "data"]
    assert is_undirected(edges.edge_index, num_nodes=32)
    assert torch.unique(edges.edge_index, dim=1).shape[1] == edges.edge_index.shape[1]
    for name in corrector.processor.sub_graph_edge_attributes:
        assert edges[name].shape[0] == edges.edge_index.shape[1]
        assert torch.isfinite(edges[name]).all()


@pytest.mark.parametrize("config_name", ["temporal_downscaler", "temporal_downscaler_ensemble"])
def test_temporal_downscaler_tendency_scalers_use_output_timestep(config_name: str) -> None:
    with initialize_config_module(version_base=None, config_module="anemoi.training.config"):
        cfg = compose(config_name=config_name, overrides=["task.output_timestep=2H"])

    OmegaConf.resolve(cfg)

    scalers = cfg.training.scalers.datasets.data
    assert scalers.stdev_tendency.timestep == "2H"
    assert scalers.var_tendency.timestep == "2H"

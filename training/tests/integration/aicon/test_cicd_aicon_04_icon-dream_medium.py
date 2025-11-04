# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
# Various tests of the Anemoi components using a sample data set.
#
# This script is not part of a productive ML workflow, but is
# used for CI/CD!
import os
import pathlib
from functools import reduce
from operator import getitem

import matplotlib as mpl
import pytest
import torch
from conftest import GetTmpPaths
from hydra import compose
from hydra import initialize
from omegaconf import DictConfig
from typeguard import typechecked

import anemoi.training
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.train.train import AnemoiTrainer
from anemoi.utils.testing import GetTestArchive
from anemoi.utils.testing import GetTestData

os.environ["ANEMOI_BASE_SEED"] = "42"
os.environ["ANEMOI_CONFIG_PATH"] = str(pathlib.Path(anemoi.training.__file__).parent / "config")
mpl.use("agg")


@pytest.fixture
@typechecked
def aicon_config_with_tmp_dir(get_tmp_paths: GetTmpPaths, get_test_archive: GetTestArchive) -> DictConfig:
    """Get AICON config with temporary output paths."""
    with initialize(version_base=None, config_path="./"):
        config = compose(config_name="test_cicd_aicon_04_icon-dream_medium")

    tmp_dir, rel_paths, dataset_urls = get_tmp_paths(config, ["dataset", "forcing_dataset"])
    config.hardware.paths.output = tmp_dir
    config.hardware.paths.graph = tmp_dir
    dataset, forcing_dataset = rel_paths
    config.hardware.paths.data = tmp_dir
    config.hardware.files.dataset = dataset
    config.hardware.files.forcing_dataset = forcing_dataset

    for url in dataset_urls:
        get_test_archive(url)

    return config


@pytest.fixture
@typechecked
def aicon_config_with_grid(aicon_config_with_tmp_dir: DictConfig, get_test_data: GetTestData) -> DictConfig:
    """Temporarily download the ICON grid specified in the config.

    Downloading the grid is required as the AICON grid is currently required as a netCDF file.
    """
    aicon_config_with_tmp_dir.graph.nodes.icon_mesh.node_builder.grid_filename = get_test_data(
        aicon_config_with_tmp_dir.graph.nodes.icon_mesh.node_builder.grid_filename,
    )
    return aicon_config_with_tmp_dir


@pytest.fixture
@typechecked
def trained_aicon(aicon_config_with_grid: DictConfig) -> tuple[AnemoiTrainer, float, float]:
    """Train AICON and return testable objects."""
    trainer = AnemoiTrainer(aicon_config_with_grid)
    initial_sum = float(torch.tensor(list(map(torch.sum, trainer.model.parameters()))).sum())
    trainer.train()
    final_sum = float(torch.tensor(list(map(torch.sum, trainer.model.parameters()))).sum())
    return trainer, initial_sum, final_sum


@typechecked
def assert_metadatakeys(metadata: dict, *metadata_keys: tuple[str, ...]) -> None:
    """Assert the presence of all `metadata_keys` in `metadata`.

    A metadata key is a tuple that describes a path within the `metadata` dict of dicts.
    E.g., if `metadata_keys[0] = ("a", "b", "c")`, this test will verify the existence of `metadata["a"]["b"]["c"]`.
    """
    errors = []
    for keys in metadata_keys:
        try:
            reduce(getitem, keys, metadata)
        except KeyError:  # noqa: PERF203
            keys = "".join(f"[{k!r}]" for k in keys)
            errors.append("missing metadata" + keys)
    if errors:
        raise KeyError("\n".join(errors))


@typechecked
def test_config_validation_aicon(aicon_config_with_tmp_dir: DictConfig) -> None:
    BaseSchema(**aicon_config_with_tmp_dir)


@pytest.mark.slow
@typechecked
def test_aicon_metadata(aicon_config_with_grid: DictConfig) -> None:
    """Test for presence of metadata required for inference.

    The objective of this test is to monitor changes to the metadata structure and ensure compatibility.
    Please update the path to each datum accordingly whenever a revision to the metadata structure is implemented.
    """
    trainer = AnemoiTrainer(aicon_config_with_grid)

    assert_metadatakeys(
        trainer.metadata,
        ("config", "data", "timestep"),
        ("config", "graph", "nodes", "icon_mesh", "node_builder", "max_level_dataset"),
        ("config", "training", "precision"),
        ("data_indices", "data", "input", "diagnostic"),
        ("data_indices", "data", "input", "full"),
        ("data_indices", "data", "output", "full"),
        ("data_indices", "model", "input", "forcing"),
        ("data_indices", "model", "input", "full"),
        ("data_indices", "model", "input", "prognostic"),
        ("data_indices", "model", "output", "full"),
        ("dataset", "shape"),
    )

    assert torch.is_tensor(trainer.graph_data["data"].x), "data coordinates not present"

    # Assert heterogeneity of num_chunks setting.
    assert aicon_config_with_grid.model.encoder.num_chunks != aicon_config_with_grid.model.decoder.num_chunks

    # Monitor path and setting of num_chunks
    assert trainer.model.model.model.encoder.proc.num_chunks == aicon_config_with_grid.model.encoder.num_chunks
    assert trainer.model.model.model.decoder.proc.num_chunks == aicon_config_with_grid.model.decoder.num_chunks


@pytest.mark.slow
@typechecked
def test_aicon_training(trained_aicon: tuple) -> None:
    _, initial_sum, final_sum = trained_aicon
    assert initial_sum != final_sum

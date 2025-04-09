# (C) Copyright 2025 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
#
# Various tests of the Anemoi components using a sample data set.
#
# This script is not part of a productive ML workflow, but is
# used for CI/CD!
import os
import pathlib
import tempfile
from functools import reduce
from operator import getitem
from typing import Optional

import matplotlib as mpl
import pytest
import torch
from hydra import compose
from hydra import initialize
from omegaconf import DictConfig
from typeguard import typechecked

import anemoi.training
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.train.train import AnemoiTrainer

os.environ["ANEMOI_BASE_SEED"] = "42"
os.environ["ANEMOI_CONFIG_PATH"] = str(pathlib.Path(anemoi.training.__file__).parent / "config")
mpl.use("agg")


@typechecked
def aicon_config(output_dir: Optional[str] = None) -> DictConfig:
    """Generate AICON config and overwrite output paths, if output_dir is given."""
    with initialize(version_base=None, config_path="./"):
        config = compose(config_name="test_cicd_aicon_04_icon-dream_medium")

    if output_dir is not None:
        config.hardware.paths.output = output_dir
        config.hardware.paths.graph = output_dir

    return config


@typechecked
def train(config: DictConfig) -> tuple[AnemoiTrainer, float, float]:
    """
    Download the grid required to train AICON and run the Trainer.

    Downloading the grid is required as the AICON grid is currently required as a netCDF file.

    Returns testable objects.
    """
    grid_filename = config.graph.nodes.icon_mesh.node_builder.grid_filename
    with tempfile.NamedTemporaryFile(suffix=".nc") as grid_fp:
        if grid_filename.startswith(("http://", "https://")):
            import urllib.request

            urllib.request.urlretrieve(grid_filename, grid_fp.name)  # noqa: S310
            config.graph.nodes.icon_mesh.node_builder.grid_filename = grid_fp.name

        trainer = AnemoiTrainer(config)
        initial_sum = float(torch.tensor(list(map(torch.sum, trainer.model.parameters()))).sum())
        trainer.train()
        final_sum = float(torch.tensor(list(map(torch.sum, trainer.model.parameters()))).sum())
    return trainer, initial_sum, final_sum


@pytest.fixture
def get_trainer() -> tuple:
    with tempfile.TemporaryDirectory() as output_dir:
        return train(aicon_config(output_dir=output_dir))


@typechecked
def assert_metadatakeys(metadata: dict, *metadata_keys: tuple[str, ...]) -> None:
    """Test presence of metadata required for inference."""
    try:
        for keys in metadata_keys:
            reduce(getitem, keys, metadata)
    except KeyError:
        keys = "".join(f"[{k!r}]" for k in keys)
        raise KeyError("missing metadata" + keys) from None


def test_config_validation_aicon() -> None:
    BaseSchema(**aicon_config())


@pytest.mark.longtests
def test_main(get_trainer: tuple) -> None:
    trainer, initial_sum, final_sum = get_trainer
    assert initial_sum != final_sum

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

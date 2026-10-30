# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
import shutil
from collections.abc import Callable
from functools import partial
from pathlib import Path

import pytest
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.training.testing import GetTmpPath
from anemoi.utils.testing import GetTestData
from anemoi.utils.testing import TemporaryDirectoryForTestData


# NOTE: Do not delete
# It is not called explictly, but
# is run by pytest during initalisation
def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    # suppress logging spam when using torch compile
    logging.getLogger("torch.__trace").setLevel(logging.WARNING)
    logging.getLogger("torch.__trace").propagate = False


# NOTE: Do not delete
# It is not called explictly, but
# it runs before every integration test
@pytest.fixture(autouse=True)
def _reset_torch_compile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset torch compile state before each test."""
    import torch._dynamo

    inductor_cache = tmp_path / "torchinductor_cache"
    shutil.rmtree(inductor_cache, ignore_errors=True)
    inductor_cache.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(inductor_cache))

    torch._dynamo.reset()
    return


@pytest.fixture(autouse=True)
def set_working_directory() -> None:
    """Automatically set the working directory to the repo root."""
    repo_root = Path(__file__).resolve().parent
    while not (repo_root / ".git").exists() and repo_root != repo_root.parent:
        repo_root = repo_root.parent

    os.chdir(repo_root)


@pytest.fixture
def config_with_tempdir(tmp_path: Path) -> DictConfig:
    return OmegaConf.create({"system": {"output": {"root": str(tmp_path)}}})


@pytest.fixture
def get_tmp_path(temporary_directory_for_test_data: TemporaryDirectoryForTestData) -> GetTmpPath:
    return GetTmpPath(temporary_directory_for_test_data)


def _get_multiscale_cfgs(training_loss_cfg: DictConfig) -> list[DictConfig]:
    """Extract multiscale_config dicts that contain loss_matrices."""
    multiscale_cfg = training_loss_cfg.get("multiscale_config")
    if multiscale_cfg is not None and "loss_matrices" in multiscale_cfg:
        return [multiscale_cfg]
    if "losses" in training_loss_cfg:
        results = []
        for sub_loss in training_loss_cfg.losses:
            mc = sub_loss.get("multiscale_config")
            if mc is not None and "loss_matrices" in mc:
                results.append(mc)
        return results
    return []


def _handle_truncation_matrices(cfg: DictConfig, get_test_data: GetTestData) -> DictConfig:
    url_loss_matrices = cfg.system.input.loss_matrices_path
    tmp_path_loss_matrices = None

    training_losses_cfg = get_multiple_datasets_config(cfg.training.training_loss)
    for dataset_name, training_loss_cfg in training_losses_cfg.items():
        multiscale_cfgs = _get_multiscale_cfgs(training_loss_cfg)

        for multiscale_cfg in multiscale_cfgs:
            for file in multiscale_cfg.get("loss_matrices") or []:
                if file is not None:
                    tmp_path_loss_matrices = get_test_data(url_loss_matrices + file)
            if tmp_path_loss_matrices is not None:
                OmegaConf.set_struct(multiscale_cfg, False)
                multiscale_cfg.loss_matrices_path = str(Path(tmp_path_loss_matrices).parent)

        if tmp_path_loss_matrices is not None:
            resolved_path = str(Path(tmp_path_loss_matrices).parent)
            cfg.system.input.loss_matrices_path = Path(tmp_path_loss_matrices).parent

            val_multiscale_cfg = cfg.training.validation_metrics.datasets[dataset_name].multiscale.multiscale_config
            OmegaConf.set_struct(val_multiscale_cfg, False)
            val_multiscale_cfg.loss_matrices_path = resolved_path
        cfg.training.training_loss.datasets[dataset_name] = training_loss_cfg
    return cfg


@pytest.fixture
def handle_truncation_matrices(get_test_data: GetTestData) -> Callable[[DictConfig], DictConfig]:
    """Return a function that downloads the loss truncation matrices of a config and points the config to them."""
    return partial(_handle_truncation_matrices, get_test_data=get_test_data)

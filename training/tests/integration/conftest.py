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
from pathlib import Path

import pytest
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.training.testing import GetTmpPath
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

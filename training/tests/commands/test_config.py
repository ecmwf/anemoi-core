# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import pytest
from omegaconf import OmegaConf

from anemoi.training.commands.config import ConfigGenerator


@pytest.fixture
def config_generator() -> ConfigGenerator:
    return ConfigGenerator()


@pytest.fixture
def temp_dir() -> Path:
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


class TestConfig:

    def test_dump_config(self, config_generator: ConfigGenerator, temp_dir: Path) -> None:
        config_path = temp_dir / "config"
        config_path.mkdir(parents=True, exist_ok=True)
        (config_path / "test.yaml").write_text("test: value")

        output_file = temp_dir / "output.yaml"

        with mock.patch("anemoi.training.commands.config.chdir"), mock.patch(
            "anemoi.training.commands.config.initialize",
        ), mock.patch("anemoi.training.commands.config.compose", return_value=OmegaConf.create({"test": "value"})):
            config_generator.dump_config(config_path, "test", output_file)

        assert output_file.exists()
        with output_file.open() as f:
            dumped_config = OmegaConf.load(f)
        assert dumped_config == {"test": "value"}

    def test_dump_config_no_files(self, config_generator: ConfigGenerator, temp_dir: Path) -> None:
        config_path = temp_dir / "config"
        config_path.mkdir(parents=True, exist_ok=True)

        output_file = temp_dir / "output.yaml"

        with pytest.raises(FileNotFoundError):
            config_generator.dump_config(config_path, "test", output_file)

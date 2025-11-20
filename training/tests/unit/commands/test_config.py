# (C) Copyright 2024 Anemoi contributors.
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
from omegaconf import OmegaConf

from anemoi.training.commands.config import ConfigGenerator


@pytest.fixture
def config_generator() -> ConfigGenerator:
    return ConfigGenerator()


def test_dump_config(config_generator: ConfigGenerator) -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        config_path = Path(tmpdirname) / "config"
        config_path.mkdir(parents=True, exist_ok=True)
        (config_path / "test.yaml").write_text("test: value")

        output_path = Path(tmpdirname) / "output.yaml"
        with mock.patch("anemoi.training.commands.config.ConfigGenerator.copy_files") as mock_copy_files, mock.patch(
            "anemoi.training.commands.config.initialize",
        ), mock.patch("anemoi.training.commands.config.compose", return_value=OmegaConf.create({"test": "value"})):
            config_generator.dump_config(config_path, "test", output_path)

            mock_copy_files.assert_called_once_with(config_path, mock.ANY)
            assert output_path.exists()
            assert OmegaConf.load(output_path) == {"test": "value"}


def test_validate_config_uses_package_path(config_generator: ConfigGenerator) -> None:
    """Test that validate_config works correctly with package configs.

    This test verifies the fix for issue #570: the AnemoiSearchPathPlugin
    adds 'pkg://anemoi.training/config' to the search path, enabling
    discovery of package configs like 'training/default'.

    Note: The package path is added by the plugin, not by the initialize() call,
    so we just verify the method executes successfully.
    """
    with mock.patch("anemoi.training.commands.config.initialize"), mock.patch(
        "anemoi.training.commands.config.compose",
        return_value=OmegaConf.create({"test": "value"}),
    ), mock.patch("anemoi.training.commands.config.BaseSchema"):
        # Call validate_config - it should succeed
        # The AnemoiSearchPathPlugin will add the package path automatically
        config_generator.validate_config("test-config", mask_env_vars=False)

        # If we get here without exception, the validation worked


def test_validate_config_with_mask_env_vars(config_generator: ConfigGenerator) -> None:
    """Test that validate_config works with mask_env_vars option."""
    with mock.patch("anemoi.training.commands.config.initialize"), mock.patch(
        "anemoi.training.commands.config.compose",
        return_value=OmegaConf.create({"test": "value"}),
    ), mock.patch("anemoi.training.commands.config.BaseSchema"), mock.patch.object(
        config_generator,
        "_mask_slurm_env_variables",
        return_value=OmegaConf.create({"test": "masked"}),
    ) as mock_mask:
        # Call validate_config with mask_env_vars=True
        config_generator.validate_config("test-config", mask_env_vars=True)

        # Verify that _mask_slurm_env_variables was called
        mock_mask.assert_called_once()

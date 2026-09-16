# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Tests for MetadataV0 robustness and null-handling.

Covers null-section handling where config keys exist but are set to None.
"""

from anemoi.metadata.versions.v0 import MetadataV0


class TestMetadataV0NullSectionRobustness:
    """MetadataV0 handles config sections set to None gracefully."""

    def test_null_data_does_not_break_get_timestep(self, sample_v0_dict):
        """config.data=None returns the default timestep without raising."""
        sample_v0_dict["config"]["data"] = None
        meta = MetadataV0.model_validate(sample_v0_dict)
        # Should return the default "6h"
        assert meta.get_timestep() == "6h"

    def test_null_data_does_not_break_get_data_frequency(self, sample_v0_dict):
        """config.data=None does not raise in get_data_frequency."""
        sample_v0_dict["config"]["data"] = None
        sample_v0_dict["dataset"]["frequency"] = "12h"
        meta = MetadataV0.model_validate(sample_v0_dict)
        # Should fall back to dataset.frequency
        assert meta.get_data_frequency() == "12h"

    def test_null_training_does_not_break_multistep_input(self, sample_v0_dict):
        """config.training=None returns default multistep_input=1."""
        sample_v0_dict["config"]["training"] = None
        meta = MetadataV0.model_validate(sample_v0_dict)
        # get_input_relative_date_indices should use default multistep_input=1
        assert meta.get_input_relative_date_indices() == [0]

    def test_null_training_does_not_break_get_tensor_shapes(self, sample_v0_dict):
        """config.training=None in get_tensor_shapes uses default."""
        sample_v0_dict["config"]["training"] = None
        meta = MetadataV0.model_validate(sample_v0_dict)
        shapes = meta.get_tensor_shapes()
        # input_timesteps should default to 1
        assert shapes["input_timesteps"] == 1

    def test_null_dataloader_returns_empty_dict(self, sample_v0_dict):
        """config.dataloader=None causes get_dataloader_config to return {}."""
        sample_v0_dict["config"]["dataloader"] = None
        meta = MetadataV0.model_validate(sample_v0_dict)
        result = meta.get_dataloader_config("training")
        assert result == {}

    def test_null_partition_value_returns_empty_dict(self, sample_v0_dict):
        """config.dataloader.training=None returns {} without raising."""
        sample_v0_dict["config"]["dataloader"] = {"training": None}
        meta = MetadataV0.model_validate(sample_v0_dict)
        result = meta.get_dataloader_config("training")
        assert result == {}

    def test_all_null_config_sections(self, sample_v0_dict):
        """Multiple null config sections do not break accessors."""
        sample_v0_dict["config"] = {
            "data": None,
            "training": None,
            "dataloader": None,
        }
        sample_v0_dict["dataset"]["frequency"] = "3h"
        meta = MetadataV0.model_validate(sample_v0_dict)
        # get_timestep returns default
        assert meta.get_timestep() == "6h"
        # get_data_frequency falls back to dataset.frequency
        assert meta.get_data_frequency() == "3h"
        # get_dataloader_config returns empty dict
        assert meta.get_dataloader_config() == {}
        # multistep-related methods use defaults
        assert meta.get_input_relative_date_indices() == [0]

    def test_null_data_does_not_break_get_variable_types(self, sample_v0_dict):
        """config.data=None in get_variable_types returns empty lists."""
        sample_v0_dict["config"]["data"] = None
        meta = MetadataV0.model_validate(sample_v0_dict)
        var_types = meta.get_variable_types()
        # forcing and diagnostic should be empty lists (default from .get())
        assert var_types["forcing"] == []
        assert var_types["diagnostic"] == []

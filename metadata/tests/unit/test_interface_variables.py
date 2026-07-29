# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Tests for the variable and environment methods on the Metadata interface."""

import sys

import pytest

from anemoi.metadata.interface import Metadata
from anemoi.metadata.versions.v1 import MetadataV1

# ---------------------------------------------------------------------------
# Variable listing and selection
# ---------------------------------------------------------------------------


class TestVariables:
    """Metadata provides variable listing and selection."""

    @pytest.fixture()
    def meta(self, sample_metadata_v1):
        """Return a Metadata instance wrapping sample_metadata_v1."""
        return Metadata(sample_metadata_v1)

    # -- variables property --------------------------------------------------

    def test_variables_returns_list(self, meta):
        """variables property returns a list."""
        assert isinstance(meta.variables, list)

    def test_variables_contains_t2m(self, meta):
        """variables includes '2t'."""
        assert "2t" in meta.variables

    def test_variables_contains_all_input_vars(self, meta):
        """variables contains all keys from data_indices.input."""
        expected = {"2t", "msl", "10u", "10v", "lsm"}
        assert set(meta.variables) == expected

    def test_variables_order_matches_input_indices(self, meta):
        """variables are ordered by their input index values."""
        indices = meta._raw.metadata_inference.datasets["era5_1deg"].data_indices.input
        expected_order = sorted(indices.keys(), key=lambda k: indices[k])
        assert meta.variables == expected_order

    def test_num_variables(self, meta):
        """num_variables matches len(variables)."""
        assert meta.num_variables == len(meta.variables)

    # -- variable_categories -------------------------------------------------

    def test_variable_categories_returns_dict(self, meta):
        """variable_categories() returns a dict."""
        cats = meta.variable_categories()
        assert isinstance(cats, dict)

    def test_variable_categories_has_expected_keys(self, meta):
        """variable_categories() has forcing, prognostic, diagnostic, target."""
        cats = meta.variable_categories()
        assert set(cats.keys()) == {"forcing", "prognostic", "diagnostic", "target"}

    def test_prognostic_variables_correct(self, meta):
        """prognostic category contains the expected variables."""
        cats = meta.variable_categories()
        assert set(cats["prognostic"]) == {"2t", "msl", "10u", "10v"}

    def test_forcing_variables_correct(self, meta):
        """forcing category contains lsm."""
        cats = meta.variable_categories()
        assert "lsm" in cats["forcing"]

    def test_diagnostic_is_empty(self, meta):
        """diagnostic category is empty in the sample data."""
        cats = meta.variable_categories()
        assert cats["diagnostic"] == []

    # -- select_variables: include by category -------------------------------

    def test_select_include_prognostic(self, meta):
        """select_variables(include=['prognostic']) returns prognostic vars."""
        result = meta.select_variables(include=["prognostic"])
        assert set(result) == {"2t", "msl", "10u", "10v"}

    def test_select_include_forcing(self, meta):
        """select_variables(include=['forcing']) returns forcing vars."""
        result = meta.select_variables(include=["forcing"])
        assert result == ["lsm"]

    def test_select_include_multiple_categories(self, meta):
        """select_variables(include=['prognostic', 'forcing']) combines both."""
        result = meta.select_variables(include=["prognostic", "forcing"])
        assert set(result) == {
            "2t",
            "msl",
            "10u",
            "10v",
            "lsm",
        }

    # -- select_variables: include by substring (backward compat) ------------

    def test_select_include_substring_t2m(self, meta):
        """select_variables(include=['2t']) filters by substring."""
        result = meta.select_variables(include=["2t"])
        assert result == ["2t"]

    def test_select_include_substring_partial(self, meta):
        """select_variables(include=['10']) matches 10u and 10v."""
        result = meta.select_variables(include=["10"])
        assert set(result) == {"10u", "10v"}

    def test_select_include_substring_no_match(self, meta):
        """select_variables(include=['solar']) returns empty when no match."""
        result = meta.select_variables(include=["solar"])
        assert result == []

    # -- select_variables: exclude by category -------------------------------

    def test_select_exclude_forcing(self, meta):
        """select_variables(exclude=['forcing']) removes forcing vars."""
        result = meta.select_variables(exclude=["forcing"])
        assert "lsm" not in result
        assert "2t" in result

    def test_select_exclude_prognostic(self, meta):
        """select_variables(exclude=['prognostic']) removes prognostic vars."""
        result = meta.select_variables(exclude=["prognostic"])
        assert "2t" not in result
        assert "lsm" in result

    # -- select_variables: combined include + exclude ------------------------

    def test_select_include_and_exclude(self, meta):
        """include and exclude can be combined."""
        result = meta.select_variables(
            include=["prognostic", "forcing"],
            exclude=["forcing"],
        )
        assert set(result) == {"2t", "msl", "10u", "10v"}

    # -- select_variables: no filters ----------------------------------------

    def test_select_no_filters_returns_all(self, meta):
        """select_variables() with no args returns all variables."""
        result = meta.select_variables()
        assert set(result) == set(meta.variables)

    # -- select_variables: empty result --------------------------------------

    def test_select_unknown_category_returns_empty(self, meta):
        """include with an unknown category/substring returns empty list."""
        result = meta.select_variables(include=["nonexistent_var_xyz"])
        assert result == []

    # -- select_variables: compound (+) expressions --------------------------

    def test_select_compound_prognostic_and_target(self, meta):
        """prognostic+target matches variables in BOTH categories."""
        result = meta.select_variables(include=["prognostic+target"])
        # All prognostic vars are also targets in the sample data.
        assert set(result) == {"2t", "msl", "10u", "10v"}

    def test_select_compound_prognostic_and_forcing_is_empty(self, meta):
        """prognostic+forcing matches no variable (disjoint sets)."""
        result = meta.select_variables(include=["prognostic+forcing"])
        assert result == []

    def test_select_compound_order_independent(self, meta):
        """target+prognostic gives the same result as prognostic+target."""
        fwd = meta.select_variables(include=["prognostic+target"])
        rev = meta.select_variables(include=["target+prognostic"])
        assert set(fwd) == set(rev)

    def test_select_compound_exclude(self, meta):
        """exclude with a compound expression removes the intersection."""
        # Exclude variables that are both prognostic AND target (all 4 of them).
        result = meta.select_variables(exclude=["prognostic+target"])
        assert set(result) == {"lsm"}

    def test_select_compound_mixed_with_simple(self, meta):
        """Mixing compound and simple expressions in include works."""
        # "prognostic+target" gives {2t, msl, 10u, 10v}; "forcing" adds lsm.
        result = meta.select_variables(include=["prognostic+target", "forcing"])
        assert set(result) == {"2t", "msl", "10u", "10v", "lsm"}

    def test_select_compound_unknown_part_raises(self, meta):
        """Compound expression with an unknown category raises ValueError."""
        with pytest.raises(ValueError, match="Unknown category"):
            meta.select_variables(include=["prognostic+nonexistent"])

    def test_select_single_unknown_token_is_substring_fallback(self, meta):
        """A single unknown token is treated as a substring, not an error."""
        # "10" is not a category name but matches "10u" and "10v" by substring.
        result = meta.select_variables(include=["10"])
        assert set(result) == {"10u", "10v"}

    def test_select_compound_exclude_multiple_expressions(self, meta):
        """Multiple exclude expressions (simple + compound) are combined."""
        # Exclude forcing (lsm) and prognostic+target (2t, msl, 10u, 10v).
        result = meta.select_variables(exclude=["forcing", "prognostic+target"])
        assert result == []


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Metadata validates the runtime environment against the checkpoint."""

    @pytest.fixture()
    def meta(self, sample_metadata_v1):
        """Return a Metadata instance wrapping sample_metadata_v1."""
        return Metadata(sample_metadata_v1)

    def test_validate_environment_returns_list(self, meta):
        """validate_environment() always returns a list."""
        result = meta.validate_environment()
        assert isinstance(result, list)

    def test_validate_environment_python_mismatch_produces_warning(self, sample_v1_dict):
        """Python version mismatch produces a warning string."""
        # Force a mismatch by setting an obviously wrong Python version.
        sample_v1_dict["environment"]["python_version"] = "0.0.1"
        meta = Metadata(MetadataV1.model_validate(sample_v1_dict))
        warnings = meta.validate_environment()
        assert len(warnings) >= 1
        assert any("Python" in w or "python" in w for w in warnings)

    def test_validate_environment_matching_python_no_warning(self, sample_v1_dict):
        """Matching Python version produces no warning."""
        current = f"{sys.version_info.major}.{sys.version_info.minor}" f".{sys.version_info.micro}"
        sample_v1_dict["environment"]["python_version"] = current
        meta = Metadata(MetadataV1.model_validate(sample_v1_dict))
        warnings = meta.validate_environment()
        assert warnings == []

    def test_validate_environment_empty_env_returns_empty(self, sample_v1_dict):
        """Empty environment dict returns an empty warnings list."""
        sample_v1_dict["environment"] = {}
        meta = Metadata(MetadataV1.model_validate(sample_v1_dict))
        assert meta.validate_environment() == []

    def test_validate_environment_no_python_key_no_warning(self, sample_v1_dict):
        """Environment dict without python_version key produces no warning."""
        sample_v1_dict["environment"] = {"cuda_version": "12.1"}
        meta = Metadata(MetadataV1.model_validate(sample_v1_dict))
        assert meta.validate_environment() == []

    def test_get_environment_info_returns_dict(self, meta):
        """get_environment_info() returns the environment dict."""
        info = meta.get_environment_info()
        assert isinstance(info, dict)

    def test_get_environment_info_contains_python_version(self, meta):
        """get_environment_info() contains python_version from sample data."""
        info = meta.get_environment_info()
        assert "python_version" in info

    def test_get_environment_info_empty_when_no_env(self, sample_v1_dict):
        """get_environment_info() returns {} when environment is empty."""
        sample_v1_dict["environment"] = {}
        meta = Metadata(MetadataV1.model_validate(sample_v1_dict))
        assert meta.get_environment_info() == {}


# ---------------------------------------------------------------------------
# Meta-test: no torch import in interface module
# ---------------------------------------------------------------------------


class TestNoTorchInInterface:
    """interface.py must not import torch (keeps the module lightweight)."""

    def test_no_torch_import_in_interface_source(self):
        """The interface module source does not contain 'import torch'."""
        import inspect

        from anemoi.metadata import interface as interface_module

        source = inspect.getsource(interface_module)
        assert "import torch" not in source
        assert "from torch" not in source

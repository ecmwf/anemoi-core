# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Tests for the variable and environment methods on the Metadata interface."""

import sys
from collections.abc import Callable
from typing import Any

import pytest

from anemoi.metadata.interface import DatasetData
from anemoi.metadata.interface import Metadata

# ---------------------------------------------------------------------------
# Variable listing and selection
# ---------------------------------------------------------------------------


class TestVariables:
    """DatasetView.data lists and selects variables.

    The categories and selection rules are covered in full in ``test_component_data.py``.
    """

    @pytest.fixture
    def meta(self, golden_loader: Callable[[str], dict[str, Any]]) -> Metadata:
        """Return a Metadata instance over the F6 golden file (one dataset, ``data``)."""
        return Metadata.from_dict(golden_loader("f6_main"))

    @pytest.fixture
    def data(self, meta: Metadata) -> DatasetData:
        """Return the data component of the only dataset."""
        return meta.dataset("data").data

    def test_variables_are_the_inputs_and_outputs_in_index_order(
        self,
        data: DatasetData,
        golden_loader: Callable[[str], dict[str, Any]],
    ) -> None:
        """variables() is the union of data_indices.input and output, inputs first, each ordered by its index."""
        indices = golden_loader("f6_main")["metadata_inference"]["data"]["data_indices"]
        inputs = sorted(indices["input"], key=lambda name: indices["input"][name])
        assert set(data.variables()) == set(indices["input"]) | set(indices["output"])
        assert data.variables()[: len(inputs)] == inputs
        assert data.variables()[-2:] == ["cp", "tp"]

    def test_shortcut_matches_data(self, meta: Metadata, data: DatasetData) -> None:
        """The single-dataset shortcut gives the same list as the data component."""
        assert meta.variables == data.variables()

    def test_select_variables_takes_keyword_include_and_exclude(self, data: DatasetData) -> None:
        """select_variables(include=..., exclude=...) is keyword-only and selects by category."""
        assert data.select_variables(include=["diagnostic"]) == ["cp", "tp"]
        assert set(data.select_variables(include=["prognostic", "diagnostic"], exclude=["diagnostic"])) == set(
            data.variable_categories()["prognostic"],
        )

    def test_select_unknown_name_raises(self, data: DatasetData) -> None:
        """Entries that are not categories raise ValueError."""
        with pytest.raises(ValueError, match="Unknown variable category"):
            data.select_variables(include=["2t"])


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Metadata validates the runtime environment against the checkpoint."""

    @pytest.fixture
    def raw(self, golden_loader: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
        """The F6 golden dict, which records no ``environment``."""
        raw = golden_loader("f6_main")
        assert "environment" not in raw
        return raw

    @pytest.mark.parametrize("environment", [None, {}, {"cuda_version": "12.1"}])
    def test_no_python_version_recorded_gives_no_warning(
        self,
        raw: dict[str, Any],
        environment: dict[str, Any] | None,
    ) -> None:
        """An absent or empty environment, or one without python_version, produces no warning."""
        if environment is not None:
            raw["environment"] = environment
        assert Metadata.from_dict(raw).validate_environment() == []

    def test_python_mismatch_produces_warning(self, raw: dict[str, Any]) -> None:
        """Python version mismatch produces a warning string."""
        raw["environment"] = {"python_version": "0.0.1"}
        warnings = Metadata.from_dict(raw).validate_environment()
        current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        assert warnings == [f"Python version mismatch: checkpoint=0.0.1, current={current}"]

    def test_matching_python_no_warning(self, raw: dict[str, Any]) -> None:
        """Matching Python version produces no warning."""
        raw["environment"] = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        }
        assert Metadata.from_dict(raw).validate_environment() == []

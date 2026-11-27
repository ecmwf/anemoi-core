# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Tests for the lenient reading of the ``metadata_inference`` block by the component adapters.

The adapters read the block as training writes it: scalars (``seed``,
``run_id``, ``task``, ``dataset_names``) next to one per-dataset dict keyed by
dataset name.  Reading is lenient: unknown keys are ignored and missing
optional keys read as empty values.  Values per format are pinned in
``test_component_data.py``, ``test_component_task.py`` and ``test_golden.py``;
the cases here are the edits of a golden file that those do not make.
"""

import copy
from collections.abc import Callable
from typing import Any

import pytest

from anemoi.metadata.base import CheckpointAdapter
from anemoi.metadata.components.data import DataAdapter
from anemoi.metadata.components.task import resolve_task_name


def data_of(raw: dict[str, Any], dataset: str = "data") -> DataAdapter:
    """Return the data adapter of *dataset* for the metadata dict *raw*."""
    return CheckpointAdapter(raw).data(dataset)


class TestDataAdapter:
    """The data adapters read dataset names, data indices, variable types and shapes."""

    @pytest.fixture
    def raw(self, golden_loader: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
        return golden_loader("f6_main")

    def test_grid_size_key_is_not_read_in_0_2(self, raw: dict[str, Any]) -> None:
        """The 0.2 layout has no ``grid_size`` key: the stored ``shapes.grid`` or the dataset shape is used."""
        block = raw["metadata_inference"]["data"]
        block["grid_size"] = 10
        block["shapes"]["grid"] = 123
        assert data_of(raw).grid_size() == 123
        del block["shapes"]
        assert data_of(raw).grid_size() == 40320
        del raw["dataset"]["data"]["shape"]
        assert data_of(raw).grid_size() is None

    def test_grid_size_null_in_1_0_without_dataset_shape_is_not_recorded(
        self,
        golden_loader: Callable[[str], dict[str, Any]],
    ) -> None:
        raw = golden_loader("f8_richer_batch")
        raw["metadata_inference"]["data"]["grid_size"] = None
        del raw["dataset"]["data"]["shape"]
        assert data_of(raw).grid_size() is None

    def test_listed_dataset_without_block_reads_empty(self, raw: dict[str, Any]) -> None:
        raw["metadata_inference"]["dataset_names"].append("ghost")
        checkpoint = CheckpointAdapter(raw)
        assert checkpoint.dataset_names == ["data", "ghost"]
        assert checkpoint.data("ghost").variable_indices() == {}

    def test_unknown_keys_ignored(self, raw: dict[str, Any]) -> None:
        raw["metadata_inference"]["data"]["data_indices"]["extra"] = 1
        raw["metadata_inference"]["data"]["new_block"] = {"a": 1}
        assert data_of(raw).variable_indices()["10u"] == 0

    def test_adapter_never_modifies_raw(self, raw: dict[str, Any]) -> None:
        before = copy.deepcopy(raw)
        adapter = data_of(raw)
        adapter.variable_indices()
        adapter.variable_types()
        adapter.variables_metadata()
        adapter.dataloader_config()
        assert raw == before

    def test_no_metadata_inference(self) -> None:
        checkpoint = CheckpointAdapter({"metadata_inference": {}})
        assert checkpoint.dataset_names == []
        with pytest.raises(KeyError):
            checkpoint.data("data")


class TestResolveTaskName:
    """The task name is read from metadata_inference, then from the top level."""

    def test_absent(self) -> None:
        assert resolve_task_name({}) is None
        assert resolve_task_name({"metadata_inference": {}}) is None

    def test_non_string_ignored(self) -> None:
        assert resolve_task_name({"task": {"name": "x"}}) is None

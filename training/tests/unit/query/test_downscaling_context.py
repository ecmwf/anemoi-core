# (C) Copyright 2026 Anemoi contributors.

"""Tests for leakage-free zero-lead query downscaling."""

from types import SimpleNamespace
from collections import Counter
from datetime import timedelta
from typing import NamedTuple

import numpy as np
import pytest

from anemoi.training.query.dataset import QueryDataset
from anemoi.training.tasks.query_forecasting import QueryForecasting


def _task(**overrides) -> QueryForecasting:
    values = {
        "lead_times": ["0h"],
        "input_history": "0h",
        "samples_per_epoch": 1,
        "reference_provenance": "IFS",
    }
    values.update(overrides)
    return QueryForecasting(**values)


def test_zero_lead_is_supported() -> None:
    task = _task()
    assert task.lead_times[0].total_seconds() == 0


def test_negative_lead_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        _task(lead_times=["-1h"])


def test_target_static_context_excludes_regional_dynamics() -> None:
    dataset = QueryDataset.__new__(QueryDataset)
    dataset.task = _task(global_context_sources=["IFS"], target_static_context=True)
    dynamic = SimpleNamespace(time_invariant=False)
    static = SimpleNamespace(time_invariant=True)
    dataset.fields_by_dataset = {"IFS": [dynamic, static], "MEPS": [dynamic, static]}

    assert dataset._source_fields("IFS") == [dynamic, static]
    assert dataset._source_fields("MEPS") == [static]


def test_residual_baseline_must_be_global_context() -> None:
    with pytest.raises(ValueError, match="global_context_sources"):
        _task(residual_baseline_source="IFS")

    task = _task(global_context_sources=["IFS"], residual_baseline_source="IFS")
    assert task.residual_baseline_source == "IFS"


class _Field(NamedTuple):
    provenance: str
    variable: str
    level: int


def test_provenance_variable_level_sampling_balances_each_hierarchy() -> None:
    dataset = QueryDataset.__new__(QueryDataset)
    dataset.task = _task(
        sampling_strategy="provenance_variable_level",
        provenance_weights={},
        variable_weights={},
    )
    fields = [
        _Field("A", "x", 1),
        _Field("A", "x", 2),
        _Field("A", "x", 3),
        _Field("A", "y", 1),
        _Field("B", "x", 1),
    ]
    dataset.target_fields = fields
    dataset.sampling_options = [(field, timedelta(0), None) for field in fields]
    dataset.sampling_weights = {"A": 1.0, "B": 1.0}

    rng = np.random.default_rng(20260925)
    draws = [dataset._sample_option(rng)[0] for _ in range(30_000)]
    provenance = Counter(field.provenance for field in draws)
    variables_a = Counter(field.variable for field in draws if field.provenance == "A")
    levels_ax = Counter(field.level for field in draws if (field.provenance, field.variable) == ("A", "x"))

    assert provenance["A"] / provenance["B"] == pytest.approx(1.0, rel=0.04)
    assert variables_a["x"] / variables_a["y"] == pytest.approx(1.0, rel=0.05)
    expected_level_count = sum(levels_ax.values()) / 3
    assert all(count == pytest.approx(expected_level_count, rel=0.08) for count in levels_ax.values())


def test_provenance_field_cycle_covers_every_field_without_replacement() -> None:
    dataset = QueryDataset.__new__(QueryDataset)
    dataset.task = _task(sampling_strategy="provenance_field_cycle")
    fields = [
        _Field("A", "x", 1),
        _Field("A", "x", 2),
        _Field("A", "y", 1),
        _Field("B", "x", 1),
        _Field("B", "y", 1),
    ]
    dataset.target_fields = fields
    dataset.sampling_options = [(field, timedelta(0), None) for field in fields]
    dataset.sampling_weights = {"A": 1.0, "B": 1.0}
    dataset.seed = 11
    dataset.epoch = 0
    dataset.length = 6
    dataset.sample_group_id = 0
    dataset.sample_group_count = 1

    rng = np.random.default_rng(0)
    draws = [dataset._sample_option(rng, index=index)[0] for index in range(6)]
    assert [field.provenance for field in draws] == ["A", "B", "A", "B", "A", "B"]
    assert len(set(draws[0::2])) == 3
    assert len(set(draws[1::2])) == 2


def test_full_valid_time_pass_is_shuffled_once_and_partitioned_over_model_groups() -> None:
    dataset = QueryDataset.__new__(QueryDataset)
    dataset.full_valid_time_pass = True
    dataset.full_samples = [(value, timedelta(0), None, value) for value in range(11)]
    dataset.sample_group_count = 4
    dataset.sample_group_id = 0
    dataset.seed = 17
    dataset.epoch = 3
    dataset._full_permutation_epoch = None
    dataset._full_permutation = None

    assert len(dataset) == 3
    selected = []
    for step in range(len(dataset)):
        for group in range(dataset.sample_group_count):
            dataset.sample_group_id = group
            *_, global_index = dataset._full_sample(step)
            selected.append(int(dataset._full_permutation[global_index % len(dataset.full_samples)]))

    assert len(set(selected[:11])) == 11
    assert selected[11] == selected[0]


def test_task_records_full_valid_time_pass() -> None:
    task = _task(query_all_fields_per_provenance=True, full_valid_time_pass=True)

    assert task.full_valid_time_pass is True

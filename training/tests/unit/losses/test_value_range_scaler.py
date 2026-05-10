# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses.scalers.value_range import TargetValueRangeScaler
from anemoi.training.losses.scalers.variable import GeneralVariableLossScaler
from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel


def _make_index_collection(normalizer_config: dict, *, include_target: bool = False) -> IndexCollection:
    config = DictConfig(
        {
            "forcing": [],
            "diagnostic": [],
            "target": ["aux_target"] if include_target else [],
            "processors": {
                "normalizer": {
                    "config": normalizer_config,
                },
            },
        },
    )
    name_to_index = {"refc": 0, "temp_500": 1}
    if include_target:
        name_to_index["aux_target"] = 2
    return IndexCollection(config, name_to_index)


def _make_model(multistep_input: int = 1, multistep_output: int = 1):
    return SimpleNamespace(
        n_step_input=multistep_input,
        config=SimpleNamespace(training=SimpleNamespace(multistep_output=multistep_output)),
    )


def test_target_value_range_scaler_infers_min_max_normalization() -> None:
    data_indices = _make_index_collection(
        {
            "default": "mean-std",
            "min-max": ["refc"],
            "none": [],
            "max": [],
            "std": [],
        }
    )
    statistics = {
        "mean": torch.tensor([20.0, 0.0], dtype=torch.float32),
        "stdev": torch.tensor([10.0, 1.0], dtype=torch.float32),
        "minimum": torch.tensor([0.0, 0.0], dtype=torch.float32),
        "maximum": torch.tensor([100.0, 1.0], dtype=torch.float32),
    }
    scaler = TargetValueRangeScaler(
        variable="refc",
        thresholds=[20.0],
        range_weight_factors=[1.0, 5.0],
        data_indices=data_indices,
        statistics=statistics,
        normalization=None,
    )

    batch = torch.zeros((1, 2, 1, 2, 2), dtype=torch.float32)
    batch[0, 1, 0, :, 0] = torch.tensor([0.10, 0.30], dtype=torch.float32)  # normalized min-max refc
    weights = scaler.on_batch_start(model=_make_model(), batch=batch)

    assert scaler.normalization == "min-max"
    assert weights is not None
    assert torch.allclose(weights[0, 0, :, 0], torch.tensor([1.0, 5.0]))


def test_target_value_range_scaler_raises_on_normalization_mismatch() -> None:
    data_indices = _make_index_collection(
        {
            "default": "mean-std",
            "min-max": ["refc"],
            "none": [],
            "max": [],
            "std": [],
        }
    )
    statistics = {
        "mean": torch.tensor([20.0, 0.0], dtype=torch.float32),
        "stdev": torch.tensor([10.0, 1.0], dtype=torch.float32),
        "minimum": torch.tensor([0.0, 0.0], dtype=torch.float32),
        "maximum": torch.tensor([100.0, 1.0], dtype=torch.float32),
    }

    with pytest.raises(ValueError, match="normalization mismatch"):
        TargetValueRangeScaler(
            variable="refc",
            thresholds=[20.0],
            range_weight_factors=[1.0, 5.0],
            data_indices=data_indices,
            statistics=statistics,
            normalization="mean-std",
        )


def test_target_value_range_scaler_rejects_ensemble_size_gt_one() -> None:
    data_indices = _make_index_collection(
        {
            "default": "none",
            "none": ["refc"],
        }
    )
    statistics = {
        "mean": torch.tensor([0.0, 0.0], dtype=torch.float32),
        "stdev": torch.tensor([1.0, 1.0], dtype=torch.float32),
        "minimum": torch.tensor([0.0, 0.0], dtype=torch.float32),
        "maximum": torch.tensor([1.0, 1.0], dtype=torch.float32),
    }
    scaler = TargetValueRangeScaler(
        variable="refc",
        thresholds=[0.5],
        range_weight_factors=[1.0, 2.0],
        data_indices=data_indices,
        statistics=statistics,
        normalization=None,
    )
    batch = torch.zeros((1, 2, 2, 3, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="ensemble size 1"):
        scaler.on_batch_start(model=_make_model(), batch=batch)


def test_general_variable_loss_scaler_initializes_non_model_output_positions() -> None:
    data_indices = _make_index_collection(
        {
            "default": "none",
            "none": ["refc", "temp_500", "aux_target"],
        },
        include_target=True,
    )
    metadata_extractor = ExtractVariableGroupAndLevel(
        variable_groups={"default": "sfc", "pl": {"param": ["temp"]}},
        metadata_variables={
            "refc": {"mars": {"param": "refc"}},
            "temp_500": {"mars": {"param": "temp", "levelist": 500}},
            "aux_target": {"mars": {"param": "aux_target"}},
        },
    )
    scaler = GeneralVariableLossScaler(
        data_indices=data_indices,
        metadata_extractor=metadata_extractor,
        weights=DictConfig({"default": 1.0, "refc": 20.0, "temp": 6.0}),
    )

    values = scaler.get_scaling_values()

    assert values.shape[0] == len(data_indices.data.output.full)
    assert torch.allclose(values, torch.tensor([20.0, 6.0, 1.0], dtype=torch.float32))

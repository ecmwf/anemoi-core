# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing.imputer import InputOnlyImputer
from anemoi.models.preprocessing.spatial_dropout import RandomSpatialDropout


@pytest.fixture()
def data_indices() -> IndexCollection:
    config = DictConfig(
        {
            "data": {
                "forcing": ["z", "q"],
                "diagnostic": ["other"],
            },
        },
    )
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4}
    return IndexCollection(data_config=config.data, name_to_index=name_to_index)


def _make_dropout(data_indices: IndexCollection, **overrides) -> RandomSpatialDropout:
    config = {"dropout_prob": 0.5, "multi_step": 2}
    config.update(overrides)
    return RandomSpatialDropout(config=DictConfig(config), data_indices=data_indices)


def test_spatial_dropout_only_input_steps(data_indices) -> None:
    torch.manual_seed(0)
    dropout = _make_dropout(data_indices, dropout_prob=1.0)
    x = torch.ones(3, 5, 10, 5)  # batch, time, grid, vars — 3 target steps
    out = dropout.transform(x, in_place=False)
    # non-forcing variables (x, y, other) dropped everywhere in the first 2 steps
    non_forcing = [0, 1, 4]
    assert torch.isnan(out[:, :2, :, non_forcing]).all()
    # target steps untouched
    assert not torch.isnan(out[:, 2:]).any()
    # forcing variables untouched by default
    assert not torch.isnan(out[..., [2, 3]]).any()


def test_spatial_dropout_fraction(data_indices) -> None:
    torch.manual_seed(0)
    dropout = _make_dropout(data_indices, dropout_prob=0.3)
    x = torch.ones(8, 4, 500, 5)
    out = dropout.transform(x, in_place=False)
    dropped_fraction = torch.isnan(out[:, :2, :, [0, 1, 4]]).float().mean().item()
    assert dropped_fraction == pytest.approx(0.3, abs=0.02)


def test_spatial_dropout_only_valid_cells(data_indices) -> None:
    torch.manual_seed(0)
    dropout = _make_dropout(data_indices, dropout_prob=1.0)
    x = torch.ones(1, 4, 10, 5)
    x[0, 0, :5, 0] = torch.nan  # already-missing points
    out = dropout.transform(x, in_place=False)
    # originally-NaN cells stay NaN; the mask never "un-drops" them
    assert torch.isnan(out[0, 0, :5, 0]).all()


def test_spatial_dropout_explicit_variables(data_indices) -> None:
    torch.manual_seed(0)
    dropout = _make_dropout(data_indices, dropout_prob=1.0, dropout_variables=["y", "z"])
    x = torch.ones(2, 4, 10, 5)
    out = dropout.transform(x, in_place=False)
    assert torch.isnan(out[:, :2, :, [1, 2]]).all()  # y and explicitly-listed forcing z
    assert not torch.isnan(out[..., [0, 3, 4]]).any()


def test_spatial_dropout_inference_noop(data_indices) -> None:
    dropout = _make_dropout(data_indices, dropout_prob=1.0)
    x = torch.ones(2, 2, 10, 5)  # time dim == multi_step -> inference-like
    out = dropout.transform(x, in_place=False)
    assert not torch.isnan(out).any()


def test_spatial_dropout_zero_prob_noop(data_indices) -> None:
    dropout = _make_dropout(data_indices, dropout_prob=0.0)
    x = torch.ones(2, 4, 10, 5)
    out = dropout.transform(x, in_place=False)
    assert not torch.isnan(out).any()


def test_spatial_dropout_inverse_noop(data_indices) -> None:
    dropout = _make_dropout(data_indices, dropout_prob=1.0)
    x = torch.randn(2, 4, 10, 5)
    assert torch.equal(dropout.inverse_transform(x.clone(), in_place=False), x)


def test_spatial_dropout_invalid_prob(data_indices) -> None:
    with pytest.raises(ValueError, match="dropout_prob"):
        _make_dropout(data_indices, dropout_prob=1.5)


# ── InputOnlyImputer ──────────────────────────────────────────────────────


@pytest.fixture()
def input_only_imputer_setup() -> tuple[InputOnlyImputer, IndexCollection]:
    config = DictConfig(
        {
            "data": {
                "imputer": {
                    "default": 0.0,
                    "multi_step": 2,
                    "none": ["other"],
                },
                "forcing": ["z", "q"],
                "diagnostic": ["other"],
                "corrector": ["c"],
            },
        },
    )
    statistics = {
        "mean": np.array([1.0, 2.0, 3.0, 4.5, 3.0, 1.0]),
    }
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4, "c": 5}
    data_indices = IndexCollection(data_config=config.data, name_to_index=name_to_index)
    imputer = InputOnlyImputer(config=config.data.imputer, data_indices=data_indices, statistics=statistics)
    return imputer, data_indices


def test_input_only_imputer_training(input_only_imputer_setup) -> None:
    imputer, _ = input_only_imputer_setup
    x = torch.full((2, 5, 10, 6), torch.nan)

    out = imputer.transform(x, in_place=False)

    # prognostic variables (x, y): imputed with 0.0 in the first multi_step steps only
    assert (out[:, :2, :, [0, 1]] == 0.0).all()
    assert torch.isnan(out[:, 2:, :, [0, 1]]).all()  # target NaNs preserved
    # forcing (z, q) and corrector (c) imputed across ALL timesteps
    assert (out[..., [2, 3, 5]] == 0.0).all()
    # 'none' variable untouched everywhere
    assert torch.isnan(out[..., 4]).all()


def test_input_only_imputer_inference_width(input_only_imputer_setup) -> None:
    imputer, data_indices = input_only_imputer_setup
    n_model_input = len(data_indices.model.input.name_to_index)
    x = torch.full((2, 2, 10, n_model_input), torch.nan)
    out = imputer.transform(x, in_place=False)
    # inference tensors carry only input steps and only model-input variables
    # (the diagnostic 'none' variable is absent): everything is filled
    assert (out == 0.0).all()


def test_input_only_imputer_rejects_output_tensor(input_only_imputer_setup) -> None:
    imputer, _ = input_only_imputer_setup
    with pytest.raises(ValueError, match="ONLY be used on inputs"):
        imputer.transform(torch.zeros(2, 2, 10, 3))


def test_input_only_imputer_skip_imputation(input_only_imputer_setup) -> None:
    imputer, _ = input_only_imputer_setup
    x = torch.full((2, 5, 10, 6), torch.nan)
    out = imputer(x, in_place=False, skip_imputation=True)
    assert torch.isnan(out).all()


def test_input_only_imputer_statistic_value() -> None:
    config = DictConfig(
        {
            "data": {
                "imputer": {"default": "none", "mean": ["x"], "multi_step": 2},
                "forcing": [],
                "diagnostic": [],
            },
        },
    )
    statistics = {"mean": np.array([7.5, 1.0])}
    data_indices = IndexCollection(data_config=config.data, name_to_index={"x": 0, "y": 1})
    imputer = InputOnlyImputer(config=config.data.imputer, data_indices=data_indices, statistics=statistics)
    x = torch.full((1, 3, 4, 2), torch.nan)
    out = imputer.transform(x, in_place=False)
    assert (out[:, :2, :, 0] == 7.5).all()
    assert torch.isnan(out[:, 2:, :, 0]).all()
    assert torch.isnan(out[..., 1]).all()


def test_processors_nan_check_relaxed(input_only_imputer_setup) -> None:
    """A processor chain containing an InputOnlyImputer tolerates target-step NaNs."""
    imputer, _ = input_only_imputer_setup
    processors = Processors([["imputer", imputer]])
    x = torch.full((2, 5, 10, 6), torch.nan)
    out = processors(x, in_place=False)  # must not raise despite remaining NaNs
    assert torch.isnan(out[:, 2:, :, [0, 1]]).all()


def test_processors_nan_check_still_enforced(data_indices) -> None:
    """Without an allows_output_nans processor, the no-NaN assert still fires."""
    dropout = _make_dropout(data_indices, dropout_prob=1.0)
    processors = Processors([["spatial_dropout", dropout]])
    x = torch.ones(2, 4, 10, 5)
    with pytest.raises(AssertionError, match="NaNs"):
        processors(x, in_place=False)

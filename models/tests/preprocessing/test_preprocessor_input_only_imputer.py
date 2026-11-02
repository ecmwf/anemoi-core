# (C) Copyright 2026- Anemoi contributors.
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

from anemoi.models.data import TensorLayout
from anemoi.models.data.sources import GriddedSource
from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing.imputer import InputOnlyImputer
from anemoi.models.schemas.data_processor import InputOnlyImputerSchema

VARIABLES = ["x", "y", "z", "w"]
STATISTICS = {
    "mean": np.array([1.0, 2.0, 3.0, 4.0]),
    "stdev": np.ones(4),
    "minimum": np.array([-1.0, -2.0, -3.0, -4.0]),
    "maximum": np.array([10.0, 20.0, 30.0, 40.0]),
}


def _source(data: torch.Tensor) -> GriddedSource:
    return GriddedSource(
        name="data",
        variables=VARIABLES,
        layout=TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4),
        statistics=STATISTICS,
        data=data,
        coordinates=torch.zeros(data.shape[3], 2),
    )


def _imputer(**config) -> Processors:
    config = {"default": "none", "mean": ["x"], "maximum": ["y"], 0.0: ["z"], **config}
    return Processors([["imputer", InputOnlyImputer(config=DictConfig(config))]])


def _nan_input() -> torch.Tensor:
    data = torch.ones(2, 3, 1, 5, 4)
    data[:, :, :, :2] = torch.nan
    return data


def test_statistics_and_constants_in_one_configuration() -> None:
    out = _imputer()(_source(_nan_input()), in_place=False).data
    missing = out[:, :, :, :2]
    assert torch.all(missing[..., 0] == 1.0)  # mean of x
    assert torch.all(missing[..., 1] == 20.0)  # maximum of y
    assert torch.all(missing[..., 2] == 0.0)  # constant
    assert torch.isnan(missing[..., 3]).all()  # "none" (default)
    assert torch.all(out[:, :, :, 2:] == 1.0)  # present values untouched


def test_targets_keep_their_missing_values() -> None:
    out = _imputer()(_source(_nan_input()), in_place=False, skip_imputation=True).data
    assert torch.isnan(out[:, :, :, :2]).all()


def test_multi_step_is_accepted_and_every_step_imputed() -> None:
    out = _imputer(multi_step=1)(_source(_nan_input()), in_place=False).data
    assert not torch.isnan(out[..., :3]).any()


def test_schema() -> None:
    InputOnlyImputerSchema.model_validate({"default": "none", "multi_step": 2, "mean": ["x"], 0.0: ["z"]})
    with pytest.raises(TypeError, match="multi_step"):
        InputOnlyImputerSchema.model_validate({"multi_step": 0})

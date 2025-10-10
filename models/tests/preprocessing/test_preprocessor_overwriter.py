# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing.overwriter import ZeroOverwriter


@pytest.fixture()
def zero_overwriter():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "zero_overwriter": {
                    "groups": [
                        {
                            "vars": ["tp_accum", "cp_accum"],
                            "time_index": [0],
                        }
                    ]
                },
                # training data categories
                "forcing": ["z", "tp_accum", "cp_accum", "q"],
                "diagnostic": ["other"],
            },
        },
    )
    # Dataset variable order (dataset space)
    name_to_index = {"z": 0, "tp_accum": 1, "cp_accum": 2, "q": 3, "other": 4}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    return ZeroOverwriter(config=config.data.zero_overwriter, data_indices=data_indices, statistics=None)


@pytest.fixture()
def training_input_data():
    # one sample, two time steps, two grid points, 5 dataset variables (includes diagnostic)
    base = torch.Tensor(
        [
            [
                [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]],  # t=0
                [[11.0, 12.0, 13.0, 14.0, 15.0], [16.0, 17.0, 18.0, 19.0, 20.0]],  # t=1
            ]
        ]
    )
    return base


@pytest.fixture()
def inference_input_data():
    # one sample, two time steps, two grid points, 4 model input variables (excludes diagnostic)
    base = torch.Tensor(
        [
            [
                [[1.0, 2.0, 3.0, 4.0], [6.0, 7.0, 8.0, 9.0]],  # t=0
                [[11.0, 12.0, 13.0, 14.0], [16.0, 17.0, 18.0, 19.0]],  # t=1
            ]
        ]
    )
    return base


def test_zero_overwriter_not_inplace_training(zero_overwriter, training_input_data) -> None:
    """ZeroOverwriter should not modify input when in_place=False (training shape)."""
    x = training_input_data.clone()
    x_old = x.clone()
    _ = zero_overwriter(x, in_place=False)
    assert torch.allclose(x, x_old, equal_nan=True)


def test_zero_overwriter_inplace_training(zero_overwriter, training_input_data) -> None:
    """ZeroOverwriter should modify input when in_place=True (training shape)."""
    x = training_input_data.clone()
    x_old = x.clone()
    out = zero_overwriter(x, in_place=True)
    assert not torch.allclose(x, x_old, equal_nan=True)
    assert torch.allclose(x, out, equal_nan=True)


def test_zero_overwriter_transform_training(zero_overwriter, training_input_data) -> None:
    """Check correct zeroing on training-shaped tensors (dataset variable dimension)."""
    x = training_input_data.clone()

    # Build expected by zeroing configured vars at t=0 using dataset (training) mapping
    expected = x.clone()
    train_map = zero_overwriter.data_indices.data.input.name_to_index  # dataset var -> dataset index
    idxs = [train_map["tp_accum"], train_map["cp_accum"]]
    expected[:, [0], ..., idxs] = 0

    transformed = zero_overwriter.transform(x, in_place=False)
    assert torch.allclose(transformed, expected, equal_nan=True)


def test_zero_overwriter_transform_inference(zero_overwriter, inference_input_data) -> None:
    """Check correct zeroing on inference-shaped tensors (model input variable dimension)."""
    x = inference_input_data.clone()

    # Build expected by zeroing configured vars at t=0 using model-input (inference) mapping
    expected = x.clone()
    infer_map = zero_overwriter.data_indices.model.input.name_to_index  # model var -> input index (0..n-1)
    idxs = [infer_map["tp_accum"], infer_map["cp_accum"]]
    expected[:, [0], ..., idxs] = 0

    transformed = zero_overwriter.transform(x, in_place=False)
    assert torch.allclose(transformed, expected, equal_nan=True)

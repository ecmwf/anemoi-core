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
                            # zero accumulative precipitation variables at t=0
                            "vars": ["tp_accum", "cp_accum"],
                            "time_indices": [0],
                        }
                    ]
                },
                # training data categories
                "forcing": ["z", "tp_accum", "cp_accum", "q"],
                "diagnostic": ["tp", "cp"],
            },
        },
    )
    # Dataset variable order (dataset space):
    name_to_index = {"tp": 0, "cp": 1, "z": 2, "tp_accum": 3, "cp_accum": 4, "q": 5}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    return ZeroOverwriter(config=config.data.zero_overwriter, data_indices=data_indices, statistics=None)


@pytest.fixture()
def training_input_data():
    # one sample, two time steps, two grid points, 6 dataset variables (tp, cp, z, tp_accum, cp_accum, q)
    base = torch.Tensor(
        [
            [
                # t=0
                [[101.0, 102.0, 1.0, 2.0, 3.0, 4.0], [201.0, 202.0, 6.0, 7.0, 8.0, 9.0]],
                # t=1
                [[103.0, 104.0, 11.0, 12.0, 13.0, 14.0], [203.0, 204.0, 16.0, 17.0, 18.0, 19.0]],
            ]
        ]
    )
    return base


@pytest.fixture()
def inference_input_data():
    # one sample, two time steps, two grid points, 4 model input variables (excludes diagnostics)
    # model inputs remain [z, tp_accum, cp_accum, q]
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
    max_diff = torch.nan_to_num(x - x_old).abs().max().item()
    assert torch.allclose(
        x, x_old, equal_nan=True
    ), f"ZeroOverwriter: in_place=False should not modify input; max_abs_diff={max_diff}, shape={tuple(x.shape)}"


def test_zero_overwriter_inplace_training(zero_overwriter, training_input_data) -> None:
    """ZeroOverwriter should modify input when in_place=True (training shape)."""
    x = training_input_data.clone()
    x_old = x.clone()
    out = zero_overwriter(x, in_place=True)
    max_diff_changed = torch.nan_to_num(x - x_old).abs().max().item()
    assert not torch.allclose(
        x, x_old, equal_nan=True
    ), f"in_place=True should modify input; tensors are still equal (max_abs_diff={max_diff_changed})"
    max_diff_out = torch.nan_to_num(x - out).abs().max().item()
    assert torch.allclose(
        x, out, equal_nan=True
    ), f"ZeroOverwriter: Output should alias modified input when in_place=True; max_abs_diff={max_diff_out}, shape={tuple(x.shape)}"


def test_zero_overwriter_transform_training(zero_overwriter, training_input_data) -> None:
    """Check correct zeroing on training-shaped tensors (dataset variable dimension)."""
    x = training_input_data.clone()

    # Build expected by zeroing configured vars at t=0 using dataset (training) mapping
    expected = x.clone()
    train_map = zero_overwriter.data_indices.data.input.name_to_index  # dataset var -> dataset index
    idxs = [train_map["tp_accum"], train_map["cp_accum"]]
    expected[:, [0], ..., idxs] = 0

    transformed = zero_overwriter.transform(x, in_place=False)
    max_diff = torch.nan_to_num(transformed - expected).abs().max().item()
    assert torch.allclose(
        transformed, expected, equal_nan=True
    ), f"ZeroOverwriter: Training transform mismatch; expected zeros at t=0 for ['tp_accum','cp_accum']; max_abs_diff={max_diff}"


def test_zero_overwriter_transform_inference(zero_overwriter, inference_input_data) -> None:
    """Check correct zeroing on inference-shaped tensors (model input variable dimension)."""
    x = inference_input_data.clone()

    # Build expected by zeroing configured vars at t=0 using model-input (inference) mapping
    expected = x.clone()
    infer_map = zero_overwriter.data_indices.model.input.name_to_index  # model var -> input index (0..n-1)
    idxs = [infer_map["tp_accum"], infer_map["cp_accum"]]
    expected[:, [0], ..., idxs] = 0

    transformed = zero_overwriter.transform(x, in_place=False)
    max_diff = torch.nan_to_num(transformed - expected).abs().max().item()
    assert torch.allclose(
        transformed, expected, equal_nan=True
    ), f"ZeroOverwriter: Inference transform mismatch; expected zeros at t=0 for ['tp_accum','cp_accum']; max_abs_diff={max_diff}"

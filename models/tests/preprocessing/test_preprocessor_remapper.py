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
from anemoi.models.preprocessing.remapper import Remapper


@pytest.fixture()
def input_remapper_1d():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "remapper": {
                    "log1p": "d",
                    "sqrt": "q",
                },
                "forcing": ["z", "q"],
                "diagnostic": ["other"],
            },
        },
    )
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "d": 4, "other": 5}
    data_indices = IndexCollection(data_config=config.data, name_to_index=name_to_index)
    return Remapper(config=config.data.remapper, data_indices=data_indices)


def test_monomap_transform(input_remapper_1d) -> None:
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 150.0, 5.0], [6.0, 7.0, 8.0, 9.0, 201.0, 10.0]])
    expected_output = torch.Tensor(
        [
            [1.0, 2.0, 3.0, np.sqrt(4.0), np.log1p(150.0), 5.0],
            [6.0, 7.0, 8.0, np.sqrt(9.0), np.log1p(201.0), 10.0],
        ]
    )
    assert torch.allclose(input_remapper_1d.transform(x, in_place=False), expected_output)
    # inference mode (without prognostic variables)
    assert torch.allclose(
        input_remapper_1d.transform(
            x[..., input_remapper_1d.data_indices.data.todict()["input"]["full"]], in_place=False
        ),
        expected_output[..., input_remapper_1d.data_indices.data.todict()["input"]["full"]],
    )
    # this one actually changes the values in x so need to be last
    assert torch.allclose(input_remapper_1d.transform(x), expected_output)


def test_monomap_inverse_transform(input_remapper_1d) -> None:
    expected_output = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 150.0, 5.0], [6.0, 7.0, 8.0, 9.0, 201.0, 10.0]])
    y = torch.Tensor(
        [
            [1.0, 2.0, 3.0, np.sqrt(4.0), np.log1p(150.0), 5.0],
            [6.0, 7.0, 8.0, np.sqrt(9.0), np.log1p(201.0), 10.0],
        ]
    )
    assert torch.allclose(input_remapper_1d.inverse_transform(y, in_place=False), expected_output)
    # inference mode (without prognostic variables)
    assert torch.allclose(
        input_remapper_1d.inverse_transform(
            y[..., input_remapper_1d.data_indices.data.todict()["output"]["full"]], in_place=False
        ),
        expected_output[..., input_remapper_1d.data_indices.data.todict()["output"]["full"]],
    )


def test_unsupported_remapper():
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "remapper": {"log1p": "q", "cos": "d"},
                "forcing": [],
                "diagnostic": [],
            },
        }
    )
    name_to_index = {"x": 0, "y": 1, "q": 2, "d": 3}
    data_indices = IndexCollection(data_config=config.data, name_to_index=name_to_index)

    with pytest.raises(KeyError):
        Remapper(
            config=config.data.remapper,
            data_indices=data_indices,
        )


def test_transform_clips_to_domain() -> None:
    """``clip_to_domain`` clips values outside a remapping's domain, as the inverse remappings clip outputs."""
    config = DictConfig(
        {
            "remapper": {
                "sqrt": "q",
                "log1p": "d",
                "atanh": "a",
                "boxcox": "b",
                "asinh": "s",
                "method_kwargs": {"boxcox": {"lambd": 0.0}},
            },
            "forcing": [],
            "diagnostic": [],
        },
    )
    name_to_index = {"q": 0, "d": 1, "a": 2, "b": 3, "s": 4}
    remapper = Remapper(
        config=config.remapper, data_indices=IndexCollection(data_config=config, name_to_index=name_to_index)
    )
    x = torch.tensor([[-4.0, -3.0, 1.5, -1.0, -2.0], [4.0, 1.0, 0.5, 1.0, 2.0]])

    with pytest.raises(AssertionError, match="x >= 0"):
        remapper.transform(x, in_place=False, name_to_index=name_to_index)
    clipped = remapper.transform(x, in_place=False, name_to_index=name_to_index, clip_to_domain=True)

    assert torch.isfinite(clipped).all()
    # In the domain: unchanged remapping.
    expected = remapper.transform(x[1:], in_place=False, name_to_index=name_to_index)
    torch.testing.assert_close(clipped[1:], expected)
    # Outside: the remapping of the domain bound, so the inverse gives the bound back.
    torch.testing.assert_close(clipped[0, 0], torch.tensor(0.0))
    torch.testing.assert_close(clipped[0, 2], remapper.transform(torch.ones(1, 5), name_to_index=name_to_index)[0, 2])
    torch.testing.assert_close(clipped[0, 4], torch.asinh(torch.tensor(-2.0)))  # unbounded: not clipped
    back = remapper.inverse_transform(clipped[:1].clone(), name_to_index=name_to_index)
    torch.testing.assert_close(back[0, :4], torch.tensor([0.0, -1.0, 1.0, 0.0]), atol=1e-6, rtol=0)

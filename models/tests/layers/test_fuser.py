# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from anemoi.models.layers.fuser import LinearFuser


def test_linear_fuser_projects_each_dataset_to_num_channels() -> None:
    fuser = LinearFuser(num_channels=8, in_channels={"era5": 3, "cerra": 5})

    assert fuser.num_channels == 8
    assert fuser(torch.randn(4, 3), "era5").shape == (4, 8)
    assert fuser(torch.randn(2, 6, 5), dataset_name="cerra").shape == (2, 6, 8)
    # Indexing by dataset name is the call contract the model uses (and old pickled ModuleDicts support).
    torch.testing.assert_close(fuser["era5"](torch.ones(1, 3)), fuser(torch.ones(1, 3), "era5"))


def test_linear_fuser_state_dict_keys_are_keyed_by_dataset() -> None:
    fuser = LinearFuser(num_channels=4, in_channels={"era5": 3, "cerra": 5})

    assert sorted(fuser.state_dict()) == ["cerra.bias", "cerra.weight", "era5.bias", "era5.weight"]
    assert fuser.state_dict()["cerra.weight"].shape == (4, 5)


@pytest.mark.parametrize("num_channels", [0, -3, 2.5, True, None])
def test_linear_fuser_rejects_invalid_num_channels(num_channels) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        LinearFuser(num_channels=num_channels, in_channels={"era5": 3})


def test_linear_fuser_rejects_empty_in_channels() -> None:
    with pytest.raises(ValueError, match="at least one source dataset"):
        LinearFuser(num_channels=4, in_channels={})


def test_linear_fuser_honours_layer_kernels() -> None:
    fuser = LinearFuser(
        num_channels=4,
        in_channels={"era5": 3},
        layer_kernels={"Linear": {"_target_": "torch.nn.Linear", "bias": False}},
    )

    assert fuser["era5"].bias is None


def test_linear_fuser_instantiates_from_config() -> None:
    config = OmegaConf.create({"_target_": "anemoi.models.layers.fuser.LinearFuser", "num_channels": 6})

    fuser = instantiate(config, _recursive_=False, in_channels={"era5": 3}, layer_kernels=None)

    assert isinstance(fuser, LinearFuser)
    assert fuser(torch.randn(2, 3), "era5").shape == (2, 6)

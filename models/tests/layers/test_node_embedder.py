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

from anemoi.models.layers.node_embedder import LinearNodeEmbedder
from anemoi.models.layers.node_embedder import NoOpNodeEmbedder
from anemoi.models.layers.node_embedder import is_noop_node_embedder_config


def test_linear_node_embedder_projects_each_dataset_to_num_channels() -> None:
    node_embedder = LinearNodeEmbedder(num_channels=8, in_channels={"era5": 3, "cerra": 5})

    assert node_embedder.num_channels == 8
    assert node_embedder(torch.randn(4, 3), "era5").shape == (4, 8)
    assert node_embedder(torch.randn(2, 6, 5), dataset_name="cerra").shape == (2, 6, 8)
    # Indexing by dataset name is the call contract the model uses (and old pickled ModuleDicts support).
    torch.testing.assert_close(node_embedder["era5"](torch.ones(1, 3)), node_embedder(torch.ones(1, 3), "era5"))


def test_linear_node_embedder_state_dict_keys_are_keyed_by_dataset() -> None:
    node_embedder = LinearNodeEmbedder(num_channels=4, in_channels={"era5": 3, "cerra": 5})

    assert sorted(node_embedder.state_dict()) == ["cerra.bias", "cerra.weight", "era5.bias", "era5.weight"]
    assert node_embedder.state_dict()["cerra.weight"].shape == (4, 5)


@pytest.mark.parametrize("num_channels", [0, -3, 2.5, True, None])
def test_linear_node_embedder_rejects_invalid_num_channels(num_channels) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        LinearNodeEmbedder(num_channels=num_channels, in_channels={"era5": 3})


def test_linear_node_embedder_rejects_empty_in_channels() -> None:
    with pytest.raises(ValueError, match="at least one source dataset"):
        LinearNodeEmbedder(num_channels=4, in_channels={})


def test_linear_node_embedder_honours_layer_kernels() -> None:
    node_embedder = LinearNodeEmbedder(
        num_channels=4,
        in_channels={"era5": 3},
        layer_kernels={"Linear": {"_target_": "torch.nn.Linear", "bias": False}},
    )

    assert node_embedder["era5"].bias is None


def test_linear_node_embedder_instantiates_from_config() -> None:
    config = OmegaConf.create({"_target_": "anemoi.models.layers.node_embedder.LinearNodeEmbedder", "num_channels": 6})

    node_embedder = instantiate(config, _recursive_=False, in_channels={"era5": 3}, layer_kernels=None)

    assert isinstance(node_embedder, LinearNodeEmbedder)
    assert node_embedder(torch.randn(2, 3), "era5").shape == (2, 6)


def test_noop_node_embedder_passes_the_input_through() -> None:
    node_embedder = NoOpNodeEmbedder(in_channels={"era5": 5, "cerra": 5})
    x = torch.randn(3, 5)

    assert node_embedder.num_channels == 5
    assert node_embedder(x, "cerra") is x
    assert not list(node_embedder.parameters())
    assert node_embedder.state_dict() == {}


def test_noop_node_embedder_rejects_differing_input_channels() -> None:
    with pytest.raises(ValueError, match="same number of input channels"):
        NoOpNodeEmbedder(in_channels={"era5": 3, "cerra": 5})


def test_noop_node_embedder_rejects_empty_in_channels() -> None:
    with pytest.raises(ValueError, match="at least one source dataset"):
        NoOpNodeEmbedder(in_channels={})


def test_noop_node_embedder_instantiates_from_config() -> None:
    config = OmegaConf.create({"_target_": "anemoi.models.layers.node_embedder.NoOpNodeEmbedder"})

    node_embedder = instantiate(config, _recursive_=False, in_channels={"era5": 3}, layer_kernels=None)

    assert isinstance(node_embedder, NoOpNodeEmbedder)
    assert node_embedder.num_channels == 3


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (None, True),
        ({"_target_": "anemoi.models.layers.node_embedder.NoOpNodeEmbedder"}, True),
        ({"_target_": "anemoi.models.layers.node_embedder.LinearNodeEmbedder", "num_channels": 4}, False),
    ],
)
def test_is_noop_node_embedder_config(config: dict | None, expected: bool) -> None:
    assert is_noop_node_embedder_config(config) is expected

# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from anemoi.models.utils import InstantiationError
from anemoi.models.utils import current_backend
from anemoi.models.utils import get_class
from anemoi.models.utils import get_object
from anemoi.models.utils import instantiate
from anemoi.models.utils import instantiation_backend
from anemoi.models.utils import set_instantiation_backend
from anemoi.utils.config import DotDict


@pytest.fixture
def native():
    """Run the body with the native backend active, restoring state afterwards."""
    with instantiation_backend("native"):
        yield


def test_default_backend_is_hydra():
    assert current_backend() == "hydra"


def test_dotted_path_resolution():
    assert get_object("torch.nn.Linear") is torch.nn.Linear
    assert get_class("torch.nn.Linear") is torch.nn.Linear
    # nested attribute resolution
    assert get_object("torch.nn.functional.relu") is torch.nn.functional.relu


def test_get_class_rejects_non_class():
    with pytest.raises(InstantiationError):
        get_class("torch.nn.functional.relu")


def test_get_object_bad_path():
    with pytest.raises(InstantiationError):
        get_object("does.not.exist.Thing")


def test_native_basic(native):
    obj = instantiate({"_target_": "torch.nn.Linear", "in_features": 3, "out_features": 4})
    assert isinstance(obj, torch.nn.Linear)
    assert obj.in_features == 3
    assert obj.out_features == 4


def test_native_call_kwargs_override_config(native):
    obj = instantiate({"_target_": "torch.nn.Linear", "in_features": 3, "out_features": 4}, out_features=8)
    assert obj.out_features == 8


def test_native_partial(native):
    factory = instantiate({"_target_": "torch.nn.Linear", "bias": False}, _partial_=True)
    module = factory(2, 5)
    assert isinstance(module, torch.nn.Linear)
    assert module.bias is None


def test_native_partial_from_config_flag(native):
    factory = instantiate({"_target_": "torch.nn.Linear", "_partial_": True, "bias": False})
    assert callable(factory)
    assert isinstance(factory(2, 5), torch.nn.Linear)


def test_native_recursive_args(native):
    cfg = DotDict(
        {
            "_target_": "torch.nn.Sequential",
            "_args_": [
                {"_target_": "torch.nn.Linear", "in_features": 4, "out_features": 4},
                {"_target_": "torch.nn.ReLU"},
            ],
        }
    )
    seq = instantiate(cfg)
    assert isinstance(seq, torch.nn.Sequential)
    assert len(seq) == 2
    assert isinstance(seq[0], torch.nn.Linear)


def test_native_non_recursive_passes_config_through(native):
    out = instantiate(
        {"_target_": "builtins.dict", "_recursive_": False, "child": {"_target_": "torch.nn.ReLU"}},
    )
    # With _recursive_=False the nested config is forwarded untouched.
    assert out["child"] == {"_target_": "torch.nn.ReLU"}


def test_native_none_and_primitive(native):
    assert instantiate(None) is None
    assert instantiate(5) == 5


def test_native_error_wrapping(native):
    with pytest.raises(InstantiationError):
        instantiate({"_target_": "nonexistent.module.Thing"})


def test_backend_switch_set_and_reset():
    try:
        set_instantiation_backend("native")
        assert current_backend() == "native"
        obj = instantiate({"_target_": "torch.nn.ReLU"})
        assert isinstance(obj, torch.nn.ReLU)
    finally:
        set_instantiation_backend(None)
    assert current_backend() == "hydra"


def test_backend_switch_rejects_unknown():
    with pytest.raises(ValueError):
        set_instantiation_backend("nonsense")
    with pytest.raises(ValueError):
        with instantiation_backend("nonsense"):
            pass


@pytest.mark.parametrize(
    "config",
    [
        {"_target_": "torch.nn.Linear", "in_features": 6, "out_features": 7, "bias": False},
        {"_target_": "torch.nn.GELU"},
    ],
)
def test_parity_with_hydra(config):
    """Native and Hydra backends must produce equivalent objects."""
    pytest.importorskip("hydra")

    set_instantiation_backend("hydra")
    try:
        hydra_obj = instantiate(dict(config))
    finally:
        set_instantiation_backend(None)

    with instantiation_backend("native"):
        native_obj = instantiate(dict(config))

    assert type(hydra_obj) is type(native_obj)
    if isinstance(hydra_obj, torch.nn.Linear):
        assert hydra_obj.in_features == native_obj.in_features
        assert hydra_obj.out_features == native_obj.out_features
        assert (hydra_obj.bias is None) == (native_obj.bias is None)

# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.training.diagnostics.mlflow.utils import clean_config_params
from anemoi.training.diagnostics.mlflow.utils import expand_iterables


def test_clean_config_params() -> None:
    params = {
        "config.dataset.format": None,
        "config.model.num_channels": None,
        "model.num_channels": None,
        "data.frequency": None,
        "diagnostics.plot": None,
        "system.hardware.num_gpus": None,
        "metadata.config.dataset": None,
        "metadata.dataset.sources/1.specific.forward.forward.attrs.variables_metadata.z_500.mars.expver": None,
        "metadata.dataset.specific.forward.forward.attrs.variables_metadata.z_500.mars.expver": None,
        "config.data.normalizer.default": None,
        "config.data.normalizer.std": None,
        "config.data.normalizer.min-max": None,
        "config.data.normalizer.max": None,
    }

    cleaned = clean_config_params(params)
    result = {
        "config.dataset.format": None,
        "config.model.num_channels": None,
        "config.data.normalizer.default": None,
        "config.data.normalizer.std": None,
        "config.data.normalizer.min-max": None,
        "config.data.normalizer.max": None,
    }
    assert cleaned == result


def test_expand_iterables_single_iterable() -> None:
    # Test case with a single iterable
    dictionary = {"a": ["a", "b", "c"]}
    expanded = expand_iterables(dictionary)
    assert expanded == {"a": ["a", "b", "c"]}


def test_expand_iterables_with_nested_dict() -> None:
    dictionary = {"a": {"b": ["a", "b", "c"]}}
    expanded = expand_iterables(dictionary)
    assert expanded == {"a": {"b": ["a", "b", "c"]}}


def test_expand_iterables_with_nested_list() -> None:
    dictionary = {"a": [[0, 1, 2], "b", "c"]}
    expanded = expand_iterables(dictionary)
    assert expanded == {
        "a": {0: [0, 1, 2], 1: "b", 2: "c"},
        "a.length": 3,
        "a.all": [[0, 1, 2], "b", "c"],
    }


def test_expand_iterables_with_omegaconf() -> None:
    from omegaconf.dictconfig import DictConfig
    from omegaconf.listconfig import ListConfig

    dictionary = DictConfig({"a": ListConfig([ListConfig([0, 1, 2]), "b", "c"])})
    expanded = expand_iterables(dictionary)
    assert expanded == {
        "a": {0: [0, 1, 2], 1: "b", 2: "c"},
        "a.all": [[0, 1, 2], "b", "c"],
        "a.length": 3,
    }
    # Note that ListConfig and plain list are comparible and that ListConfig (and lists) of primitives are preserved
    assert isinstance(expanded["a"][0], ListConfig)

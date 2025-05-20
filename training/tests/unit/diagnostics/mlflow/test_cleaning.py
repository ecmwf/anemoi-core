# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.training.diagnostics.mlflow.utils import clean_config_params


def test_clean_config_params() -> None:
    params = {
        "config.dataset.format": None,
        "config.model.num_channels": None,
        "model.num_channels": None,
        "data.frequency": None,
        "diagnostics.plot": None,
        "hardware.num_gpus": None,
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

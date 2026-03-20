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
from omegaconf import OmegaConf

from anemoi.models.models.base import _get_backbone_config
from anemoi.models.models.naive import NaiveModel


def test_get_backbone_config_reads_backbone_layout() -> None:
    config = OmegaConf.create({"model": {"backbone": {"hidden_nodes_name": "hidden", "latent_skip": True}}})

    assert _get_backbone_config(config).hidden_nodes_name == "hidden"
    assert _get_backbone_config(config).latent_skip is True


def test_get_backbone_config_requires_backbone() -> None:
    config = OmegaConf.create({"model": {}})

    with pytest.raises(KeyError, match="model.backbone"):
        _get_backbone_config(config)


def test_naive_model_preserves_ensemble_dimension() -> None:
    model = NaiveModel(n_input=2, n_output=3, n_step_input=2, n_step_output=1)
    x = {"data": torch.randn(4, 2, 5, 7, 2)}

    y = model(x)

    assert y["data"].shape == (4, 1, 5, 7, 3)

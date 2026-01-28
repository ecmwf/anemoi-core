# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

import pytest
from omegaconf import OmegaConf

from anemoi.training.api import build_trainer
from anemoi.training.api import normalize_config
from anemoi.training.train.train import AnemoiTrainer
from anemoi.utils.testing import GetTestArchive
from anemoi.utils.testing import skip_if_offline


def _param_count(module: Any) -> int:
    return sum(param.numel() for param in module.parameters())


@skip_if_offline
@pytest.mark.slow
def test_python_api_smoke(
    architecture_config: tuple,
    get_test_archive: GetTestArchive,
) -> None:
    cfg, url, _ = architecture_config
    get_test_archive(url)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_hydra = OmegaConf.create(cfg_dict)
    cfg_python = OmegaConf.create(cfg_dict)

    hydra_trainer = AnemoiTrainer(normalize_config(cfg_hydra))
    python_trainer = build_trainer(cfg_python)

    hydra_model = hydra_trainer.model
    python_model = python_trainer.model

    assert type(hydra_model) is type(python_model)
    assert _param_count(hydra_model) == _param_count(python_model)
    assert set(hydra_model.output_mask.keys()) == set(python_model.output_mask.keys())
    assert set(hydra_trainer.data_indices.keys()) == set(python_trainer.data_indices.keys())

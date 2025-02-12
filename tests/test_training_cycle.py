# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import shutil

import pytest
import torch
from hydra import compose
from hydra import initialize
from omegaconf import OmegaConf

from anemoi.training.train.train import AnemoiTrainer

LOGGER = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "hydra_overrides", [["model=gnn"], ["model=graphtransformer"], ["model=transformer", "graph=encoder_decoder_only"]]
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
def test_training_cycle_debug_gnn_config(hydra_overrides) -> None:

    with initialize(version_base=None, config_path="", job_name="test_training"):
        cfg = compose(config_name="basic_config", overrides=hydra_overrides)
        OmegaConf.resolve(cfg)

        AnemoiTrainer(cfg).train()
        shutil.rmtree(cfg.hardware.paths.output)

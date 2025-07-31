# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback

LOGGER = logging.getLogger(__name__)


class ParentUUIDCallback(Callback):
    """A callback that retrieves the parent UUID for a model, if it is a child model."""

    def __init__(self, config: OmegaConf) -> None:
        """Initialise the ParentUUIDCallback callback.

        Parameters
        ----------
        config : OmegaConf
            Config object

        """
        super().__init__()
        self.config = config

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: torch.nn.Module,
    ) -> None:
        del trainer  # unused
        pl_module.hparams["metadata"]["parent_uuid"] = checkpoint["hyper_parameters"]["metadata"]["uuid"]

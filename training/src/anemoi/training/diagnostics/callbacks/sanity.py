# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import TYPE_CHECKING

import pytorch_lightning as pl

if TYPE_CHECKING:
    from anemoi.training.data.dataset import NativeGridDataset

LOGGER = logging.getLogger(__name__)


class CheckVariableOrder(pl.callbacks.Callback):
    """Check the order of the variables in a pre-trained / fine-tuning model."""

    def __init__(self) -> None:
        super().__init__()

    def _check_variable_order(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        dataset: "NativeGridDataset",
    ) -> None:
        """Check the order of variables between model checkpoint and dataset.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer
        pl_module : pl.LightningModule
            Lightning module (already unwrapped by PyTorch Lightning)
        dataset : NativeGridDataset
            Dataset to compare against
        """
        data_name_to_index = dataset.name_to_index

        if hasattr(pl_module, "_ckpt_model_name_to_index"):
            model_name_to_index = pl_module._ckpt_model_name_to_index
        else:
            model_name_to_index = trainer.datamodule.data_indices.name_to_index

        trainer.datamodule.data_indices.compare_variables(model_name_to_index, data_name_to_index)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check the order of the variables in the model from checkpoint and the training data.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer
        pl_module : pl.LightningModule
            Lightning module
        """
        self._check_variable_order(trainer, pl_module, trainer.datamodule.ds_train)

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check the order of the variables in the model from checkpoint and the validation data.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer
        pl_module : pl.LightningModule
            Lightning module
        """
        self._check_variable_order(trainer, pl_module, trainer.datamodule.ds_valid)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check the order of the variables in the model from checkpoint and the test data.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer
        pl_module : pl.LightningModule
            Lightning module
        """
        self._check_variable_order(trainer, pl_module, trainer.datamodule.ds_test)

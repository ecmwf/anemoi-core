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

LOGGER = logging.getLogger(__name__)


class CheckVariableOrder(pl.callbacks.Callback):
    """Check the order of the variables in a pre-trained / fine-tuning model."""

    def __init__(self) -> None:
        super().__init__()

    def on_train_start(self, trainer: pl.Trainer, _: pl.LightningModule) -> None:
        """Check the order of the variables in the model from checkpoint and the training data.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer
        _ : pl.LightningModule
            Not used
        """
        data_name_to_index = trainer.datamodule.ds_train.name_to_index

        if not hasattr(trainer.model, "module"):
            LOGGER.warning("Dataset order is not being checked")
            return

        if hasattr(trainer.model.module, "_ckpt_model_name_to_index"):
            self._model_name_to_index = trainer.model.module._ckpt_model_name_to_index
        else:
            self._model_name_to_index = trainer.datamodule.data_indices.name_to_index

        trainer.datamodule.data_indices.compare_variables(self._model_name_to_index, data_name_to_index)

    def on_validation_start(self, trainer: pl.Trainer, _: pl.LightningModule) -> None:
        """Check the order of the variables in the model from checkpoint and the validation data.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer
        _ : pl.LightningModule
            Not used
        """
        data_name_to_index = trainer.datamodule.ds_valid.name_to_index

        if not hasattr(trainer.model, "module"):
            LOGGER.warning("Dataset order is not being checked")
            return

        if hasattr(trainer.model.module, "_ckpt_model_name_to_index"):
            self._model_name_to_index = trainer.model.module._ckpt_model_name_to_index
        else:
            self._model_name_to_index = trainer.datamodule.data_indices.name_to_index

        trainer.datamodule.data_indices.compare_variables(self._model_name_to_index, data_name_to_index)

    def on_test_start(self, trainer: pl.Trainer, _: pl.LightningModule) -> None:
        """Check the order of the variables in the model from checkpoint and the test data.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer
        _ : pl.LightningModule
            Not used
        """
        data_name_to_index = trainer.datamodule.ds_test.name_to_index

        if not hasattr(trainer.model, "module"):
            LOGGER.warning("Dataset order is not being checked")
            return

        if hasattr(trainer.model.module, "_ckpt_model_name_to_index"):
            self._model_name_to_index = trainer.model.module._ckpt_model_name_to_index
        else:
            self._model_name_to_index = trainer.datamodule.data_indices.name_to_index

        trainer.datamodule.data_indices.compare_variables(self._model_name_to_index, data_name_to_index)

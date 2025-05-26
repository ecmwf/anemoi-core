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
        self._model_name_to_index = None

    def on_load_checkpoint(self, trainer: pl.Trainer, _: pl.LightningModule, checkpoint: dict) -> None:
        """Cache the model mapping from the checkpoint.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer
        _ : pl.LightningModule
            Not used
        checkpoint : dict
            Pytorch Lightning checkpoint
        """
        self._model_name_to_index = checkpoint["hyper_parameters"]["data_indices"].name_to_index
        data_name_to_index = trainer.datamodule.data_indices.name_to_index

        trainer.datamodule.data_indices._compare_variables(self._model_name_to_index, data_name_to_index)

    def on_sanity_check_start(self, trainer: pl.Trainer, _: pl.LightningModule) -> None:
        """Cache the model mapping from the datamodule if not loaded from checkpoint.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer
        _ : pl.LightningModule
            Not used
        """
        if self._model_name_to_index is None:
            self._model_name_to_index = trainer.datamodule.data_indices.name_to_index

    def on_train_epoch_start(self, trainer: pl.Trainer, _: pl.LightningModule) -> None:
        """Check the order of the variables in the model from checkpoint and the training data.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer
        _ : pl.LightningModule
            Not used
        """
        data_name_to_index = trainer.datamodule.ds_train.name_to_index

        trainer.datamodule.data_indices._compare_variables(self._model_name_to_index, data_name_to_index)

    def on_validation_epoch_start(self, trainer: pl.Trainer, _: pl.LightningModule) -> None:
        """Check the order of the variables in the model from checkpoint and the validation data.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer
        _ : pl.LightningModule
            Not used
        """
        data_name_to_index = trainer.datamodule.ds_valid.name_to_index

        trainer.datamodule.data_indices._compare_variables(self._model_name_to_index, data_name_to_index)

    def on_test_epoch_start(self, trainer: pl.Trainer, _: pl.LightningModule) -> None:
        """Check the order of the variables in the model from checkpoint and the test data.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer
        _ : pl.LightningModule
            Not used
        """
        data_name_to_index = trainer.datamodule.ds_test.name_to_index

        trainer.datamodule.data_indices._compare_variables(self._model_name_to_index, data_name_to_index)

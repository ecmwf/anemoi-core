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

    def _get_model_name_to_index(self, trainer: pl.Trainer):  # type: ignore[no-untyped-def]
        """Get the model name to index mapping, handling both checkpoint and data indices."""
        if hasattr(trainer.model.module, "_ckpt_model_name_to_index"):
            return trainer.model.module._ckpt_model_name_to_index
        if isinstance(trainer.datamodule.data_indices, dict):
            model_name_to_index = {}
            for dataset_name, data_indices in trainer.datamodule.data_indices.items():
                model_name_to_index[dataset_name] = data_indices.name_to_index
            return model_name_to_index
        return trainer.datamodule.data_indices.name_to_index

    def _compare_variables(self, trainer: pl.Trainer, model_name_to_index, data_name_to_index) -> None:  # type: ignore[misc]
        """Compare variables between model and data indices."""
        if isinstance(trainer.datamodule.data_indices, dict):
            for dataset_name, data_indices in trainer.datamodule.data_indices.items():
                data_indices.compare_variables(model_name_to_index[dataset_name], data_name_to_index[dataset_name])
        else:
            trainer.datamodule.data_indices.compare_variables(model_name_to_index, data_name_to_index)

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
        self._model_name_to_index = self._get_model_name_to_index(trainer)
        self._compare_variables(trainer, self._model_name_to_index, data_name_to_index)

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
        self._model_name_to_index = self._get_model_name_to_index(trainer)
        self._compare_variables(trainer, self._model_name_to_index, data_name_to_index)

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
        self._model_name_to_index = self._get_model_name_to_index(trainer)
        self._compare_variables(trainer, self._model_name_to_index, data_name_to_index)

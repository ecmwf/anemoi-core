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

        self._compare_variables(data_name_to_index)

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

        self._compare_variables(data_name_to_index)

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

        self._compare_variables(data_name_to_index)

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

        self._compare_variables(data_name_to_index)

    def _compare_variables(self, data_name_to_index: dict[str, int]) -> None:
        """Compare the order of the variables in the model from checkpoint and the data.

        Parameters
        ----------
        data_name_to_index : dict[str, int]
            The dictionary mapping variable names to their indices in the data.

        Raises
        ------
        ValueError
            If the variable order in the model and data is verifiably different.
        """
        if self._model_name_to_index is None:
            LOGGER.info("No variable order to compare. Skipping variable order check.")
            return

        if self._model_name_to_index == data_name_to_index:
            LOGGER.info("The order of the variables in the model matches the order in the data.")
            LOGGER.debug("%s, %s", self._model_name_to_index, data_name_to_index)
            return

        keys1 = set(self._model_name_to_index.keys())
        keys2 = set(data_name_to_index.keys())

        error_msg = ""

        # Find keys unique to each dictionary
        only_in_model = {key: self._model_name_to_index[key] for key in (keys1 - keys2)}
        only_in_data = {key: data_name_to_index[key] for key in (keys2 - keys1)}

        # Find common keys
        common_keys = keys1 & keys2

        # Compare values for common keys
        different_values = {
            k: (self._model_name_to_index[k], data_name_to_index[k])
            for k in common_keys
            if self._model_name_to_index[k] != data_name_to_index[k]
        }

        LOGGER.warning(
            "The variables in the model do not match the variables in the data. "
            "If you're fine-tuning or pre-training, you may have to adjust the "
            "variable order and naming in your config.",
        )
        if only_in_model:
            LOGGER.warning("Variables only in model: %s", only_in_model)
        if only_in_data:
            LOGGER.warning("Variables only in data: %s", only_in_data)
        if set(only_in_model.values()) == set(only_in_data.values()):
            # This checks if the order is the same, but the naming is different. This is not be treated as an error.
            LOGGER.warning(
                "The variable naming is different, but the order appears to be the same. Continuing with training.",
            )
        else:
            # If the renamed variables are not in the same index locations, raise an error.
            error_msg += (
                "The variable order in the model and data is different.\n"
                "Please adjust the variable order in your config, you may need to "
                "use the 'reorder' and 'rename' key in the dataloader config.\n"
                "Refer to the Anemoi Datasets documentation for more information.\n"
            )
        if different_values:
            # If the variables are named the same but in different order, raise an error.
            error_msg += (
                f"Detected a different sort order of the same variables: {different_values}.\n"
                "Please adjust the variable order in your config, you may need to use the "
                f"'reorder' key in the dataloader config. With:\n `reorder: {self._model_name_to_index}`\n"
            )

        if error_msg:
            LOGGER.error(error_msg)
            raise ValueError(error_msg)

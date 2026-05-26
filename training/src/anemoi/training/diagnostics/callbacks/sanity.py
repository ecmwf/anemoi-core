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

from anemoi.transform.variables import Variable

LOGGER = logging.getLogger(__name__)


class CheckVariableOrder(pl.callbacks.Callback):
    """Check the order of the variables in a pre-trained / fine-tuning model."""

    def _get_model_name_to_index(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Get the model name to index mapping, handling both checkpoint and data indices."""
        if hasattr(pl_module, "_ckpt_model_name_to_index"):
            return pl_module._ckpt_model_name_to_index
        if isinstance(trainer.datamodule.data_indices, dict):
            model_name_to_index = {}
            for dataset_name, data_indices in trainer.datamodule.data_indices.items():
                model_name_to_index[dataset_name] = data_indices.name_to_index
            return model_name_to_index
        return trainer.datamodule.data_indices.name_to_index

    def _compare_variables(self, trainer: pl.Trainer, model_name_to_index: dict, data_name_to_index: dict) -> None:  # type: ignore[misc]
        """Compare variables between model and data indices."""
        for dataset_name, data_indices in trainer.datamodule.data_indices.items():
            # Only compare if dataset exists in model (handles transfer learning scenarios)
            if dataset_name in model_name_to_index and dataset_name in data_name_to_index:
                data_indices.compare_variables(model_name_to_index[dataset_name], data_name_to_index[dataset_name])
            else:
                LOGGER.debug(
                    "Skipping variable comparison for dataset '%s' (not found in checkpoint)",
                    dataset_name,
                )

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check the order of the variables in the model from checkpoint and the training data.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer
        _ : pl.LightningModule
            Not used
        """
        data_name_to_index = trainer.datamodule.ds_train.name_to_index
        self._model_name_to_index = self._get_model_name_to_index(trainer, pl_module)
        self._compare_variables(trainer, self._model_name_to_index, data_name_to_index)
        self._check_variable_units(trainer, pl_module)

    @staticmethod
    def _check_variable_units(trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check unit compatibility between checkpoint and current dataset.

        Raises
        ------
        ValueError
            If variables have incompatible units between checkpoint and dataset.
        """
        ckpt_variables_metadata = getattr(pl_module, "_ckpt_variables_metadata", None)

        if ckpt_variables_metadata is None:
            LOGGER.warning(
                "Checkpoint does not contain variables_metadata. Skipping unit compatibility check.",
            )
            return

        for dataset_name, ckpt_var_meta in ckpt_variables_metadata.items():
            dataset_metadata = trainer.datamodule.metadata.get(dataset_name, {})
            ds_var_meta = dataset_metadata.get("variables_metadata")

            if ds_var_meta is None:
                LOGGER.warning(
                    "Dataset '%s' does not contain variables_metadata. "
                    "Skipping unit compatibility check for this dataset.",
                    dataset_name,
                )
                continue

            ckpt_vars = {name: Variable.from_dict(name, data) for name, data in ckpt_var_meta.items()}
            ds_vars = {name: Variable.from_dict(name, data) for name, data in ds_var_meta.items()}

            Variable.check_compatibility(ckpt_vars, ds_vars)

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check the order of the variables in the model from checkpoint and the validation data.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer
        _ : pl.LightningModule
            Not used
        """
        data_name_to_index = trainer.datamodule.ds_valid.name_to_index
        self._model_name_to_index = self._get_model_name_to_index(trainer, pl_module)
        self._compare_variables(trainer, self._model_name_to_index, data_name_to_index)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check the order of the variables in the model from checkpoint and the test data.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer
        _ : pl.LightningModule
            Not used
        """
        data_name_to_index = trainer.datamodule.ds_test.name_to_index
        self._model_name_to_index = self._get_model_name_to_index(trainer, pl_module)
        self._compare_variables(trainer, self._model_name_to_index, data_name_to_index)

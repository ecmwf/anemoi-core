# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import pytorch_lightning as pl
from omegaconf import OmegaConf

from anemoi.training.losses.utils import check_loss_tree_variable_units
from anemoi.training.utils.variables_metadata import check_variables_metadata_compatibility

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
        # Opt-in fine-tuning into FEWER variables (issue #838): tolerate the current data
        # being a strict subset of the checkpoint's variables, matching the trainer-side check.
        allow_subset = bool(trainer.datamodule.config.training.get("allow_variable_subset", False))
        for dataset_name, data_indices in trainer.datamodule.data_indices.items():
            # Only compare if dataset exists in model (handles transfer learning scenarios)
            if dataset_name in model_name_to_index and dataset_name in data_name_to_index:
                data_indices.compare_variables(
                    model_name_to_index[dataset_name],
                    data_name_to_index[dataset_name],
                    allow_subset=allow_subset,
                )
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
            Pytorch Lightning trainer.
        pl_module : pl.LightningModule
            Training module.
        """
        data_name_to_index = trainer.datamodule.ds_train.name_to_index
        self._model_name_to_index = self._get_model_name_to_index(trainer, pl_module)
        self._compare_variables(trainer, self._model_name_to_index, data_name_to_index)
        self._check_variable_units(trainer, pl_module)

    @staticmethod
    def _check_variable_units(trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check variable compatibility required by the configured training run.

        Raises
        ------
        ValueError
            If variables have incompatible metadata.
        """
        ckpt_variables_metadata = getattr(pl_module, "_ckpt_variables_metadata", None)
        compat_cfg = trainer.datamodule.config.training.get("check_variables_compatibility", {})
        compat_options = (
            OmegaConf.to_container(compat_cfg, resolve=True) if OmegaConf.is_config(compat_cfg) else (compat_cfg or {})
        )

        trainer.datamodule.loading_policy.check_dataset_compatibility(
            trainer.datamodule.ds_train,
            **compat_options,
        )

        for dataset_name in pl_module.target_dataset_names:
            check_loss_tree_variable_units(
                pl_module.loss[dataset_name],
                trainer.datamodule.metadata[dataset_name].get("variables_metadata"),
            )

        allow_subset = bool(trainer.datamodule.config.training.get("allow_variable_subset", False))
        check_variables_metadata_compatibility(
            ckpt_variables_metadata,
            trainer.datamodule.metadata,
            allow_subset=allow_subset,
            **compat_options,
        )

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check the order of the variables in the model from checkpoint and the validation data.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer.
        pl_module : pl.LightningModule
            Training module.
        """
        data_name_to_index = trainer.datamodule.ds_valid.name_to_index
        self._model_name_to_index = self._get_model_name_to_index(trainer, pl_module)
        self._compare_variables(trainer, self._model_name_to_index, data_name_to_index)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check the order of the variables in the model from checkpoint and the test data.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer.
        pl_module : pl.LightningModule
            Training module.
        """
        data_name_to_index = trainer.datamodule.ds_test.name_to_index
        self._model_name_to_index = self._get_model_name_to_index(trainer, pl_module)
        self._compare_variables(trainer, self._model_name_to_index, data_name_to_index)

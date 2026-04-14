# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from torch.utils.checkpoint import checkpoint

from anemoi.training.diagnostics.callbacks.plot_adapter import AutoencoderPlotAdapter
from anemoi.training.train.tasks.base import BaseGraphModule
from anemoi.training.utils.index_space import IndexSpace

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch
    from omegaconf import DictConfig
    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.models.interface import ModelInterface


LOGGER = logging.getLogger(__name__)


class GraphAutoEncoder(BaseGraphModule):
    """Graph neural network autoencoder for PyTorch Lightning."""

    task_type = "autoencoder"

    def __init__(
        self,
        *,
        model: ModelInterface,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        **kwargs,
    ) -> None:
        """Initialize graph neural network interpolator.

        Parameters
        ----------
        model : ModelInterface
        config : DictConfig
            Job configuration
        graph_data : HeteroData
            Graph object
        statistics : dict
            Statistics of the training data
        data_indices : IndexCollection
            Indices of the training data,

        """
        super().__init__(
            model=model,
            config=config,
            graph_data=graph_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            **kwargs,
        )

        assert (
            self.n_step_input == self.n_step_output
        ), "Autoencoders must have the same number of input and output steps."

        self._plot_adapter = AutoencoderPlotAdapter(self)

        self.fill_metadata(self.metadata)

    def fill_metadata(self, metadata: dict) -> None:
        for dataset_name in self.dataset_names:
            ts = metadata["metadata_inference"][dataset_name]["timesteps"]
            rel = ts["relative_date_indices_training"]
            ts["input_relative_date_indices"] = rel[: self.n_step_input]
            ts["output_relative_date_indices"] = rel[-self.n_step_output :]

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:

        required_time_steps = max(self.n_step_input, self.n_step_output)
        x = {}

        for dataset_name, dataset_batch in batch.items():
            msg = (
                f"Batch length not sufficient for requested n_step_input/n_step_output for {dataset_name}!"
                f" {dataset_batch.shape[1]} !>= {required_time_steps}"
            )
            assert dataset_batch.shape[1] >= required_time_steps, msg
            x[dataset_name] = dataset_batch[
                :,
                0:required_time_steps,
                ...,
                self.data_indices[dataset_name].data.input.full,
            ]

        y_pred = self(x)

        y = self.get_target(
            batch,
            start=0,
        )

        # y includes the auxiliary variables, so we must leave those out when computing the loss
        loss, metrics, y_pred = checkpoint(
            self.compute_loss_metrics,
            y_pred,
            y,
            rollout_step=0,
            training_mode=True,
            validation_mode=validation_mode,
            pred_layout=IndexSpace.MODEL_OUTPUT,
            target_layout=IndexSpace.DATA_FULL,
            use_reentrant=False,
        )

        # All tasks return (loss, metrics, list of per-step dicts) for consistent plot callback contract.
        return loss, metrics, [y_pred]

    def on_train_epoch_end(self) -> None:
        pass

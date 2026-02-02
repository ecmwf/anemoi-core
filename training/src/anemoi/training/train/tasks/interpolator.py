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
from operator import itemgetter
from typing import TYPE_CHECKING

import torch
from omegaconf import DictConfig
from omegaconf import open_dict
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.training.train.objectives import DirectPredictionObjective
from anemoi.training.train.tasks.base import BaseGraphModule

if TYPE_CHECKING:
    from collections.abc import Mapping

    from omegaconf import DictConfig
    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection


LOGGER = logging.getLogger(__name__)


class GraphInterpolator(BaseGraphModule):
    """Graph neural network interpolator for PyTorch Lightning."""

    task_type = "time-interpolator"
    supported_objectives = (DirectPredictionObjective,)

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network interpolator.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        graph_data : HeteroData
            Graph object
        statistics : dict
            Statistics of the training data
        data_indices : IndexCollection
            Indices of the training data,
        metadata : dict
            Provenance information
        supporting_arrays : dict
            Supporting NumPy arrays to store in the checkpoint

        """
        super().__init__(
            config=config,
            graph_data=graph_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        assert self.multi_out == 1, "For multiple outputs, use GraphMultiOutInterpolator"

        target_forcing_config = get_multiple_datasets_config(config.training.target_forcing)
        self.target_forcing_indices, self.use_time_fraction = {}, {}
        for dataset_name in self.dataset_names:
            if len(target_forcing_config[dataset_name].data) >= 1:
                self.target_forcing_indices[dataset_name] = itemgetter(*target_forcing_config[dataset_name].data)(
                    data_indices[dataset_name].data.input.name_to_index,
                )
                if isinstance(self.target_forcing_indices[dataset_name], int):
                    self.target_forcing_indices[dataset_name] = [self.target_forcing_indices[dataset_name]]
            else:
                self.target_forcing_indices[dataset_name] = []

            self.use_time_fraction[dataset_name] = target_forcing_config[dataset_name].time_fraction

        self.num_tfi = {name: len(idxs) for name, idxs in self.target_forcing_indices.items()}

        self.boundary_times = config.training.explicit_times.input
        self.interp_times = config.training.explicit_times.target
        sorted_indices = sorted(set(self.boundary_times + self.interp_times))
        self.imap = {data_index: batch_index for batch_index, data_index in enumerate(sorted_indices)}
        self.multi_step = 1
        self.rollout = 1

    def get_target_forcing(self, batch: dict[str, torch.Tensor], interp_step: int) -> dict[str, torch.Tensor]:
        batch_size = next(iter(batch.values())).shape[0]
        ens_size = next(iter(batch.values())).shape[2]
        grid_size = next(iter(batch.values())).shape[3]
        batch_type = next(iter(batch.values())).dtype

        target_forcing = {}
        for dataset_name, num_tfi in self.num_tfi.items():
            target_forcing[dataset_name] = torch.empty(
                batch_size,
                ens_size,
                grid_size,
                num_tfi + self.use_time_fraction[dataset_name],
                device=self.device,
                dtype=batch_type,
            )

            # get the forcing information for the target interpolation time:
            if num_tfi >= 1:
                target_forcing[dataset_name][..., :num_tfi] = batch[dataset_name][
                    :,
                    self.imap[interp_step],
                    :,
                    :,
                    self.target_forcing_indices[dataset_name],
                ]

            if self.use_time_fraction[dataset_name]:
                target_forcing[dataset_name][..., -1] = (interp_step - self.boundary_times[-2]) / (
                    self.boundary_times[-1] - self.boundary_times[-2]
                )

        return target_forcing

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        loss = torch.zeros(1, dtype=next(iter(batch.values())).dtype, device=self.device, requires_grad=False)
        metrics = {}
        y_preds = []

        x_bound = {}
        for dataset_name in self.dataset_names:
            # Boundary inputs: conditioning states at explicit boundary times.
            x_bound[dataset_name] = batch[dataset_name][:, itemgetter(*self.boundary_times)(self.imap)][
                ...,
                self.data_indices[dataset_name].data.input.full,
            ]  # (bs, time, ens, latlon, nvar)

        for interp_step in self.interp_times:
            target_forcing = self.get_target_forcing(batch, interp_step)

            y = {}
            for dataset_name, dataset_batch in batch.items():
                # Target output state at the interpolation time.
                y[dataset_name] = dataset_batch[
                    :,
                    self.imap[interp_step],
                    :,
                    :,
                    self.data_indices[dataset_name].data.output.full,
                ]

            shapes = {k: y_.shape for k, y_ in y.items()}
            model_impl = self.model.model
            # Objective provides schedule (e.g., sigma or time) per batch.
            # Unused in direct prediction objective.
            schedule = self.objective.sample_schedule(
                shape=shapes,
                device=next(iter(batch.values())).device,
                model=model_impl,
            )
            # Objective builds (conditioning, target) pair for the training loss.
            # y_cond is None for direct prediction objective.
            y_cond, target = self.objective.build_training_pair(y, schedule)
            # Forward pass in objective space (e.g., denoising or velocity).
            y_pred = self.objective.forward(
                model_impl,
                x_bound,
                y_cond,
                schedule,
                model_comm_group=self.model_comm_group,
                grid_shard_shapes=self.grid_shard_shapes,
                target_forcing=target_forcing,
            )
            metrics_y_pred = None
            metrics_y = None
            if validation_mode:
                # Clean prediction/target in normalized output state space for metrics.
                y_pred_clean, y_clean = self.objective.clean_pred_target_pair(y_pred, y, y_cond, schedule)
                metrics_y_pred = y_pred_clean
                metrics_y = y_clean
            # Optional pre-loss weights (e.g., diffusion noise weighting).
            # None for direct prediction objective.
            pre_loss_weights = self.objective.pre_loss_weights(schedule, model=model_impl)

            loss_step, metrics_next, y_pred = checkpoint(
                self.compute_loss_metrics,
                y_pred,
                target,
                step=interp_step - 1,
                validation_mode=validation_mode,
                metrics_y_pred=metrics_y_pred,
                metrics_y=metrics_y,
                pre_loss_weights=pre_loss_weights,
                use_reentrant=False,
            )

            loss += loss_step
            metrics.update(metrics_next)
            y_preds.append(y_pred)

        # Aggregate loss across interpolation times; predictions used for diagnostics.
        loss *= 1.0 / len(self.interp_times)
        return loss, metrics, y_preds

    def forward(self, x: torch.Tensor, target_forcing: torch.Tensor) -> torch.Tensor:
        return super().forward(x, target_forcing=target_forcing)


class GraphMultiOutInterpolator(BaseGraphModule):
    """Graph neural network interpolator with multiple output steps for PyTorch Lightning."""

    task_type = "time-interpolator"
    supported_objectives = (DirectPredictionObjective,)

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: dict[str, HeteroData],
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: dict[str, IndexCollection],
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network interpolator.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        graph_data : dict[str, HeteroData]
            Graph objects keyed by dataset name
        statistics : dict
            Statistics of the training data
        data_indices : dict[str, IndexCollection]
            Indices of the training data
        metadata : dict
            Provenance information
        supporting_arrays : dict
            Supporting NumPy arrays to store in the checkpoint

        """
        with open_dict(config.training):
            config.training.multistep_output = len(config.training.explicit_times.target)
        super().__init__(
            config=config,
            graph_data=graph_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        self.boundary_times = config.training.explicit_times.input
        self.interp_times = config.training.explicit_times.target
        self.multi_out = len(self.interp_times)
        sorted_indices = sorted(set(self.boundary_times + self.interp_times))
        self.imap = {data_index: batch_index for batch_index, data_index in enumerate(sorted_indices)}

        self.multi_step = 1
        self.rollout = 1

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        x_bound = {}
        y = {}
        for dataset_name, dataset_batch in batch.items():
            # Boundary inputs: conditioning states at explicit boundary times.
            x_bound[dataset_name] = dataset_batch[:, itemgetter(*self.boundary_times)(self.imap)][
                ...,
                self.data_indices[dataset_name].data.input.full,
            ]  # (bs, time, ens, latlon, nvar)

            # Target output window: full output variables for all interpolation times.
            y[dataset_name] = dataset_batch[:, itemgetter(*self.interp_times)(self.imap)][
                ...,
                self.data_indices[dataset_name].data.output.full,
            ]

        loss = torch.zeros(1, dtype=next(iter(batch.values())).dtype, device=self.device, requires_grad=False)

        shapes = {k: y_.shape for k, y_ in y.items()}
        model_impl = self.model.model
        # Objective provides schedule (e.g., sigma or time) per batch.
        # Unused in direct prediction objective.
        schedule = self.objective.sample_schedule(
            shape=shapes,
            device=next(iter(batch.values())).device,
            model=model_impl,
        )
        # Objective builds (conditioning, target) pair for the training loss.
        # y_cond is None for direct prediction objective.
        y_cond, target = self.objective.build_training_pair(y, schedule)
        # Forward pass in objective space (e.g., denoising or velocity).
        y_pred = self.objective.forward(
            model_impl,
            x_bound,
            y_cond,
            schedule,
            model_comm_group=self.model_comm_group,
            grid_shard_shapes=self.grid_shard_shapes,
        )
        metrics_y_pred = None
        metrics_y = None
        if validation_mode:
            # Clean prediction/target in normalized output state space for metrics.
            y_pred_clean, y_clean = self.objective.clean_pred_target_pair(y_pred, y, y_cond, schedule)
            metrics_y_pred = y_pred_clean
            metrics_y = y_clean
        # Optional pre-loss weights (e.g., diffusion noise weighting).
        # None for direct prediction objective.
        pre_loss_weights = self.objective.pre_loss_weights(schedule, model=model_impl)

        loss, metrics, _ = checkpoint(
            self.compute_loss_metrics,
            y_pred,
            target,
            validation_mode=validation_mode,
            metrics_y_pred=metrics_y_pred,
            metrics_y=metrics_y,
            pre_loss_weights=pre_loss_weights,
            use_reentrant=False,
        )

        return loss, metrics, y_pred

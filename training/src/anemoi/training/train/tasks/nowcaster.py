# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Mapping
from operator import itemgetter

import torch
from omegaconf import DictConfig
from omegaconf import open_dict
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.train.tasks.base import BaseGraphModule

LOGGER = logging.getLogger(__name__)


class GraphNowcaster(BaseGraphModule):
    """Interpolates between NWP states using surface observations.

    A graph neural network that leverages surface observations to inform interpolation between NWP states
    for fine-scale, high-frequency nowcasts of atmospheric variables
    (see https://arxiv.org/abs/2509.00017).
    """

    task_type = "time-interpolator"

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
        self.n_step_input = self.model.n_step_input
        self.known_future_variables = self.model.model.known_future_variables
        self.known_future_indices = {
            dataset_name: (
                itemgetter(*kfv)(
                    data_indices[dataset_name].data.input.name_to_index,
                )
                if len(kfv) > 0
                else []
            )
            for dataset_name, kfv in self.known_future_variables.items()
        }
        self.known_future_indices = {
            dataset_name: ([idx] if isinstance(idx, int) else idx)
            for dataset_name, idx in self.known_future_indices.items()
        }
        boundary_times = config.training.explicit_times.input
        self.boundary_times = [t + self.n_step_input - 1 for t in boundary_times]
        interp_times = config.training.explicit_times.target
        self.interp_times = [t + self.n_step_input - 1 for t in interp_times]
        sorted_indices = sorted(
            set(range(self.n_step_input)).union(
                self.boundary_times,
                self.interp_times,
            ),
        )
        self.imap = {data_index: batch_index for batch_index, data_index in enumerate(sorted_indices)}
    
    @property
    def output_times(self) -> int:
        """Number of interpolation times (outer loop in plot callbacks; one forward, n_step_output steps)."""
        return len(self.interp_times)

    def get_init_step(self, rollout_step: int) -> int:
        return rollout_step
    
    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> tuple[Tensor, Mapping[str, Tensor], Tensor]:
        x, y = {}, {}
        for dataset_name, data_batch in batch.items():
            data_batch = self.model.pre_processors[dataset_name](data_batch)
            obs = {var.item() for var in self.data_indices[dataset_name].data.input.full}.difference(
                set(self.known_future_indices[dataset_name]),
            )
            if len(obs) == 0:
                assert (
                    len(self.known_future_indices[dataset_name]) > 0
                ), "If no observed variables, need known future variables to derive bounds."
                LOGGER.warning("Adding dataset %s to inputs, with no observed variables.", dataset_name)
                x_init = data_batch[:, itemgetter(*self.boundary_times)(self.imap)][
                    ...,
                    self.known_future_indices[dataset_name],
                ]  # bounds are derived from variables we know in the future
            else:
                LOGGER.warning("Adding observation dataset %s to inputs.", dataset_name)
                assert (
                    len(self.known_future_indices[dataset_name]) == 0
                ), "Known future variables not supported for datasets with observed variables."
                x_init = data_batch[
                    :,
                    : self.n_step_input,
                    ...,
                    list(obs),
                ]  # here only past steps are used for observed vars
            y[dataset_name] = data_batch[:, itemgetter(*self.interp_times)(self.imap)]
            x[dataset_name] = x_init
        decoder_ctx = self._build_decoder_context(batch)
        y_pred = self(x, decoder_context=decoder_ctx)
        loss, metrics, y_pred = checkpoint(
            self.compute_loss_metrics,
            y_pred,
            y,
            validation_mode=validation_mode,
            use_reentrant=False,
        )

        return loss, metrics, y_pred

    def _build_decoder_context(self, batch: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
        """Build decoder_context with key 'cond'.

        cond[t] = concat(target_forcings_at_t, time_fraction)
        """
        batch_size = next(iter(batch.values())).shape[0]
        ens_size = next(iter(batch.values())).shape[2]
        dtype = next(iter(batch.values())).dtype

        ctx = {}
        for ds in self.dataset_names:
            idxs = self.known_future_indices[ds]
            grid_size = batch[ds].shape[3]
            n_forc = len(idxs)
            if n_forc == 0:
                continue
            cond = torch.empty(
                batch_size,
                self.n_step_output,
                ens_size,
                grid_size,
                n_forc + 1,
                device=self.device,
                dtype=dtype,
            )

            for t_i, interp_step in enumerate(self.interp_times):
                if n_forc:
                    cond[:, t_i, ..., :n_forc] = batch[ds][:, self.imap[interp_step], :, :, idxs]
                    cond[:, t_i, ..., -1] = (interp_step - self.boundary_times[-2]) / (
                        self.boundary_times[-1] - self.boundary_times[-2]
                    )  # time fraction between the last two boundary times
            ctx[ds] = {"cond": cond}
        return ctx

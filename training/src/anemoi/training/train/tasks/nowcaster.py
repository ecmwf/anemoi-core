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

import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from omegaconf import open_dict
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.train.tasks.base import BaseGraphModule

LOGGER = logging.getLogger(__name__)


class GraphNowcaster(BaseGraphModule):
    """Nowcaster using shifted interpolation indices with decoder-side step looping."""

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
        with open_dict(config.training):
            config.training.multistep_output = 1
        super().__init__(
            config=config,
            graph_data=graph_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        self.dataset_name = self.dataset_names[0]
        self.n_step_input = int(config.training.get("multistep_input", self.model.n_step_input))

        raw_known_future = config.training.get("known_future_variables", None)
        known_future_names = [] if raw_known_future is None else list(OmegaConf.to_container(raw_known_future, resolve=True))
        self.known_future_variables = {dataset_name: [] for dataset_name in self.dataset_names}
        self.known_future_indices = {}
        self.input_indices = {}
        for dataset_name in self.dataset_names:
            name_to_index = data_indices[dataset_name].data.input.name_to_index
            missing = [name for name in known_future_names if name not in name_to_index]
            if missing:
                raise ValueError(
                    f"Unknown known_future_variables for dataset '{dataset_name}': {missing}. "
                    f"Available inputs: {sorted(name_to_index)}"
                )
            self.known_future_variables[dataset_name] = list(known_future_names)
            self.known_future_indices[dataset_name] = [int(name_to_index[name]) for name in known_future_names]
            self.input_indices[dataset_name] = [int(idx) for idx in data_indices[dataset_name].data.input.full.tolist()]

        boundary_times = [int(t) for t in config.training.explicit_times.input]
        if len(boundary_times) == 0:
            raise ValueError("`training.explicit_times.input` cannot be empty for nowcaster.")
        self.boundary_times = [t + self.n_step_input - 1 for t in boundary_times]

        interp_times = [int(t) for t in config.training.explicit_times.target]
        if len(interp_times) == 0:
            raise ValueError("`training.explicit_times.target` cannot be empty for nowcaster.")
        self.interp_times = [t + self.n_step_input - 1 for t in interp_times]

        sorted_indices = sorted(set(range(self.n_step_input)).union(self.boundary_times, self.interp_times))
        self.imap = {data_index: batch_index for batch_index, data_index in enumerate(sorted_indices)}

    @property
    def output_times(self) -> int:
        return len(self.interp_times)

    def get_init_step(self, rollout_step: int) -> int:
        return rollout_step

    def _build_decoder_context(
        self,
        batch: dict[str, torch.Tensor],
        decode_dataset_names: tuple[str, ...] | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        active_datasets = decode_dataset_names or tuple(batch.keys())

        present_rel = int(self.boundary_times[0])
        future_rel = int(self.boundary_times[-1]) if len(self.boundary_times) > 1 else int(self.interp_times[-1])
        span = future_rel - present_rel
        ratio_values = []
        for interp_rel in self.interp_times:
            if span == 0:
                ratio_values.append(0.0)
            else:
                ratio_values.append((int(interp_rel) - present_rel) / span)

        ctx = {}
        for dataset_name in active_datasets:
            data_batch = batch[dataset_name]
            batch_size, _, ens_size, grid_size, _ = data_batch.shape
            ratio_tensor = torch.tensor(ratio_values, device=data_batch.device, dtype=data_batch.dtype)
            cond_fraction = ratio_tensor.view(1, len(self.interp_times), 1, 1, 1).expand(
                batch_size,
                len(self.interp_times),
                ens_size,
                grid_size,
                1,
            )

            cond_parts = [cond_fraction]
            future_indices = self.known_future_indices[dataset_name]
            if future_indices:
                target_batch_indices = [self.imap[int(t)] for t in self.interp_times]
                cond_parts.append(data_batch[:, target_batch_indices, ..., future_indices])

            ctx[dataset_name] = {"cond": torch.cat(cond_parts, dim=-1)}
        return ctx

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> tuple[Tensor, Mapping[str, Tensor], Tensor]:
        x, y, processed_batch = {}, {}, {}
        for dataset_name, data_batch in batch.items():
            if data_batch.is_floating_point() and data_batch.dtype == torch.float64:
                data_batch = data_batch.float()
            processed_batch[dataset_name] = data_batch
            x[dataset_name] = data_batch[:, : self.n_step_input, ..., self.input_indices[dataset_name]]
            target_batch_indices = [self.imap[int(t)] for t in self.interp_times]
            y[dataset_name] = data_batch[:, target_batch_indices]

        decoder_ctx = self._build_decoder_context(processed_batch, decode_dataset_names=tuple(x.keys()))
        y_pred = self(
            x,
            decoder_context=decoder_ctx,
            decode_dataset_names=tuple(x.keys()),
        )
        loss, metrics, y_pred = checkpoint(
            self.compute_loss_metrics,
            y_pred,
            y,
            validation_mode=validation_mode,
            use_reentrant=False,
        )

        return loss, metrics, y_pred

# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.training.train.training_task.base import BaseTask
import torch
import logging
from anemoi.models.data_indices.collection import IndexCollection
from operator import itemgetter

LOGGER = logging.getLogger(__name__)

class TimeInterpolationTask(BaseTask):
    """Time interpolation task implementation."""

    name: str = "timeinterpolation"

    def __init__(
        self,
        explicit_input_times: list[int],
        explicit_output_times: list[int],
        target_forcings: dict[str, list[int]],
        **_kwargs,
    ) -> None:
        self.boundary_times = explicit_input_times  # [0, 6]
        self.interp_times = explicit_output_times  # [1, 2, 3, 4, 5]
        self.target_forcings = target_forcings # {"data": []}
        self.use_time_fraction = True

        self.num_tfi = {name: len(elements) for name, elements in self.target_forcings.items()}
        sorted_indices = sorted(set(self.boundary_times + self.interp_times))
        self.imap = {data_index: batch_index for batch_index, data_index in enumerate(sorted_indices)}

    def get_inputs(
        self,
        batch: dict[str, torch.Tensor],
        data_indices: dict[str, IndexCollection],
    ) -> dict[str, torch.Tensor]:

        x_bound = {}
        for dataset_name, dataset_batch in batch.items():
            time_indices = itemgetter(*self.boundary_times)(self.imap)
            x_bound[dataset_name] = dataset_batch[:, time_indices][...,data_indices[dataset_name].data.input.full]
            # shape: (bs, time, ens, latlon, nvar)
        return x_bound

    def get_target_forcing(self, batch: dict[str, torch.Tensor], interp_step: int, data_indices: dict[str, IndexCollection]) -> dict[str, torch.Tensor]:
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
                num_tfi + self.use_time_fraction,
                dtype=batch_type,
            )

            # get the forcing information for the target interpolation time:
            if num_tfi >= 1:
                target_forcing_indices = itemgetter(*self.target_forcings[dataset_name].data)(data_indices[dataset_name].data.input.name_to_index)
                target_forcing[dataset_name][..., :num_tfi] = batch[dataset_name][
                    :,
                    self.imap[interp_step],
                    :,
                    :,
                    target_forcing_indices,
                ]

            if self.use_time_fraction:
                time_range = self.boundary_times[-1] - self.boundary_times[-2]
                target_forcing[dataset_name][..., -1] = (interp_step - self.boundary_times[-2]) / time_range

        return target_forcing

    def get_targets(
        self,
        batch: dict[str, torch.Tensor],
        data_indices: dict[str, IndexCollection],
        step: int,
    ) -> dict[str, torch.Tensor]:
        y = {}
        for dataset_name, dataset_batch in batch.items():
            var_indices = data_indices[dataset_name].data.output.full.to(device=dataset_batch.device)
            y[dataset_name] = dataset_batch[:, self.imap[step], None, :, :, var_indices,]
            LOGGER.debug("SHAPE: y[%s].shape = %s", dataset_name, list(y[dataset_name].shape))
        return y

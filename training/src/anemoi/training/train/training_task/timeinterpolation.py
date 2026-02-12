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

    def __init__(self, explicit_input_times: list[int], explicit_output_times: list[int], **_kwargs):
        self.boundary_times = explicit_input_times  # [0, 6]
        self.interp_times = explicit_output_times  # [1, 2, 3, 4, 5]
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

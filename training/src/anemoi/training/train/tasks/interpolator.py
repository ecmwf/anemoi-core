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
from dataclasses import replace
from operator import itemgetter
from typing import TYPE_CHECKING

import torch
from omegaconf import open_dict
from torch.utils.checkpoint import checkpoint

from anemoi.training.diagnostics.callbacks.plot_adapter import InterpolatorMultiOutPlotAdapter
from anemoi.training.train.tasks.base import BaseGraphModule

if TYPE_CHECKING:
    from collections.abc import Mapping

    from anemoi.models.interface import ModelInterface
    from anemoi.training.config_bundle import TaskConfigBundle
    from anemoi.training.runtime import TaskRuntimeArtifacts


LOGGER = logging.getLogger(__name__)


class GraphMultiOutInterpolator(BaseGraphModule):
    """Graph neural network interpolator with multiple output steps for PyTorch Lightning."""

    task_type = "time-interpolator"

    def __init__(
        self,
        *,
        model: ModelInterface,
        config_bundle: TaskConfigBundle,
        runtime_artifacts: TaskRuntimeArtifacts,
        **kwargs,
    ) -> None:
        """Initialize graph neural network interpolator.

        Parameters
        ----------
        model : ModelInterface
        config_bundle : TaskConfigBundle
            Parts of the config used by this task.
        runtime_artifacts : TaskRuntimeArtifacts
            Data prepared by the trainer for this task.

        """
        config = config_bundle.to_dictconfig()
        with open_dict(config.training):
            config.training.multistep_output = len(config.training.explicit_times.target)
        config_bundle = replace(config_bundle, training=config.training)
        super().__init__(
            model=model,
            config_bundle=config_bundle,
            runtime_artifacts=runtime_artifacts,
            **kwargs,
        )

        self.boundary_times = self.config.training.explicit_times.input
        self.interp_times = self.config.training.explicit_times.target
        self.n_step_output = len(self.interp_times)
        sorted_indices = sorted(set(self.boundary_times + self.interp_times))
        self.imap = {data_index: batch_index for batch_index, data_index in enumerate(sorted_indices)}

        self.n_step_input = 1

        self._plot_adapter = InterpolatorMultiOutPlotAdapter(self)

        self.fill_metadata(self.metadata)

    def fill_metadata(self, metadata: dict) -> None:
        for dataset_name in self.dataset_names:
            ts = metadata["metadata_inference"][dataset_name]["timesteps"]
            ts["input_relative_date_indices"] = self.boundary_times
            ts["output_relative_date_indices"] = self.interp_times

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        x_bound = {}
        y = {}
        for dataset_name, dataset_batch in batch.items():
            x_bound[dataset_name] = dataset_batch[:, itemgetter(*self.boundary_times)(self.imap)][
                ...,
                self.data_indices[dataset_name].data.input.full,
            ]  # (bs, time, ens, latlon, nvar)

            y[dataset_name] = dataset_batch[:, itemgetter(*self.interp_times)(self.imap)][
                ...,
                self.data_indices[dataset_name].data.output.full,
            ]

        loss = torch.zeros(1, dtype=next(iter(batch.values())).dtype, device=self.device, requires_grad=False)

        y_pred = self(x_bound)

        loss, metrics, _ = checkpoint(
            self.compute_loss_metrics,
            y_pred,
            y,
            validation_mode=validation_mode,
            use_reentrant=False,
        )

        # All tasks return (loss, metrics, list of per-step dicts) for consistent plot callback contract.
        return loss, metrics, [y_pred]

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

import torch

from anemoi.training.train.tasks.base import BaseGraphModule

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.schemas.base_schema import BaseSchema

LOGGER = logging.getLogger(__name__)


class GraphDownscaler(BaseGraphModule):
    """Graph neural network downscaler for PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: BaseSchema,
        graph_data: dict[str, HeteroData],
        truncation_data: dict,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: dict[str, IndexCollection],
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network forecaster.

        Parameters
        ----------
        config : BaseSchema
            Configuration object
        graph_data : dict[str, HeteroData]
            Dictionary of graph data for each dataset
        truncation_data : dict
            Truncation configuration
        statistics : dict
            Training statistics
        statistics_tendencies : dict
            Tendency statistics
        data_indices : dict[str, IndexCollection]
            Data indices for each dataset
        metadata : dict
            Metadata
        supporting_arrays : dict
            Supporting arrays

        """
        super().__init__(
            config=config,
            graph_data=graph_data,
            truncation_data=truncation_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        # Multi-dataset setup (always expect dict inputs now)
        self.dataset_names = list(graph_data.keys())
        LOGGER.info("Forecaster initialized with datasets: %s", self.dataset_names)

    def _step(
        self,
        batch: dict,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:

        batch_dtype = next(iter(batch.values())).dtype
        loss = torch.zeros(1, dtype=batch_dtype, device=self.device, requires_grad=False)

        x = self.get_inputs(batch, sample_length=self.multi_step)

        y_pred = self(x)

        y = self.get_targets(batch, lead_step=self.multi_step - 1)

        # y includes the auxiliary variables, so we must leave those out when computing the loss
        # Compute loss for each dataset and sum them up
        loss, metrics = self.compute_loss_metrics(y_pred, y, 0, validation_mode=validation_mode)

        return loss, metrics, y_pred

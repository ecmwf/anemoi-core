# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import uuid

import torch
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from torch_geometric.data import HeteroData

from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.preprocessing import Processors


class AnemoiModelInterface(torch.nn.Module):
    """Anemoi model on torch level."""

    def __init__(
        self, *, config: DotDict, graph_data: HeteroData, statistics: dict, data_indices: dict, metadata: dict
    ) -> None:
        super().__init__()
        self.config = config
        self.id = str(uuid.uuid4())
        self.multi_step = self.config.training.multistep_input
        self.graph_data = graph_data
        self.statistics = statistics
        self.metadata = metadata
        self.data_indices = data_indices
        self._build_model()

    def _build_model(self) -> None:
        """Build the model and pre- and post-processors."""
        # Instantiate processors
        processors = [
            [name, instantiate(processor, statistics=self.statistics, data_indices=self.data_indices)]
            for name, processor in self.config.data.processors.items()
        ]

        # Assign the processor list pre- and post-processors
        self.pre_processors = Processors(processors)
        self.post_processors = Processors(processors, inverse=True)

        # Instantiate the model (Can be generalised to other models in the future, here we use AnemoiModelEncProcDec)
        self.model = AnemoiModelEncProcDec(
            config=self.config, data_indices=self.data_indices, graph_data=self.graph_data
        )

        # Use the forward method of the model directly
        self.forward = self.model.forward

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Prediction step for the model.

        Parameters
        ----------
        batch : torch.Tensor
            Input batched data.

        Returns
        -------
        torch.Tensor
            Predicted data.
        """
        batch = self.pre_processors(batch, in_place=False)

        with torch.no_grad():

            assert (
                len(batch.shape) == 4
            ), f"The input tensor has an incorrect shape: expected a 4-dimensional tensor, got {batch.shape}!"
            # Dimensions are
            # batch, timesteps, horizonal space, variables
            x = batch[:, 0 : self.multi_step, None, ...]  # add dummy ensemble dimension as 3rd index

            y_hat = self(x)

        return self.post_processors(y_hat, in_place=False)

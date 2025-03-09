# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import uuid
from typing import Optional

import torch
from anemoi.training.utils.debug_hydra import instantiate_debug
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.preprocessing import Processors


class AnemoiModelReconstructInterface(torch.nn.Module):
    """An interface for Anemoi reconstruction models.

    This class wraps around a Anemoi model and includes pre-processing and post-processing steps.
    It is specialized for reconstruction tasks and inherits from PyTorch’s Module class.

    Attributes
    ----------
    config : DotDict
        Configuration settings for the model.
    id : str
        A unique identifier for the model instance.
    graph_data : HeteroData
        Graph data for the model.
    statistics : dict
        Statistics for the data.
    metadata : dict
        Metadata for the model.
    data_indices : dict
        Indices for the data.
    model : AnemoiVAE (or any other reconstruction model)
        The underlying Anemoi reconstruction model.
    """

    def __init__(
        self,
        *,
        config: DotDict,
        graph_data: HeteroData,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: dict,
        metadata: dict,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.id = str(uuid.uuid4())
        self.graph_data = graph_data
        self.statistics = statistics
        self.statistics_tendencies = statistics_tendencies

        self.metadata = metadata
        self.data_indices = data_indices
        self._build_model()

    def _build_model(self) -> None:
        """Builds the model and pre- and post-processors."""
        # Instantiate processors for reconstruction task
        processors_state = [
            [name, instantiate(processor, statistics=self.statistics, data_indices=self.data_indices)]
            for name, processor in self.config.data.processors.state.items()
        ]
        self.pre_processors_state = Processors(processors_state)

        # create the post-processors for the state
        if not hasattr(self.config.data, "processors_reverse"):
            self.post_processors_state = Processors(processors_state, inverse=True)
        else:

            processors_state_reverse = [
                [
                    name,
                    instantiate(
                        processor,
                        statistics=self.statistics,
                        statistics_tendencies=self.statistics_tendencies,
                        data_indices=self.data_indices,
                    ),
                ]
                for name, processor in self.config.data.processors_reverse.state.items()
            ]
            self.post_processors_state = Processors(processors_state_reverse, inverse=True)

        # Instantiate the model (generalized for reconstruction tasks)
        # self.model = instantiate(
        self.model = instantiate_debug(
            self.config.model.model,  # Can point to AnemoiVAE or any other model
            self.config,
            data_indices=self.data_indices,
            graph_data=self.graph_data,
            # _recursive_=False,
            # _convert_=False,
        )

        self.boundings = torch.nn.ModuleList(
            [
                instantiate(cfg, name_to_index=self.data_indices.model.output.name_to_index)
                for cfg in getattr(self.config.model, "bounding", [])
            ]
        )

    def forward(self, x: torch.Tensor, model_comm_group: Optional[ProcessGroup] = None) -> dict:
        """Forward pass for reconstruction task."""

        dict_model_output = self.model(x, model_comm_group=model_comm_group)
        return dict_model_output

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Prediction step for the reconstruction task."""
        with torch.no_grad():
            assert len(batch.shape) == 5, f"Expected a 5D tensor, got {batch.shape}!"

            # Pre-process the input data
            x = self.pre_processors_state(batch, in_place=False)

            # Pass through the model (reconstruction task)
            y_hat = self(x)

            # Post-process the reconstructed output
            y_hat = self.post_processors_state(y_hat, in_place=False)

        return y_hat

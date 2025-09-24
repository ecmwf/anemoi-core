# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import uuid
from typing import Optional

import torch
from hydra.utils import instantiate
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.models.multi_model import AnemoiMultiModel
from anemoi.utils.config import DotDict


class AnemoiModelInterface(torch.nn.Module):
    """An interface for Anemoi models.

    This class is a wrapper around the Anemoi model that includes pre-processing and post-processing steps.
    It inherits from the PyTorch Module class.

    Attributes
    ----------
    config : DotDict
        Configuration settings for the model.
    id : str
        A unique identifier for the model instance.
    multi_step : bool
        Whether the model uses multi-step input.
    graph_data : HeteroData
        Graph data for the model.
    statistics : dict
        Statistics for the data.
    metadata : dict
        Metadata for the model.
    supporting_arrays : dict
        Numpy arraysto store in the checkpoint.
    data_indices : dict
        Indices for the data.
    pre_processors : Processors
        Pre-processing steps to apply to the data before passing it to the model.
    post_processors : Processors
        Post-processing steps to apply to the model's output.
    model : AnemoiModelEncProcDec
        The underlying Anemoi model.
    """

    def __init__(
        self,
        *,
        config: DotDict,
        sample_static_info,
        # graph_data: HeteroData,
        # data_indices: dict,
        metadata: dict,
    ) -> None:

        assert False, "not used anymore. Anemoi models have a _predict_step method for inference"

        super().__init__()
        self.id = str(uuid.uuid4())

        self.config = config
        self.sample_static_info = sample_static_info
        # self.graph_data = graph_data
        self.metadata = metadata
        self.supporting_arrays = {}
        self._build_model()

    def _build_model(self) -> None:
        """Builds the model and pre- and post-processors."""

        # Instantiate the model
        self.model = AnemoiMultiModel(
            # self.config.model.model,
            model_config=self.config,
            sample_static_info=self.sample_static_info,
            # graph_data=self.graph_data,
            # truncation_data=self.truncation_data,
            # _recursive_=False,  # Disables recursive instantiation by Hydra
        )

        # Use the forward method of the model directly
        self.forward = self.model.forward

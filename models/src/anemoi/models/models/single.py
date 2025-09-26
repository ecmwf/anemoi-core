# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import einops
import torch

from anemoi.models.models import AnemoiMultiModel

# from anemoi.models.preprocessing.normalisers import build_normaliser
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


NODE_COORDS_NDIMS = 4  # cos_lat, sin_lat, cos_lon, sin_lon
EDGE_ATTR_NDIM = 3  # edge_length, edge_dir0, edge_dir1


INPUT_FIELDS_NAME = "data"
OUTPUT_FIELDS_NAME = "data"


class BaseAnemoiSingleModel(AnemoiMultiModel):
    """Message passing graph neural network."""

    name = None

    def __init__(self, *, model_config: DotDict, **kwargs) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        model_config : DotDict
            Model configuration
        graph_data : HeteroData
            Graph definition
        """
        super().__init__(model_config=model_config, **kwargs)


class AnemoiSingleModel(BaseAnemoiSingleModel):
    pass


class ToyAnemoiSingleModel(BaseAnemoiSingleModel):

    def build(self):
        self.num_input_channels = len(self.input_metadata.first["name_to_index"])
        self.num_output_channels = len(self.output_metadata.first["name_to_index"])

        self.linear = torch.nn.Linear(self.num_input_channels, self.num_output_channels)

    def forward(self, x, *args, **kwargs):
        print(f"Warning: Ignoring {len(args)} args, and {len(kwargs)} kwargs in forward")
        assert len(x) == 1, (f"This toy model only supports one input, got {x.keys()}", x)
        print(x.to_str("x input to ToyAnemoiSingleModel in forward"))

        data = x[INPUT_FIELDS_NAME]["data"]
        data = einops.rearrange(data, "batch offsets variables values -> batch offsets values variables")

        # todo : define the model correctly to avoid the need to rearrange y_pred
        pred = self.linear(data)
        pred = einops.rearrange(pred, "batch offsets values variables -> batch offsets variables values")

        output = self.output_metadata.new_empty()
        output[OUTPUT_FIELDS_NAME] = dict(data=pred)
        print(output.to_str("output"))
        print("❤️🆗----- End of Forward pass -----")
        return output

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

from anemoi.models.data_structure.structure import TreeDict
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

    def _TODO_prepare_input(self, x: TreeDict) -> TreeDict:

        def merge(*leaves):
            latitudes = None
            longitudes = None
            dimensions_order = None
            for leaf in leaves:
                if latitudes is None:
                    latitudes = leaf["latitudes"]
                    longitudes = leaf["longitudes"]
                    dimensions_order = leaf["dimensions_order"]
                assert leaf["latitudes"] == latitudes, f"latitudes do not match: {leaf['latitudes']} != {latitudes}"
                assert (
                    leaf["longitudes"] == longitudes
                ), f"longitudes do not match: {leaf['longitudes']} != {longitudes}"
                assert (
                    leaf["dimensions_order"] == dimensions_order
                ), f"dimensions_order do not match: {leaf['dimensions_order']} != {dimensions_order}"

            return dict(
                data=torch.stack([leaf["data"] for leaf in leaves], dim=-1),
                latitudes=latitudes,
                longitudes=longitudes,
                dimensions_order=dimensions_order + ("offsets",),
                _offsets=(leaf["_offset"] for leaf in leaves),
            )

        res = x.new_empty()
        for k, v in self.input_metadata.get_level_minus_one_leaf_nodes():
            for _ in v.values():
                assert not isinstance(_, TreeDict), f"Only leaf nodes supported, got {type(_)}"
            res[k] = merge(v.values())

        return res

        def merge(*leaves):
            latitudes = None
            longitudes = None
            dimensions_order = None
            for leaf in leaves:
                if latitudes is None:
                    latitudes = leaf["latitudes"]
                    longitudes = leaf["longitudes"]
                    dimensions_order = leaf["dimensions_order"]
                assert leaf["latitudes"] == latitudes, f"latitudes do not match: {leaf['latitudes']} != {latitudes}"
                assert (
                    leaf["longitudes"] == longitudes
                ), f"longitudes do not match: {leaf['longitudes']} != {longitudes}"
                assert (
                    leaf["dimensions_order"] == dimensions_order
                ), f"dimensions_order do not match: {leaf['dimensions_order']} != {dimensions_order}"

            return dict(
                data=torch.stack([leaf["data"] for leaf in leaves], dim=-1),
                latitudes=latitudes,
                longitudes=longitudes,
                dimensions_order=dimensions_order + ("offsets",),
                _offsets=(leaf["_offset"] for leaf in leaves),
            )

        res = x.new_empty()
        assert False
        for k, v in self.sample_static_info.get_level_minus_one_leaf_nodes():
            for _ in v.values():
                assert not isinstance(_, TreeDict), f"Only leaf nodes supported, got {type(_)}"
            res[k] = merge(v.values())

        return res


class AnemoiSingleModel(BaseAnemoiSingleModel):
    pass


class ToyAnemoiSingleModel(BaseAnemoiSingleModel):

    def prepare_input(self, x):
        assert len(x) == 1, f"This toy model only supports one input, got {x.keys()}"
        box = x[INPUT_FIELDS_NAME]

        data = [box["data"] for _, box in x.items()]  # multiple time steps
        data = torch.cat(data, dim=-1)

        name_to_index = box.first["name_to_index"]

        new_box = dict(data=data, name_to_index=name_to_index)

        return TreeDict({INPUT_FIELDS_NAME: new_box})

    def build(self):
        self.num_input_channels = len(self.input_metadata[INPUT_FIELDS_NAME].first["name_to_index"])
        self.num_output_channels = len(self.output_metadata[OUTPUT_FIELDS_NAME].first["name_to_index"])

        self.linear = torch.nn.Linear(self.num_input_channels, self.num_output_channels)

    def forward(self, x, *args, **kwargs):
        x = self.prepare_input(x)

        print(f"Warning: Ignoring {len(args)} args, and {len(kwargs)} kwargs in forward")
        assert len(x) == 1, (f"This toy model only supports one input, got {x.keys()}", x)
        print(x.to_str("x input to ToyAnemoiSingleModel in forward"))

        data = x[INPUT_FIELDS_NAME]["data"]
        data = einops.rearrange(data, "batch variables values -> batch values variables")

        pred = self.linear(data)

        output = self.output_metadata.new_empty()
        output[OUTPUT_FIELDS_NAME] = dict(data=pred)
        print(output.to_str("output"))
        print("❤️🆗----- End of Forward pass -----")
        return output

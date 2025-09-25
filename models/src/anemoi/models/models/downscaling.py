# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

from anemoi.models.models import AnemoiMultiModel


class BaseAnemoiDownscalingModel(AnemoiMultiModel):
    name = "downscaling"


class AnemoiDownscalingModel(BaseAnemoiDownscalingModel):
    name = "downscaling"


class ToyAnemoiDownscalingModel(BaseAnemoiDownscalingModel):
    name = "downscaling"

    def build(self):
        assert len(self.target_metadata) == 1, "This toy model only supports one target key"

        self.num_input_channels = len(self.input_metadata["high_res"]["name_to_index"])
        self.num_output_channels = len(self.target_metadata.first["name_to_index"])

        self.linear = torch.nn.Linear(self.num_input_channels, self.num_output_channels)

    def forward(self, x, *args, **kwargs):
        print(f"----- Start of Forward pass of {self.__class__.__name__} -----")
        print(f"Ignoring args: {args}")
        print(f"Ignoring kwargs: {kwargs}")
        import einops

        print(self.input_metadata.to_str("input metadata"))
        print(self.output_metadata.to_str("output metadata"))

        print(x.to_str("input x"))
        data = x["high_res"]["data"]

        data = einops.rearrange(data, "b variables values -> b values variables")
        pred = self.linear(data)
        pred = einops.rearrange(pred, "b values variables -> b variables values")

        n_variables = len(self.target_metadata.first["name_to_index"])
        assert pred.shape[1] == n_variables, f"Expected output shape {n_variables}, got {pred.shape[1]}"

        print("❤️🆗----- End of Forward pass of ToyAnemoiDownscalingModel -----")
        res = x.new_empty()
        box = self.target_metadata["high_res"].copy()
        box["data"] = pred
        res["high_res"] = box
        return res

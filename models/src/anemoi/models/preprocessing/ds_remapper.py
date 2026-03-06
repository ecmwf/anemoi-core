# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Optional

import torch
from torch import nn
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import BasePreprocessor
from anemoi.models.preprocessing.mappings import boxcox_converter
from anemoi.models.preprocessing.mappings import expm1_converter
from anemoi.models.preprocessing.mappings import inverse_boxcox_converter
from anemoi.models.preprocessing.mappings import log1p_converter
from anemoi.models.preprocessing.mappings import noop
from anemoi.models.preprocessing.mappings import sqrt_converter
from anemoi.models.preprocessing.mappings import square_converter

LOGGER = logging.getLogger(__name__)

class TopRemapper(BasePreprocessor):
    """Top-level normalizer for input, output data."""

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        """Initialize the remapper.

        Parameters
        ----------
        config : DotConfig
            Configuration object
        statistics : Dicts
            Statistics for input, output data
        data_indices : dict
        """
        super().__init__(config, statistics, data_indices)

        self.remappers = {}
        # this two-step process is done to allow for casting to the correct device
        # alternative is to install tensordict, or use ModuleList
        self.remapper_input = FieldRemapper(
            config=config,
            statistics=statistics[0],
            data_indices_ds=data_indices.data.input[0],
            dataset="input_lres",
            methods=self.methods,
            default=self.default,
            remap=self.remap,
            normalizer=self.normalizer,
        )
        self.remappers["input_lres"] = self.remapper_input 

        self.remapper_input_hres = FieldRemapper(
            config=config,
            statistics=statistics[1],
            data_indices_ds=data_indices.data.input[1],
            dataset="input_hres",
            methods=self.methods,
            default=self.default,
            remap=self.remap,
            normalizer=self.normalizer,
        )
        self.remappers["input_hres"] = self.remapper_input_hres
        self.remapper_output = FieldRemapper(
            config=config,
            statistics=statistics[2],
            data_indices_ds=data_indices.data.output,
            dataset="output",
            methods=self.methods,
            default=self.default,
            remap=self.remap,
            normalizer=self.normalizer,
        )
        self.remappers["output"] = self.remapper_output

    def forward(
        self,
        x: torch.Tensor,
        dataset: str,
        in_place: bool = True,
        inverse: bool = False,
    ) -> torch.Tensor:
        if inverse:
            return self.inverse_transform(x, dataset, in_place=in_place)
        return self.transform(x, dataset, in_place=in_place)

    def transform(self, data, dataset: str, in_place: bool = True):
        try:
            remapper = self.remappers[dataset]
        except ValueError:
            raise ValueError(f"No remapper found for dataset type: {dataset}")
        return remapper.transform(data, in_place=in_place)

    def inverse_transform(
        self,
        data,
        dataset: str,
        in_place: bool = True,
        data_index: Optional[torch.Tensor] = None,
    ):
        try:
            remapper = self.remappers[dataset]
        except ValueError:
            raise ValueError(f"No remapper found for dataset type: {dataset}")
        return remapper.inverse_transform(
            data, in_place=in_place, #data_index=data_index
        )

class FieldRemapper(nn.Module):
    """Remap and convert variables for single variables."""

    supported_methods = {
        method: [f, inv]
        for method, f, inv in zip(
            ["log1p", "sqrt", "boxcox", "none"],
            [log1p_converter, sqrt_converter, boxcox_converter, noop],
            [expm1_converter, square_converter, inverse_boxcox_converter, noop],
        )
    }

    def __init__(
        self,
        config=None,
        data_indices_ds: Optional[IndexCollection] = None,
        dataset: str = None,
        statistics: Optional[dict] = None,
        methods: str = "none",
        method_kwargs: Optional[dict] = None,
        default: str = "none",
        remap: dict = {},
        normalizer: str = "none",
    ) -> None:
        super().__init__()
        self.methods = methods
        self.dataset = dataset
        self.default = default
        self.remap = remap
        self.normalizer = normalizer
        self._create_remapping_indices(data_indices_ds, statistics)
        self._validate_indices()

    def _validate_indices(self):
        assert (
            len(self.index_training_input)
            == len(self.index_inference_input)
            == len(self.index_inference_output)
            == len(self.index_training_out)
            == len(self.remappers)
        ), (
            f"Error creating conversion indices {len(self.index_training_input)}, "
            f"{len(self.index_inference_input)}, {len(self.index_training_input)}, {len(self.index_training_out)}, {len(self.remappers)}"
        )

    def _create_remapping_indices(
        self,
        data_indices_ds,
        statistics=None,
    ):
        """Create the parameter indices for remapping."""
        # list for training and inference mode as position of parameters can change
        name_to_index_training_input = data_indices_ds.name_to_index
        name_to_index_inference_input = data_indices_ds.name_to_index
        name_to_index_training_output = data_indices_ds.name_to_index
        name_to_index_inference_output = data_indices_ds.name_to_index
        self.num_training_input_vars = len(name_to_index_training_input)
        self.num_inference_input_vars = len(name_to_index_inference_input)
        self.num_training_output_vars = len(name_to_index_training_output)
        self.num_inference_output_vars = len(name_to_index_inference_output)

        (
            self.remappers,
            self.backmappers,
            self.index_training_input,
            self.index_training_out,
            self.index_inference_input,
            self.index_inference_output,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        # Create parameter indices for remapping variables
        for name in name_to_index_training_input:
            method = self.methods.get(name, self.default)
            if method in self.supported_methods:
                self.remappers.append(self.supported_methods[method][0])
                self.backmappers.append(self.supported_methods[method][1])
                self.index_training_input.append(name_to_index_training_input[name])
                if name in name_to_index_training_output:
                    self.index_training_out.append(name_to_index_training_output[name])
                else:
                    self.index_training_out.append(None)
                if name in name_to_index_inference_input:
                    self.index_inference_input.append(name_to_index_inference_input[name])
                else:
                    self.index_inference_input.append(None)
                if name in name_to_index_inference_output:
                    self.index_inference_output.append(name_to_index_inference_output[name])
                else:
                    # this is a forcing variable. It is not in the inference output.
                    self.index_inference_output.append(None)
            else:
                raise KeyError(f"Unknown remapping method for {name}: {method}")
        self.register_buffer(
            f"_{self.dataset}_idx", data_indices_ds.full, persistent=True
        )

    def transform(self, x, in_place: bool = True) -> torch.Tensor:
        if not in_place:
            x = x.clone()
        if x.shape[-1] == self.num_training_input_vars:
            idx = self.index_training_input
        elif x.shape[-1] == self.num_inference_input_vars:
            idx = self.index_inference_input
        else:
            raise ValueError(
                f"Input tensor ({x.shape[-1]}) does not match the training "
                f"({self.num_training_input_vars}) or inference shape ({self.num_inference_input_vars})",
            )
        for i, remapper in zip(idx, self.remappers):
            if i is not None:
                x[..., i] = remapper(x[..., i])
        return x

    def inverse_transform(self, x, in_place: bool = True) -> torch.Tensor:
        if not in_place:
            x = x.clone()
        if x.shape[-1] == self.num_training_output_vars:
            idx = self.index_training_out
        elif x.shape[-1] == self.num_inference_output_vars:
            idx = self.index_inference_output
        else:
            raise ValueError(
                f"Input tensor ({x.shape[-1]}) does not match the training "
                f"({self.num_training_output_vars}) or inference shape ({self.num_inference_output_vars})",
            )
        for i, backmapper in zip(idx, self.backmappers):
            if i is not None:
                x[..., i] = backmapper(x[..., i])
        return x

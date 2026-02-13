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


class Remapper(BasePreprocessor):
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
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        super().__init__(config, data_indices, statistics)
        self._create_remapping_indices(statistics)
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
        statistics=None,
    ):
        """Create the parameter indices for remapping."""
        # list for training and inference mode as position of parameters can change
        name_to_index_training_input = self.data_indices.data.input.name_to_index
        name_to_index_inference_input = self.data_indices.model.input.name_to_index
        name_to_index_training_output = self.data_indices.data.output.name_to_index
        name_to_index_inference_output = self.data_indices.model.output.name_to_index
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

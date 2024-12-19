# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from abc import ABC
from typing import Optional

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import BasePreprocessor
from anemoi.models.preprocessing.mappings import atan2_converter
from anemoi.models.preprocessing.mappings import cos_converter
from anemoi.models.preprocessing.mappings import sin_converter

LOGGER = logging.getLogger(__name__)


class Multimapper(BasePreprocessor, ABC):
    """Remap single variable to 2 or more variables, or the other way around.

    cos_sin:
        Remap the variable to cosine and sine.
        Map output back to degrees.

    ```
    cos_sin:
      "mwd" : ["cos_mwd", "sin_mwd"]
    ```
    """

    supported_methods = {
        method: [f, inv]
        for method, f, inv in zip(
            ["cos_sin"],
            [[cos_converter, sin_converter]],
            [atan2_converter],
        )
    }

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        """Initialize the remapper.

        Parameters
        ----------
        config : DotDict
            configuration object of the processor
        data_indices : IndexCollection
            Data indices for input and output variables
        statistics : dict
            Data statistics dictionary
        """
        super().__init__(config, data_indices, statistics)
        self.printed_preprocessor_warning, self.printed_postprocessor_warning = False, False
        self._create_remapping_indices(statistics)
        self._validate_indices()

    def _validate_indices(self):
        assert len(self.index_training_input) == len(self.index_inference_input) <= len(self.remappers), (
            f"Error creating conversion indices {len(self.index_training_input)}, "
            f"{len(self.index_inference_input)}, {len(self.remappers)}"
        )
        assert len(self.index_training_output) == len(self.index_inference_output) <= len(self.remappers), (
            f"Error creating conversion indices {len(self.index_training_output)}, "
            f"{len(self.index_inference_output)}, {len(self.remappers)}"
        )
        assert len(set(self.index_training_input + self.indices_keep_training_input)) == self.num_training_input_vars, (
            "Error creating conversion indices: variables remapped in config.data.remapped "
            "that have no remapping function defined. Preprocessed tensors contains empty columns."
        )

    def _create_remapping_indices(
        self,
        statistics=None,
    ):
        """Create the parameter indices for remapping."""
        # list for training and inference mode as position of parameters can change
        name_to_index_training_input = self.data_indices.data.input.name_to_index
        name_to_index_inference_input = self.data_indices.model.input.name_to_index
        name_to_index_training_remapped_input = self.data_indices.internal_data.input.name_to_index
        name_to_index_inference_remapped_input = self.data_indices.internal_model.input.name_to_index
        name_to_index_training_remapped_output = self.data_indices.internal_data.output.name_to_index
        name_to_index_inference_remapped_output = self.data_indices.internal_model.output.name_to_index
        name_to_index_training_output = self.data_indices.data.output.name_to_index
        name_to_index_inference_output = self.data_indices.model.output.name_to_index

        self.num_training_input_vars = len(name_to_index_training_input)
        self.num_inference_input_vars = len(name_to_index_inference_input)
        self.num_remapped_training_input_vars = len(name_to_index_training_remapped_input)
        self.num_remapped_inference_input_vars = len(name_to_index_inference_remapped_input)
        self.num_remapped_training_output_vars = len(name_to_index_training_remapped_output)
        self.num_remapped_inference_output_vars = len(name_to_index_inference_remapped_output)
        self.num_training_output_vars = len(name_to_index_training_output)
        self.num_inference_output_vars = len(name_to_index_inference_output)
        self.indices_keep_training_input = []
        for key, item in self.data_indices.data.input.name_to_index.items():
            if key in self.data_indices.internal_data.input.name_to_index:
                self.indices_keep_training_input.append(item)
        self.indices_keep_inference_input = []
        for key, item in self.data_indices.model.input.name_to_index.items():
            if key in self.data_indices.internal_model.input.name_to_index:
                self.indices_keep_inference_input.append(item)
        self.indices_keep_training_output = []
        for key, item in self.data_indices.data.output.name_to_index.items():
            if key in self.data_indices.internal_data.output.name_to_index:
                self.indices_keep_training_output.append(item)
        self.indices_keep_inference_output = []
        for key, item in self.data_indices.model.output.name_to_index.items():
            if key in self.data_indices.internal_model.output.name_to_index:
                self.indices_keep_inference_output.append(item)

        (
            self.index_training_input,
            self.index_training_remapped_input,
            self.index_inference_input,
            self.index_inference_remapped_input,
            self.index_training_output,
            self.index_training_backmapped_output,
            self.index_inference_output,
            self.index_inference_backmapped_output,
            self.remappers,
            self.backmappers,
        ) = ([], [], [], [], [], [], [], [], [], [])

        # Create parameter indices for remapping variables
        for name in name_to_index_training_input:

            method = self.methods.get(name, self.default)

            if method == "none":
                continue

            if method == "cos_sin":
                self.index_training_input.append(name_to_index_training_input[name])
                self.index_training_output.append(name_to_index_training_output[name])
                self.index_inference_input.append(name_to_index_inference_input[name])
                if name in name_to_index_inference_output:
                    self.index_inference_output.append(name_to_index_inference_output[name])
                else:
                    # this is a forcing variable. It is not in the inference output.
                    self.index_inference_output.append(None)
                multiple_training_output, multiple_inference_output = [], []
                multiple_training_input, multiple_inference_input = [], []
                for name_dst in self.method_config[method][name]:
                    assert name_dst in self.data_indices.internal_data.input.name_to_index, (
                        f"Trying to remap {name} to {name_dst}, but {name_dst} not a variable. "
                        f"Remap {name} to {name_dst} in config.data.remapped. "
                    )
                    multiple_training_input.append(name_to_index_training_remapped_input[name_dst])
                    multiple_training_output.append(name_to_index_training_remapped_output[name_dst])
                    multiple_inference_input.append(name_to_index_inference_remapped_input[name_dst])
                    if name_dst in name_to_index_inference_remapped_output:
                        multiple_inference_output.append(name_to_index_inference_remapped_output[name_dst])
                    else:
                        # this is a forcing variable. It is not in the inference output.
                        multiple_inference_output.append(None)

                self.index_training_remapped_input.append(multiple_training_input)
                self.index_inference_remapped_input.append(multiple_inference_input)
                self.index_training_backmapped_output.append(multiple_training_output)
                self.index_inference_backmapped_output.append(multiple_inference_output)

                self.remappers.append([cos_converter, sin_converter])
                self.backmappers.append(atan2_converter)

                LOGGER.info(f"Map {name} to cosine and sine and save result in {self.method_config[method][name]}.")

            else:
                raise ValueError[f"Unknown remapping method for {name}: {method}"]

    def transform(self, x: torch.Tensor, in_place: bool = True) -> torch.Tensor:
        """Remap and convert the input tensor.

        ```
        x : torch.Tensor
            Input tensor
        in_place : bool
            Whether to process the tensor in place.
            in_place is not possible for this preprocessor.
        ```
        """
        # Choose correct index based on number of variables
        if x.shape[-1] == self.num_training_input_vars:
            index = self.index_training_input
            indices_remapped = self.index_training_remapped_input
            indices_keep = self.indices_keep_training_input
            target_number_columns = self.num_remapped_training_input_vars

        elif x.shape[-1] == self.num_inference_input_vars:
            index = self.index_inference_input
            indices_remapped = self.index_inference_remapped_input
            indices_keep = self.indices_keep_inference_input
            target_number_columns = self.num_remapped_inference_input_vars

        else:
            raise ValueError(
                f"Input tensor ({x.shape[-1]}) does not match the training "
                f"({self.num_training_input_vars}) or inference shape ({self.num_inference_input_vars})",
            )

        # create new tensor with target number of columns
        x_remapped = torch.zeros(x.shape[:-1] + (target_number_columns,), dtype=x.dtype, device=x.device)
        if in_place and not self.printed_preprocessor_warning:
            LOGGER.warning(
                "Remapper (preprocessor) called with in_place=True. This preprocessor cannot be applied in_place as new columns are added to the tensors.",
            )
            self.printed_preprocessor_warning = True

        # copy variables that are not remapped
        x_remapped[..., : len(indices_keep)] = x[..., indices_keep]

        # Remap variables
        for idx_dst, remapper, idx_src in zip(indices_remapped, self.remappers, index):
            if idx_src is not None:
                for jj, ii in enumerate(idx_dst):
                    x_remapped[..., ii] = remapper[jj](x[..., idx_src])

        return x_remapped

    def inverse_transform(self, x: torch.Tensor, in_place: bool = True) -> torch.Tensor:
        """Convert and remap the output tensor.

        ```
        x : torch.Tensor
            Input tensor
        in_place : bool
            Whether to process the tensor in place.
            in_place is not possible for this postprocessor.
        ```
        """
        # Choose correct index based on number of variables
        if x.shape[-1] == self.num_remapped_training_output_vars:
            index = self.index_training_output
            indices_remapped = self.index_training_backmapped_output
            indices_keep = self.indices_keep_training_output
            target_number_columns = self.num_training_output_vars

        elif x.shape[-1] == self.num_remapped_inference_output_vars:
            index = self.index_inference_output
            indices_remapped = self.index_inference_backmapped_output
            indices_keep = self.indices_keep_inference_output
            target_number_columns = self.num_inference_output_vars

        else:
            raise ValueError(
                f"Input tensor ({x.shape[-1]}) does not match the training "
                f"({self.num_remapped_training_output_vars}) or inference shape ({self.num_remapped_inference_output_vars})",
            )

        # create new tensor with target number of columns
        x_remapped = torch.zeros(x.shape[:-1] + (target_number_columns,), dtype=x.dtype, device=x.device)
        if in_place and not self.printed_postprocessor_warning:
            LOGGER.warning(
                "Remapper (preprocessor) called with in_place=True. This preprocessor cannot be applied in_place as new columns are added to the tensors.",
            )
            self.printed_postprocessor_warning = True

        # copy variables that are not remapped
        x_remapped[..., indices_keep] = x[..., : len(indices_keep)]

        # Backmap variables
        for idx_dst, backmapper, idx_src in zip(index, self.backmappers, indices_remapped):
            if idx_dst is not None:
                x_remapped[..., idx_dst] = backmapper(x[..., idx_src])

        return x_remapped

    def transform_loss_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Remap the loss mask.

        ```
        x : torch.Tensor
            Loss mask
        ```
        """
        # use indices at model output level
        index = self.index_inference_backmapped_output
        indices_remapped = self.index_inference_output
        indices_keep = self.indices_keep_inference_output

        # create new loss mask with target number of columns
        mask_remapped = torch.zeros(
            mask.shape[:-1] + (mask.shape[-1] + len(indices_remapped),), dtype=mask.dtype, device=mask.device
        )

        # copy loss mask for variables that are not remapped
        mask_remapped[..., : len(indices_keep)] = mask[..., indices_keep]

        # remap loss mask for rest of variables
        for idx_src, idx_dst in zip(indices_remapped, index):
            if idx_dst is not None:
                for ii in idx_dst:
                    mask_remapped[..., ii] = mask[..., idx_src]

        return mask_remapped

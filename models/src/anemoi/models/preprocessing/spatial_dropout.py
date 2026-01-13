# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Random spatial dropout for data augmentation."""

import logging
from typing import Optional

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import BasePreprocessor

LOGGER = logging.getLogger(__name__)


class RandomSpatialDropout(BasePreprocessor):
    """Randomly drops (sets to NaN) a percentage of valid grid cells during training and validation.

    This preprocessor helps the model learn better spatial interpolation and
    generalization by creating artificial sparsity in the input data. It operates
    BEFORE imputation, so the imputer will fill the dropped values.

    Key features:
    - Operates during training and validation (distinguished from inference by tensor shape)
    - Only drops from originally valid (non-NaN) grid cells
    - Only affects input timesteps (not targets)
    - Can target specific variables or all non-forcing variables

    Configuration example:
    ```yaml
    spatial_dropout:
      _target_: anemoi.models.preprocessing.spatial_dropout.RandomSpatialDropout
      dropout_prob: 0.15  # Drop 15% of valid grid cells
      dropout_variables:  # Optional: specific variables to drop
        - z_500
        - z_850
      multi_step: 2  # Number of input timesteps (optional, defaults to 2)
    ```
    """

    @classmethod
    def _process_config(cls, config):
        """Override to add RandomSpatialDropout-specific special keys.

        Extends the base special keys to exclude preprocessor parameters
        from being treated as data processing strategies.
        """
        _special_keys = [
            "default",
            "remap",
            "normalizer",
            "dropout_prob",  # Probability of dropping each valid grid cell
            "dropout_variables",  # Optional list of variables to drop
            "multi_step",  # Number of input timesteps to apply dropout to
        ]

        default = config.get("default", "none")
        remap = config.get("remap", {})
        normalizer = config.get("normalizer", "none")
        method_config = {k: v for k, v in config.items() if k not in _special_keys and v is not None and v != "none"}

        if method_config:
            LOGGER.warning(
                f"{cls.__name__}: Unexpected config keys {list(method_config.keys())}. "
                f"This preprocessor only uses 'dropout_prob', 'dropout_variables', and 'multi_step'."
            )

        return default, remap, normalizer, method_config

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        """Initialize the random spatial dropout preprocessor.

        Parameters
        ----------
        config : DotDict
            Configuration object with dropout parameters:
            - dropout_prob: Probability of dropping each valid grid cell (0.0-1.0)
            - dropout_variables: Optional list of variable names to apply dropout to
            - multi_step: Number of input timesteps (default: 2)
        data_indices : IndexCollection
            Data indices for input variables
        statistics : dict
            Not used by this preprocessor, but required by base class
        """
        super().__init__(config, data_indices, statistics)

        # # Handle both dict configs (from _convert_: all) and object configs
        # if config is None:
        #     self.dropout_prob = 0.0
        #     self.multi_step = 2
        #     self.dropout_variables = None
        # elif isinstance(config, dict):
        #     self.dropout_prob = config.get("dropout_prob", 0.0)
        #     self.multi_step = config.get("multi_step", 2)
        #     self.dropout_variables = config.get("dropout_variables", None)
        # else:
        #     self.dropout_prob = getattr(config, "dropout_prob", 0.0)
        #     self.multi_step = getattr(config, "multi_step", 2)
        #     self.dropout_variables = getattr(config, "dropout_variables", None)

        # Get dropout probability
        self.dropout_prob = getattr(config, "dropout_prob", 0.0) if config is not None else 0.0

        if not 0.0 <= self.dropout_prob <= 1.0:
            raise ValueError(f"dropout_prob must be between 0.0 and 1.0, got {self.dropout_prob}")

        # Get multi_step from config, default to 2 if not specified
        self.multi_step = getattr(config, "multi_step", 2) if config is not None else 2

        # Get optional list of variables to apply dropout to
        self.dropout_variables = getattr(config, "dropout_variables", None) if config is not None else None

        if self.dropout_prob > 0:
            # Prepare indices for dropout
            name_to_index = self.data_indices.data.input.name_to_index
            forcing_names = set(getattr(self.data_indices, "forcing", []))

            # Get non-forcing variables
            non_forcing_names = [name for name in name_to_index.keys() if name not in forcing_names]

            if self.dropout_variables is None:
                # Drop from all non-forcing variables
                dropout_names = non_forcing_names
                var_desc = f"all {len(dropout_names)} non-forcing variables"
            else:
                # Drop only from specified variables
                dropout_names = [name for name in self.dropout_variables if name in non_forcing_names]
                missing = set(self.dropout_variables) - set(dropout_names)
                if missing:
                    LOGGER.warning(f"RandomSpatialDropout: Variables {missing} not found in non-forcing variables")
                var_desc = f"{len(dropout_names)} specified variables: {dropout_names}"

            # Convert to indices
            dropout_indices = [name_to_index[name] for name in dropout_names]

            self.register_buffer("dropout_indices", torch.tensor(dropout_indices, dtype=torch.long), persistent=False)

            LOGGER.info(
                f"RandomSpatialDropout: Will randomly drop {self.dropout_prob*100:.1f}% of valid grid cells "
                f"during training and validation in first {self.multi_step} input timesteps for {var_desc}"
            )
        else:
            # No dropout - register empty tensor
            self.register_buffer("dropout_indices", torch.tensor([], dtype=torch.long), persistent=False)
            LOGGER.info("RandomSpatialDropout: dropout_prob=0.0, no dropout will be applied")

    def transform(self, x: torch.Tensor, in_place: bool = True, **kwargs) -> torch.Tensor:
        """Apply random spatial dropout to input tensor.

        During training and validation, randomly sets a percentage of valid (non-NaN)
        grid cells to NaN for specified variables in the input timesteps.

        Automatically distinguishes between training/validation and inference by checking
        tensor shape: training/validation batches have multiple timesteps (inputs + targets),
        while inference batches typically have only input timesteps.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, time, ..., grid, variable)
            Multiple timesteps (training/validation) or few timesteps (inference)
        in_place : bool
            Whether to modify tensor in place (default: True)
        **kwargs : dict
            Additional keyword arguments (unused, for interface compatibility)

        Returns
        -------
        torch.Tensor
            Tensor with random dropout applied (training/validation) or unchanged (inference)
        """
        # Skip if dropout disabled or no variables to drop
        if self.dropout_prob == 0 or len(self.dropout_indices) == 0:
            return x if in_place else x.clone()

        # # Only apply dropout during training
        # if not self.training:
        #     return x if in_place else x.clone()

        # Distinguish training/validation from inference by tensor shape
        # Training/validation: batch has multiple timesteps (multi_step inputs + targets)
        # Inference: batch typically has only input timesteps (multi_step or fewer)
        # This check automatically handles all three modes without needing self.training or grad flags
        if x.ndim < 2 or x.shape[1] <= self.multi_step:
            # Not enough timesteps - this is inference mode, skip dropout
            return x if in_place else x.clone()

        if not in_place:
            x = x.clone()

        # Apply dropout to each variable and timestep (inputs only, not targets)
        for var_idx in self.dropout_indices:
            for t in range(min(self.multi_step, x.shape[1])):  # Ensure we don't exceed tensor size
                # Get current variable slice
                var_slice = x[:, t, ..., var_idx]

                # Identify valid (non-NaN) locations
                # We only drop from valid grid cells, never "undrop" NaNs
                valid_mask = ~torch.isnan(var_slice)

                # Generate random dropout mask for valid locations
                # Each valid grid cell has dropout_prob chance of being set to NaN
                dropout_mask = torch.rand(var_slice.shape, device=x.device, dtype=torch.float32) < self.dropout_prob

                # Combine: only drop from valid locations
                cells_to_drop = valid_mask & dropout_mask

                # Set selected cells to NaN
                if cells_to_drop.any():
                    x[:, t, ..., var_idx][cells_to_drop] = torch.nan  # Use torch.nan instead of float('nan')

        return x

    def inverse_transform(self, x: torch.Tensor, in_place: bool = True, **kwargs) -> torch.Tensor:
        """No-op: dropout is not reversible, outputs are unchanged."""
        return x if in_place else x.clone()

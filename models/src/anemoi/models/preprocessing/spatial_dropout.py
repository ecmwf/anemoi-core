# (C) Copyright 2024-2026 Anemoi contributors.
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
    """Randomly drops (sets to NaN) a percentage of valid grid cells during training.

    This preprocessor helps the model learn better spatial interpolation and
    generalization by creating artificial sparsity in the input data. It operates
    BEFORE imputation, so the imputer will fill the dropped values.

    Key features:
    - Operates during training only
    - Only drops from originally valid (non-NaN) grid cells
    - Only affects input timesteps (not targets)
    - Can target specific variables or all non-forcing variables

    Dropout is gated on ``self.training``, the standard ``nn.Module`` mode flag,
    so it follows ``.train()`` / ``.eval()`` exactly as ``nn.Dropout`` does: on
    during training, off during validation, off during inference (every runner
    calls ``model.eval()`` before predicting). Natural sparsity is already present
    in the data; this preprocessor adds artificial sparsity on top of it, so
    leaving it off in validation keeps the metric a faithful proxy for the
    deployment condition and comparable across a run whose dropout is scheduled.

    See ``anemoi.training.diagnostics.callbacks.dropout_scheduler.DropoutScheduler``
    for decaying ``dropout_prob`` over training.

    Configuration example:
    ```yaml
    spatial_dropout:
      _target_: anemoi.models.preprocessing.spatial_dropout.RandomSpatialDropout
      config:
        dropout_prob: 0.15  # Drop 15% of valid grid cells
        dropout_variables:  # Optional: specific variables to drop
          - z_500
          - z_850
        multi_step: 2  # Number of input timesteps (optional, defaults to 2)
    ```
    """

    @classmethod
    def _process_config(cls, config) -> tuple:
        """Override to add RandomSpatialDropout-specific special keys.

        Extends the base special keys to exclude preprocessor parameters
        from being treated as data processing strategies.

        Parameters
        ----------
        config : DotDict
            Configuration object of the processor.

        Returns
        -------
        tuple
            (default, remap, normalizer, method_config, method_kwargs) as in the base class.
        """
        _special_keys = [
            "default",
            "remap",
            "normalizer",
            "method_kwargs",
            "dropout_prob",  # Probability of dropping each valid grid cell
            "dropout_variables",  # Optional list of variables to drop
            "multi_step",  # Number of input timesteps to apply dropout to
        ]

        default = config.get("default", "none")
        remap = config.get("remap", {})
        normalizer = config.get("normalizer", "none")
        method_kwargs = config.get("method_kwargs", {})
        method_config = {k: v for k, v in config.items() if k not in _special_keys and v is not None and v != "none"}

        if method_config:
            LOGGER.warning(
                f"{cls.__name__}: Unexpected config keys {list(method_config.keys())}. "
                f"This preprocessor only uses 'dropout_prob', 'dropout_variables', and 'multi_step'."
            )

        return default, remap, normalizer, method_config, method_kwargs

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

        self.dropout_prob = config.get("dropout_prob", 0.0) if config is not None else 0.0

        if not 0.0 <= self.dropout_prob <= 1.0:
            msg = f"dropout_prob must be between 0.0 and 1.0, got {self.dropout_prob}"
            raise ValueError(msg)

        self.multi_step = config.get("multi_step", 2) if config is not None else 2

        self.dropout_variables = config.get("dropout_variables", None) if config is not None else None

        if self.dropout_prob > 0:
            name_to_index = self.data_indices.data.input.name_to_index
            forcing_names = set(getattr(self.data_indices, "forcing", []))

            non_forcing_names = [name for name in name_to_index.keys() if name not in forcing_names]

            if self.dropout_variables is None:
                # Drop from all non-forcing variables (safe default — never
                # touches lat/lon, time encodings, lsm, etc.).
                dropout_names = non_forcing_names
                var_desc = f"all {len(dropout_names)} non-forcing variables"
            else:
                # Explicit list: allow forcings too (e.g. sparse obs forcings
                # used as DA inputs). Caller is responsible for not listing
                # always-present forcings like lat/lon.
                dropout_names = [name for name in self.dropout_variables if name in name_to_index]
                missing = set(self.dropout_variables) - set(dropout_names)
                if missing:
                    LOGGER.warning("RandomSpatialDropout: Variables %s not found in input variables", missing)
                var_desc = f"{len(dropout_names)} specified variables: {dropout_names}"

            dropout_indices = [name_to_index[name] for name in dropout_names]

            self.register_buffer("dropout_indices", torch.tensor(dropout_indices, dtype=torch.long), persistent=False)

            LOGGER.info(
                "RandomSpatialDropout: Will randomly drop %.1f%% of valid grid cells "
                "during training and validation in first %d input timesteps for %s",
                self.dropout_prob * 100,
                self.multi_step,
                var_desc,
            )
        else:
            self.register_buffer("dropout_indices", torch.tensor([], dtype=torch.long), persistent=False)
            LOGGER.info("RandomSpatialDropout: dropout_prob=0.0, no dropout will be applied")

    def transform(self, x: torch.Tensor, in_place: bool = True, **kwargs) -> torch.Tensor:
        """Apply random spatial dropout to input tensor.

        In training mode, randomly sets a percentage of valid (non-NaN) grid cells
        to NaN for specified variables in the input timesteps. In eval mode
        (validation and inference) the tensor is returned unchanged.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, time, ..., grid, variable).
        in_place : bool
            Whether to modify tensor in place (default: True).
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        torch.Tensor
            Tensor with random dropout applied (training) or unchanged (eval).
        """
        # Skip if dropout disabled or no variables to drop
        if self.dropout_prob == 0 or len(self.dropout_indices) == 0:
            return x if in_place else x.clone()

        # Training-only, following the nn.Dropout convention: self.training is set
        # by .train()/.eval(), so this is off in validation and at inference.
        if not self.training or x.ndim < 2:
            return x if in_place else x.clone()

        if not in_place:
            x = x.clone()

        # Vectorized dropout: operate on all target variables and timesteps at once
        n_input = min(self.multi_step, x.shape[1])
        # Slice: (batch, n_input, ..., n_dropout_vars)
        sub = x[:, :n_input, ..., self.dropout_indices]

        valid_mask = ~torch.isnan(sub)
        dropout_mask = torch.rand(sub.shape, device=x.device, dtype=torch.float32) < self.dropout_prob
        cells_to_drop = valid_mask & dropout_mask

        # Write NaN into the selected cells
        sub[cells_to_drop] = torch.nan
        x[:, :n_input, ..., self.dropout_indices] = sub

        return x

    def inverse_transform(self, x: torch.Tensor, in_place: bool = True, **kwargs) -> torch.Tensor:
        """No-op: dropout is not reversible, outputs are unchanged.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        in_place : bool
            Whether to modify tensor in place (default: True).
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        torch.Tensor
            The unchanged tensor.
        """
        return x if in_place else x.clone()

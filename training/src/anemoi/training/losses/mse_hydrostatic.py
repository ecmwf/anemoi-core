# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses.base import FunctionalLoss
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


class HydrostaticLoss(FunctionalLoss):
    """Hydrostatic constraint loss.

    This loss enforces hydrostatic balance between geopotential (z),
    temperature (t), and specific humidity (q) at different pressure levels.

    The hydrostatic equation relates geopotential difference to temperature:
        dΦ = Rd * Tv * ln(p1 / p2)

    where:
        - dΦ is the geopotential difference between pressure levels (m²/s²)
        - Rd is the gas constant for dry air (287.05 J/(kg·K) = 287.05 m²/(s²·K))
        - Tv is virtual temperature = T * (1 + 0.61 * q)
        - p1, p2 are pressure levels (in Pa)

    Note: This assumes z is geopotential in m²/s², not geopotential height in m.

    This loss can be used standalone or combined with other losses via CombinedLoss.

    Example config for CombinedLoss:
    ```yaml
    training_loss:
      _target_: anemoi.training.losses.CombinedLoss
      losses:
        - _target_: anemoi.training.losses.MSELoss
        - _target_: anemoi.training.losses.HydrostaticLoss
          hydrostatic_weight: 0.1
      loss_weights: [1.0, 1.0]
    ```
    """

    name: str = "hydrostatic"

    # Physical constants
    RD: float = 287.05  # Gas constant for dry air (J/(kg·K) = m²/(s²·K))
    EPSILON: float = 0.61  # Ratio of Rd/Rv - 1, for virtual temperature

    def __init__(
        self,
        hydrostatic_weight: float = 1.0,
        z_prefix: str = "z_",
        t_prefix: str = "t_",
        q_prefix: str = "q_",
        ignore_nans: bool = False,
    ) -> None:
        """Initialize the hydrostatic loss.

        Parameters
        ----------
        hydrostatic_weight : float, optional
            Weight for the hydrostatic constraint term, by default 1.0
        z_prefix : str, optional
            Prefix for geopotential height variables, by default "z_"
        t_prefix : str, optional
            Prefix for temperature variables, by default "t_"
        q_prefix : str, optional
            Prefix for specific humidity variables, by default "q_"
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans, by default False
        """
        super().__init__(ignore_nans=ignore_nans)

        self.hydrostatic_weight = hydrostatic_weight
        self.z_prefix = z_prefix
        self.t_prefix = t_prefix
        self.q_prefix = q_prefix

        # Will be populated by set_data_indices
        self.data_indices: IndexCollection | None = None
        self._z_indices: dict[int, int] = {}  # pressure_level -> tensor index
        self._t_indices: dict[int, int] = {}
        self._q_indices: dict[int, int] = {}
        self._pressure_levels: list[int] = []

        # Will be populated by set_denorm_params - for denormalization
        # These match the normalizer's _norm_mul and _norm_add buffers
        self._norm_mul: torch.Tensor | None = None
        self._norm_add: torch.Tensor | None = None

    def set_data_indices(self, data_indices: IndexCollection) -> None:
        """Set the data indices and extract variable mappings.

        Parameters
        ----------
        data_indices : IndexCollection
            Collection of data indices containing variable name to index mappings.
        """
        self.data_indices = data_indices

        # Extract indices for z, t, q at each pressure level
        z_levels = set()
        t_levels = set()
        q_levels = set()

        for key, idx in self.data_indices.model.output.name_to_index.items():
            if key.startswith(self.z_prefix):
                level = int(key[len(self.z_prefix) :])
                self._z_indices[level] = idx
                z_levels.add(level)
            elif key.startswith(self.t_prefix):
                level = int(key[len(self.t_prefix) :])
                self._t_indices[level] = idx
                t_levels.add(level)
            elif key.startswith(self.q_prefix):
                level = int(key[len(self.q_prefix) :])
                self._q_indices[level] = idx
                q_levels.add(level)

        # Find common pressure levels where all three variables are available
        common_levels = z_levels & t_levels & q_levels

        if len(common_levels) < 2:
            LOGGER.warning(
                "HydrostaticLoss requires at least 2 common pressure levels with z, t, and q. "
                "Found z levels: %s, t levels: %s, q levels: %s, common: %s",
                sorted(z_levels),
                sorted(t_levels),
                sorted(q_levels),
                sorted(common_levels),
            )
            self._pressure_levels = []
        else:
            # Sort pressure levels in descending order (high pressure to low, i.e., surface to top)
            self._pressure_levels = sorted(common_levels, reverse=True)
            LOGGER.info(
                "HydrostaticLoss initialized with %d pressure levels: %s",
                len(self._pressure_levels),
                self._pressure_levels,
            )

    def set_denorm_params(self, norm_mul: torch.Tensor, norm_add: torch.Tensor) -> None:
        """Set the denormalization parameters from the normalizer.

        The hydrostatic equations require physical values (T in Kelvin, q in kg/kg,
        z in m²/s²), but the model outputs are normalized. This method stores the
        normalizer's parameters to correctly denormalize predictions before applying
        physical constraints.

        The normalizer uses: x_normalized = x * norm_mul + norm_add
        To denormalize: x = (x_normalized - norm_add) / norm_mul

        This correctly handles all normalization methods (mean-std, std, min-max, max, none).

        Parameters
        ----------
        norm_mul : torch.Tensor
            Multiplicative normalization factor (1/stdev for mean-std, 1/max for max, etc.)
        norm_add : torch.Tensor
            Additive normalization factor (-mean/stdev for mean-std, 0 for std, etc.)
        """
        self._norm_mul = norm_mul.clone()
        self._norm_add = norm_add.clone()

        LOGGER.debug(
            "HydrostaticLoss denormalization parameters set from normalizer (%d variables)",
            len(norm_mul),
        )

    def _denormalize(self, pred: torch.Tensor, indices: list[int]) -> torch.Tensor:
        """Denormalize specific variable indices from predictions.

        Uses the inverse of the normalization transform:
        x = (x_normalized - norm_add) / norm_mul

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (..., n_outputs)
        indices : list[int]
            List of variable indices to extract and denormalize

        Returns
        -------
        torch.Tensor
            Denormalized values for the specified indices, shape (..., len(indices))
        """
        if self._norm_mul is None or self._norm_add is None:
            # No denormalization available, return raw values with warning
            LOGGER.warning(
                "HydrostaticLoss: denormalization parameters not set. "
                "Using raw normalized values which may give incorrect physical constraints.",
            )
            return pred[..., indices]

        # Move denorm params to same device as pred if needed
        if self._norm_mul.device != pred.device:
            self._norm_mul = self._norm_mul.to(pred.device)
            self._norm_add = self._norm_add.to(pred.device)

        # Extract values and get corresponding normalization params
        # For model output, we need to map model indices to data indices
        values = pred[..., indices]

        # Get the normalization parameters for the data output indices
        # The model output indices map to data output indices
        if self.data_indices is not None:
            # Build mapping from model output index to data output index
            data_indices_list = []
            for model_idx in indices:
                # Find the variable name for this model index
                var_name = None
                for name, idx in self.data_indices.model.output.name_to_index.items():
                    if idx == model_idx:
                        var_name = name
                        break
                if var_name is not None and var_name in self.data_indices.data.output.name_to_index:
                    data_idx = self.data_indices.data.output.name_to_index[var_name]
                    data_indices_list.append(data_idx)
                else:
                    # Fallback: assume same index
                    data_indices_list.append(model_idx)

            norm_mul = self._norm_mul[data_indices_list]
            norm_add = self._norm_add[data_indices_list]
        else:
            norm_mul = self._norm_mul[indices]
            norm_add = self._norm_add[indices]

        return (values - norm_add) / norm_mul

    def _compute_hydrostatic_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Compute the hydrostatic constraint loss from predictions.

        The loss is computed as a relative MSE (normalized by the expected hydrostatic
        thickness squared) so that it's dimensionless and on a similar scale to the
        normalized MSE loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)

        Returns
        -------
        torch.Tensor
            Hydrostatic loss (scalar, dimensionless)
        """
        if len(self._pressure_levels) < 2:
            # Not enough levels, return zero loss
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        n_layers = len(self._pressure_levels) - 1

        for k in range(n_layers):
            # Get pressure levels (in hPa, convert to Pa for calculation)
            p_lower = self._pressure_levels[k] * 100.0  # Higher pressure (lower altitude)
            p_upper = self._pressure_levels[k + 1] * 100.0  # Lower pressure (higher altitude)

            # Get variable indices for these levels
            level_lower = self._pressure_levels[k]
            level_upper = self._pressure_levels[k + 1]

            z_lower_idx = self._z_indices[level_lower]
            z_upper_idx = self._z_indices[level_upper]
            t_lower_idx = self._t_indices[level_lower]
            t_upper_idx = self._t_indices[level_upper]
            q_lower_idx = self._q_indices[level_lower]
            q_upper_idx = self._q_indices[level_upper]

            # Extract and denormalize values from prediction tensor
            # The physical equations require real values (T in K, q in kg/kg, z in m²/s²)
            z_lower = self._denormalize(pred, [z_lower_idx]).squeeze(-1)
            z_upper = self._denormalize(pred, [z_upper_idx]).squeeze(-1)
            t_lower = self._denormalize(pred, [t_lower_idx]).squeeze(-1)
            t_upper = self._denormalize(pred, [t_upper_idx]).squeeze(-1)
            q_lower = self._denormalize(pred, [q_lower_idx]).squeeze(-1)
            q_upper = self._denormalize(pred, [q_upper_idx]).squeeze(-1)

            # Predicted thickness from model (now in physical units: m²/s²)
            dz_pred = z_upper - z_lower

            # Compute virtual temperature at each level (T in K, q in kg/kg)
            # Tv = T * (1 + epsilon * q), where epsilon ≈ 0.61
            tv_lower = t_lower * (1.0 + self.EPSILON * q_lower)
            tv_upper = t_upper * (1.0 + self.EPSILON * q_upper)

            # Mean virtual temperature for the layer
            tv_mean = 0.5 * (tv_lower + tv_upper)

            # Geopotential difference implied by hydrostatic balance (also our normalization scale)
            dz_hydro = self.RD * tv_mean * torch.log(torch.tensor(p_lower / p_upper, device=pred.device))

            # Compute relative MSE: ((dz_pred - dz_hydro) / dz_hydro)^2
            # This makes the loss dimensionless and O(1) when predictions are reasonable
            # Use detached dz_hydro for normalization to avoid gradient issues
            scale = dz_hydro.detach().abs().clamp(min=1.0)  # Avoid division by zero
            relative_error = (dz_pred - dz_hydro) / scale
            layer_loss = torch.mean(relative_error**2)
            total_loss = total_loss + layer_loss

        return total_loss / n_layers

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the squared error (MSE) for the standard loss component.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)

        Returns
        -------
        torch.Tensor
            Squared error tensor, same shape as input
        """
        return torch.square(pred - target)

    def forward(
        self,
        pred: torch.Tensor,
        _target: torch.Tensor,
        _squash: bool = True,
        *,
        _scaler_indices: tuple[int, ...] | None = None,
        _without_scalers: list[str] | list[int] | None = None,
        _grid_shard_slice: slice | None = None,
        _group: None = None,
    ) -> torch.Tensor:
        """Compute the hydrostatic loss.

        When used standalone, this returns only the hydrostatic constraint loss.
        When combined with other losses via CombinedLoss, the hydrostatic term
        is added to the combined loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        _target : torch.Tensor
            Target tensor (unused, kept for interface compatibility)
        _squash : bool, optional
            Average last dimension (unused, kept for interface compatibility), by default True
        _scaler_indices : tuple[int,...] | None, optional
            Indices to subset the calculated scaler with (unused), by default None
        _without_scalers : list[str] | list[int] | None, optional
            List of scalers to exclude from scaling (unused), by default None
        _grid_shard_slice : slice | None, optional
            Slice of the grid if x comes sharded (unused), by default None
        _group : None, optional
            Distributed group to reduce over (unused), by default None

        Returns
        -------
        torch.Tensor
            Hydrostatic loss
        """
        # Compute hydrostatic constraint loss
        hydrostatic_loss = self._compute_hydrostatic_loss(pred)

        return self.hydrostatic_weight * hydrostatic_loss


class MSELossWithHydrostatic(FunctionalLoss):
    """MSE loss with optional hydrostatic constraint term.

    This combines the standard MSE loss with a hydrostatic balance constraint
    as an additional penalty term.

    The total loss is:
        L = MSE(pred, target) + hydrostatic_weight * L_hydrostatic

    where L_hydrostatic enforces the hydrostatic equation between pressure levels.

    Note: This assumes z is geopotential in m²/s², not geopotential height in m.
    """

    name: str = "mse_hydrostatic"

    # Physical constants
    RD: float = 287.05  # Gas constant for dry air (J/(kg·K) = m²/(s²·K))
    EPSILON: float = 0.61  # Ratio of Rd/Rv - 1

    def __init__(
        self,
        hydrostatic_weight: float = 1.0,  # 0.1,
        z_prefix: str = "z_",
        t_prefix: str = "t_",
        q_prefix: str = "q_",
        ignore_nans: bool = False,
    ) -> None:
        """Initialize the MSE + hydrostatic loss.

        Parameters
        ----------
        hydrostatic_weight : float, optional
            Weight for the hydrostatic constraint term, by default 0.1
        z_prefix : str, optional
            Prefix for geopotential height variables, by default "z_"
        t_prefix : str, optional
            Prefix for temperature variables, by default "t_"
        q_prefix : str, optional
            Prefix for specific humidity variables, by default "q_"
        ignore_nans : bool, optional
            Allow nans in the loss, by default False
        """
        super().__init__(ignore_nans=ignore_nans)

        self.hydrostatic_weight = hydrostatic_weight
        self.z_prefix = z_prefix
        self.t_prefix = t_prefix
        self.q_prefix = q_prefix

        # Will be populated by set_data_indices
        self.data_indices: IndexCollection | None = None
        self._z_indices: dict[int, int] = {}
        self._t_indices: dict[int, int] = {}
        self._q_indices: dict[int, int] = {}
        self._pressure_levels: list[int] = []

        # Will be populated by set_denorm_params - for denormalization
        # These match the normalizer's _norm_mul and _norm_add buffers
        self._norm_mul: torch.Tensor | None = None
        self._norm_add: torch.Tensor | None = None

    def set_data_indices(self, data_indices: IndexCollection) -> None:
        """Set the data indices and extract variable mappings.

        Parameters
        ----------
        data_indices : IndexCollection
            Collection of data indices containing variable name to index mappings.
        """
        self.data_indices = data_indices

        # Extract indices for z, t, q at each pressure level
        z_levels = set()
        t_levels = set()
        q_levels = set()

        for key, idx in self.data_indices.model.output.name_to_index.items():
            if key.startswith(self.z_prefix):
                level = int(key[len(self.z_prefix) :])
                self._z_indices[level] = idx
                z_levels.add(level)
            elif key.startswith(self.t_prefix):
                level = int(key[len(self.t_prefix) :])
                self._t_indices[level] = idx
                t_levels.add(level)
            elif key.startswith(self.q_prefix):
                level = int(key[len(self.q_prefix) :])
                self._q_indices[level] = idx
                q_levels.add(level)

        # Find common pressure levels
        common_levels = z_levels & t_levels & q_levels

        if len(common_levels) < 2:
            LOGGER.warning(
                "MSELossWithHydrostatic requires at least 2 common pressure levels with z, t, and q. "
                "Found z levels: %s, t levels: %s, q levels: %s, common: %s. "
                "Hydrostatic term will be disabled.",
                sorted(z_levels),
                sorted(t_levels),
                sorted(q_levels),
                sorted(common_levels),
            )
            self._pressure_levels = []
        else:
            self._pressure_levels = sorted(common_levels, reverse=True)
            LOGGER.info(
                "MSELossWithHydrostatic initialized with %d pressure levels: %s",
                len(self._pressure_levels),
                self._pressure_levels,
            )

    def set_denorm_params(self, norm_mul: torch.Tensor, norm_add: torch.Tensor) -> None:
        """Set the denormalization parameters from the normalizer.

        The hydrostatic equations require physical values (T in Kelvin, q in kg/kg,
        z in m²/s²), but the model outputs are normalized. This method stores the
        normalizer's parameters to correctly denormalize predictions before applying
        physical constraints.

        The normalizer uses: x_normalized = x * norm_mul + norm_add
        To denormalize: x = (x_normalized - norm_add) / norm_mul

        This correctly handles all normalization methods (mean-std, std, min-max, max, none).

        Parameters
        ----------
        norm_mul : torch.Tensor
            Multiplicative normalization factor (1/stdev for mean-std, 1/max for max, etc.)
        norm_add : torch.Tensor
            Additive normalization factor (-mean/stdev for mean-std, 0 for std, etc.)
        """
        self._norm_mul = norm_mul.clone()
        self._norm_add = norm_add.clone()

        LOGGER.debug(
            "MSELossWithHydrostatic denormalization parameters set from normalizer (%d variables)",
            len(norm_mul),
        )

    def _denormalize(self, pred: torch.Tensor, indices: list[int]) -> torch.Tensor:
        """Denormalize specific variable indices from predictions.

        Uses the inverse of the normalization transform:
        x = (x_normalized - norm_add) / norm_mul

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (..., n_outputs)
        indices : list[int]
            List of variable indices to extract and denormalize

        Returns
        -------
        torch.Tensor
            Denormalized values for the specified indices, shape (..., len(indices))
        """
        if self._norm_mul is None or self._norm_add is None:
            # No denormalization available, return raw values with warning
            LOGGER.warning(
                "MSELossWithHydrostatic: denormalization parameters not set. "
                "Using raw normalized values which may give incorrect physical constraints.",
            )
            return pred[..., indices]

        # Move denorm params to same device as pred if needed
        if self._norm_mul.device != pred.device:
            self._norm_mul = self._norm_mul.to(pred.device)
            self._norm_add = self._norm_add.to(pred.device)

        # Extract values and get corresponding normalization params
        # For model output, we need to map model indices to data indices
        values = pred[..., indices]

        # Get the normalization parameters for the data output indices
        # The model output indices map to data output indices
        if self.data_indices is not None:
            # Build mapping from model output index to data output index
            data_indices_list = []
            for model_idx in indices:
                # Find the variable name for this model index
                var_name = None
                for name, idx in self.data_indices.model.output.name_to_index.items():
                    if idx == model_idx:
                        var_name = name
                        break
                if var_name is not None and var_name in self.data_indices.data.output.name_to_index:
                    data_idx = self.data_indices.data.output.name_to_index[var_name]
                    data_indices_list.append(data_idx)
                else:
                    # Fallback: assume same index
                    data_indices_list.append(model_idx)

            norm_mul = self._norm_mul[data_indices_list]
            norm_add = self._norm_add[data_indices_list]
        else:
            norm_mul = self._norm_mul[indices]
            norm_add = self._norm_add[indices]

        return (values - norm_add) / norm_mul

    def _compute_hydrostatic_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Compute the hydrostatic constraint loss.

        The loss is computed as a relative MSE (normalized by the expected hydrostatic
        thickness squared) so that it's dimensionless and on a similar scale to the
        normalized MSE loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)

        Returns
        -------
        torch.Tensor
            Hydrostatic loss (scalar, dimensionless)
        """
        if len(self._pressure_levels) < 2:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        n_layers = len(self._pressure_levels) - 1

        for k in range(n_layers):
            p_lower = self._pressure_levels[k] * 100.0  # hPa to Pa
            p_upper = self._pressure_levels[k + 1] * 100.0

            level_lower = self._pressure_levels[k]
            level_upper = self._pressure_levels[k + 1]

            # Extract and denormalize values from prediction tensor
            # The physical equations require real values (T in K, q in kg/kg, z in m²/s²)
            z_lower = self._denormalize(pred, [self._z_indices[level_lower]]).squeeze(-1)
            z_upper = self._denormalize(pred, [self._z_indices[level_upper]]).squeeze(-1)
            t_lower = self._denormalize(pred, [self._t_indices[level_lower]]).squeeze(-1)
            t_upper = self._denormalize(pred, [self._t_indices[level_upper]]).squeeze(-1)
            q_lower = self._denormalize(pred, [self._q_indices[level_lower]]).squeeze(-1)
            q_upper = self._denormalize(pred, [self._q_indices[level_upper]]).squeeze(-1)

            # Predicted thickness (now in physical units: m²/s²)
            dz_pred = z_upper - z_lower

            # Virtual temperature (T in K, q in kg/kg)
            tv_lower = t_lower * (1.0 + self.EPSILON * q_lower)
            tv_upper = t_upper * (1.0 + self.EPSILON * q_upper)
            tv_mean = 0.5 * (tv_lower + tv_upper)

            # Hydrostatic geopotential difference (this is also our normalization scale)
            dz_hydro = self.RD * tv_mean * torch.log(torch.tensor(p_lower / p_upper, device=pred.device))

            # Compute relative MSE: ((dz_pred - dz_hydro) / dz_hydro)^2
            # This makes the loss dimensionless and O(1) when predictions are reasonable
            # Use detached dz_hydro for normalization to avoid gradient issues
            scale = dz_hydro.detach().abs().clamp(min=1.0)  # Avoid division by zero
            relative_error = (dz_pred - dz_hydro) / scale
            layer_loss = torch.mean(relative_error**2)
            total_loss = total_loss + layer_loss

        return total_loss / n_layers

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the squared error for MSE.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor
        target : torch.Tensor
            Target tensor

        Returns
        -------
        torch.Tensor
            Squared error tensor
        """
        return torch.square(pred - target)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: None = None,
    ) -> torch.Tensor:
        """Compute MSE + hydrostatic loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)
        squash : bool, optional
            Average last dimension, by default True
        scaler_indices : tuple[int,...], optional
            Indices to subset the calculated scaler with, by default None
        without_scalers : list[str] | list[int] | None, optional
            List of scalers to exclude from scaling, by default None
        grid_shard_slice : slice, optional
            Slice of the grid if x comes sharded, by default None
        group : ProcessGroup, optional
            Distributed group to reduce over, by default None

        Returns
        -------
        torch.Tensor
            Combined MSE + hydrostatic loss
        """
        # Standard MSE loss via parent class
        is_sharded = grid_shard_slice is not None

        # Add hydrostatic constraint
        hydrostatic_loss = self._compute_hydrostatic_loss(pred)

        # Density-weighted NaN handling (similar to NaNAwareMSELoss)
        density_weights = None
        if self.ignore_nans:
            # Identify NaN positions in target
            nan_mask = torch.isnan(target)
            _, _, latlon, _ = target.shape

            # Count NaNs per variable (along the grid dimension)
            nan_per_var = nan_mask.sum(dim=TensorDim.GRID, keepdim=True)

            # Compute density weights: compensate for missing values so sparse variables have equal impact
            # If a variable has 50% NaNs, weight = 2.0; if 0% NaNs, weight = 1.0
            # Clamp denominator to avoid division by zero if a variable is entirely NaN
            valid_points = (latlon - nan_per_var).clamp(min=1)
            density_weights = latlon / valid_points

            # Mask out NaN positions by setting both pred and target to 0 there
            target = target.masked_fill(nan_mask, 0.0)
            pred = pred.masked_fill(nan_mask, 0.0)

        # Calculate squared error
        out = self.calculate_difference(pred, target)

        # Apply density weights to compensate for sparse variables
        if density_weights is not None:
            out = out * density_weights

        out = self.scale(out, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)
        mse_loss = self.reduce(out, squash, group=group if is_sharded else None)

        return mse_loss + self.hydrostatic_weight * hydrostatic_loss

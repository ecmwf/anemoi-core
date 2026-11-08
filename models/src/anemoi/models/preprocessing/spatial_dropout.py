# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Random dropout of input observations for data augmentation."""

import logging
from typing import Optional

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import BasePreprocessor

LOGGER = logging.getLogger(__name__)

# A gridded view is (batch, time, ensemble, grid, variables); a tabular view is processed one
# sample at a time as (ensemble, rows, variables), its time windows stacked on the rows.
GRIDDED_NDIM = 5
TABULAR_NDIM = 3


class BaseInputDropout(BasePreprocessor):
    """Base class of the preprocessors that set input values to NaN during training.

    The dropout runs before the imputer, which fills the dropped values. It only acts on model
    inputs: target views are pre-processed with ``skip_imputation=True`` (they keep their missing
    values for the loss), and are returned unchanged. ``apply_dropout=True`` overrides that, for
    observations that feed the model but must keep their missing values (the DA-cycle observations
    a DA forecaster blends into its state). Dropout follows ``self.training`` like ``nn.Dropout``,
    so it is off in validation and inference.

    ``dropout_prob`` and ``dropout_indices`` are what
    :class:`~anemoi.training.diagnostics.callbacks.dropout_scheduler.DropoutScheduler` drives.
    ``multi_step`` limits the dropout to the first time steps of a gridded view (``None``: all).
    """

    supports_skip_imputation = True

    # Configuration keys that are parameters, not processing methods.
    _parameter_keys: tuple[str, ...] = ()

    @classmethod
    def _process_config(cls, config) -> tuple:
        config = config or {}
        special_keys = {"default", "remap", "normalizer", "method_kwargs", *cls._parameter_keys}
        method_config = {k: v for k, v in config.items() if k not in special_keys and v is not None and v != "none"}
        if method_config:
            LOGGER.warning("%s: Unexpected config keys %s.", cls.__name__, list(method_config))
        return (
            config.get("default", "none"),
            config.get("remap", {}),
            config.get("normalizer", "none"),
            {},
            config.get("method_kwargs", {}),
        )

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        super().__init__(config, data_indices, statistics)
        config = config or {}
        multi_step = config.get("multi_step", None)
        self.multi_step = None if multi_step is None else int(multi_step)
        self._positions_cache: dict[tuple, dict[str, list[int]]] = {}

    def _positions(self, names_by_key: dict[str, list[str]], name_to_index: dict[str, int]) -> dict[str, list[int]]:
        """Positions of each key's variables in a view, skipping the variables it does not carry."""
        cache_key = tuple(name_to_index.items())
        positions = self._positions_cache.get(cache_key)
        if positions is None:
            positions = {
                key: [name_to_index[name] for name in names if name in name_to_index]
                for key, names in names_by_key.items()
            }
            self._positions_cache[cache_key] = positions
        return positions

    @staticmethod
    def _as_gridded(x: torch.Tensor) -> torch.Tensor:
        """View a tabular sample as a one-sample, one-step gridded tensor, sharing its storage."""
        if x.ndim == GRIDDED_NDIM:
            return x
        if x.ndim == TABULAR_NDIM:
            return x[None, None]
        msg = f"Input dropout expects a {GRIDDED_NDIM}-D gridded or {TABULAR_NDIM}-D tabular tensor, got {x.ndim}-D."
        raise ValueError(msg)

    def _active(self, x: torch.Tensor, skip_imputation: bool, apply_dropout: bool | None) -> bool:
        apply = not skip_imputation if apply_dropout is None else apply_dropout
        return self.training and apply and self.dropout_prob > 0 and x.numel() > 0

    def _n_input(self, x: torch.Tensor) -> int:
        return x.shape[1] if self.multi_step is None else min(self.multi_step, x.shape[1])

    def reset_masks(self) -> None:
        """Start a new training batch. A no-op unless the dropout reuses its masks across views."""

    def inverse_transform(self, x: torch.Tensor, **_kwargs) -> torch.Tensor:
        """No-op: dropout is not reversible."""
        return x


class RandomSpatialDropout(BaseInputDropout):
    """Randomly drops (sets to NaN) a fraction of the valid input values during training.

    Each value (time step, member, cell, variable) is drawn independently. Only originally valid
    values are dropped, so the model learns to interpolate from artificially sparser inputs.

    Configuration example:
    ```yaml
    spatial_dropout:
      _target_: anemoi.models.preprocessing.spatial_dropout.RandomSpatialDropout
      config:
        dropout_prob: 0.15     # drop 15 % of the valid values
        dropout_variables:     # optional; default: all non-forcing variables
          - z_500
        multi_step: 2          # optional; default: every time step of the view
    ```
    """

    _parameter_keys = ("dropout_prob", "dropout_variables", "multi_step")

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
            ``dropout_prob`` (in [0, 1]), optional ``dropout_variables`` and ``multi_step``.
        data_indices : IndexCollection
            Data indices of the dataset.
        statistics : dict
            Not used by this preprocessor.
        """
        super().__init__(config, data_indices, statistics)
        config = config or {}
        self.dropout_prob = float(config.get("dropout_prob", 0.0))
        if not 0.0 <= self.dropout_prob <= 1.0:
            msg = f"dropout_prob must be between 0.0 and 1.0, got {self.dropout_prob}"
            raise ValueError(msg)

        name_to_index = self.data_indices.data.input.name_to_index
        variables = config.get("dropout_variables", None)
        if variables is None:
            # Every non-forcing variable: never touches lat/lon, time encodings, lsm, etc.
            forcing = set(self.data_indices.forcing)
            names = [name for name in name_to_index if name not in forcing]
        else:
            # An explicit list may include forcings (e.g. sparse observations used as DA inputs).
            names = [name for name in variables if name in name_to_index]
            missing = sorted(set(variables) - set(names))
            if missing:
                LOGGER.warning("RandomSpatialDropout: variables %s are not input variables.", missing)
        self.dropout_names = names
        self.register_buffer(
            "dropout_indices",
            torch.tensor([name_to_index[name] for name in names], dtype=torch.long),
            persistent=False,
        )
        LOGGER.info(
            "RandomSpatialDropout: dropping %.1f%% of the valid values of %d variables during training.",
            self.dropout_prob * 100,
            len(names),
        )

    def transform(
        self,
        x: torch.Tensor,
        name_to_index: Optional[dict[str, int]] = None,
        skip_imputation: bool = False,
        apply_dropout: Optional[bool] = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Drop random valid input values.

        Parameters
        ----------
        x : torch.Tensor
            A gridded view ``(batch, time, ensemble, grid, variables)`` or one tabular sample
            ``(ensemble, rows, variables)``. Modified in place.
        name_to_index : dict[str, int], optional
            Variable positions in ``x``.
        skip_imputation : bool, optional
            Set on target views, which are returned unchanged.
        apply_dropout : bool, optional
            Overrides whether the view gets dropout; by default only views that are imputed do.

        Returns
        -------
        torch.Tensor
            ``x`` with dropped values set to NaN (training), or unchanged.
        """
        if not self._active(x, skip_imputation, apply_dropout):
            return x
        positions = self._positions({"dropout": self.dropout_names}, name_to_index)["dropout"]
        if not positions:
            return x
        view = self._as_gridded(x)
        n_input = self._n_input(view)
        idx = torch.tensor(positions, dtype=torch.long, device=x.device)
        sub = view[:, :n_input, ..., idx]
        drop = torch.rand(sub.shape, device=x.device) < self.dropout_prob
        view[:, :n_input, ..., idx] = sub.masked_fill(drop, torch.nan)
        return x

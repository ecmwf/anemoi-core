# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import torch

from anemoi.training.losses.scalers.base_scaler import BaseScaler
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


class SpectralDimensionScaler(BaseScaler):
    """Base class for scaling over the spectral dimension.

    When spectral losses are used the grid dimension is mapped to a spectral
    dimension via a spectral transform.  For SHT-based transforms the output
    has shape ``(L, M)`` (total wavenumber, order) which is then flattened to a
    single mode axis of length ``L * M``.  Since ``L == M == n_spectral_modes``
    for SHTs, the total flat size is ``n_spectral_modes ** 2``.

    Subclasses define how the weight varies as a function of total wavenumber.
    The total wavenumber for flat index *i* is ``i // n_spectral_modes``.

    The scaler produces a 1-D tensor of length ``n_spectral_modes ** 2``
    and registers it on ``TensorDim.GRID`` so that the standard scaler pipeline
    applies it along the correct axis (position 3 in the loss tensor, which
    holds spectral modes instead of grid points when a spectral transform has
    been applied).

    The default implementation scales uniformly by ``1 / n_spectral_modes``.
    """

    scale_dims: TensorDim = TensorDim.GRID

    def __init__(
        self,
        n_spectral_modes: int,
        norm: str | None = None,
        **kwargs,
    ) -> None:
        """Initialise SpectralDimensionScaler to scale by 1/n_spectral_modes in the spectral dimension.

        Parameters
        ----------
        n_spectral_modes : int
            Number of total wavenumbers (L dimension).
        norm : str, optional
            Type of normalization to apply.
            Options are None, unit-sum, unit-mean and l1.
        **kwargs : dict
            Additional keyword arguments (ignored).
        """
        super().__init__(norm=norm)
        del kwargs
        self.n_spectral_modes = n_spectral_modes
        self.n_spectral = self.n_spectral_modes**2

    def get_scaling_values(self, **_kwargs) -> torch.Tensor:
        """Return uniform scaling values (i.e. ones).

        Returns
        -------
        torch.Tensor
            Scaling values as a torch tensor.
        """
        LOGGER.info(
            "Spectral Scaling: Applying %s with n_spectral_modes=%d.",
            self.__class__.__name__,
            self.n_spectral_modes,
        )

        return torch.ones(self.n_spectral, dtype=torch.float32) / self.n_spectral_modes


class LinearSpectralDimensionScaler(SpectralDimensionScaler):
    """Linearly increasing weights with total wavenumber.

    For SHT output of shape ``(L, M)`` flattened to ``L * M``, with
    ``L == M == n_spectral_modes``, the weight at flat index *i* is based on
    the total wavenumber ``L = i // n_spectral_modes``:

        weight[i] = slope * (i // n_spectral_modes) + y_intercept

    This means all orders within the same total wavenumber receive the same
    weight.  With the default parameters (``slope=1/n_spectral_modes``,
    ``y_intercept=1/n_spectral_modes``), higher wavenumbers receive higher
    weights.
    """

    def __init__(
        self,
        n_spectral_modes: int,
        slope: float | None = None,
        y_intercept: float | None = None,
        norm: str | None = None,
        **kwargs,
    ) -> None:
        """Initialise LinearSpectralDimensionScaler.

        Parameters
        ----------
        n_spectral_modes : int
            Number of total wavenumbers (L dimension).
        slope : float
            Slope of the linear function.  Positive values give higher weights
            to higher wavenumbers.
        y_intercept : float
            Constant offset (value at wavenumber 0).
        norm : str, optional
            Type of normalization to apply.
        **kwargs : dict
            Additional keyword arguments (ignored).
        """
        super().__init__(n_spectral_modes=n_spectral_modes, norm=norm, **kwargs)
        self.slope = slope if slope is not None else 1.0 / self.n_spectral_modes
        self.y_intercept = y_intercept if y_intercept is not None else 1.0 / self.n_spectral_modes

    def get_scaling_values(self) -> torch.Tensor:
        flat_indices = torch.arange(self.n_spectral, dtype=torch.float32)
        wavenumbers = flat_indices // self.n_spectral_modes

        LOGGER.info(
            "Spectral Scaling: Applying %s with n_spectral_modes=%d, slope=%s, y_intercept=%s.",
            self.__class__.__name__,
            self.n_spectral_modes,
            self.slope,
            self.y_intercept,
        )

        return self.slope * wavenumbers + self.y_intercept

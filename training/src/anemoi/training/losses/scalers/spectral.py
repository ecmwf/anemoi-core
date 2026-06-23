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
    representation via a spectral transform.  The output of the transforms has 
    - shape ``(L, M)`` (total wavenumber, zonal wavenumber) for SHT.
    - shape  ``(x_dim, y_dim)`` for 2D spectral transforms.
    Different losses have different output shape: some losses use the flattened
    output (L*M) and others collapse over one dimension.

    n_spectral refers to the length of the spectral dimension, 
    and n_spectral_modes the number of total wavenumbers/frequencies.
    The default implementation scales uniformly by ``1 / n_spectral_modes``.
    """

    scale_dims: TensorDim = TensorDim.GRID

    def __init__(
        self,
        n_spectral_modes: int,
        n_spectral: int | None = None,
        norm: str | None = None,
        **kwargs,
    ) -> None:
        """Initialise SpectralDimensionScaler to scale by 1/n_spectral_modes in the spectral dimension.

        Parameters
        ----------
        n_spectral_modes : int
            Number of total wavenumbers (L dimension) for SHT and total frequencies for 2D spectral transforms.
        n_spectral : int, optional
            Total number of spectral modes (length of the spectral dimension). Default to n_spectral_modes
        norm : str, optional
            Type of normalization to apply.
            Options are None, unit-sum, unit-mean and l1.
        **kwargs : dict
            Additional keyword arguments (ignored).
        """
        super().__init__(norm=norm)
        del kwargs
        self.n_spectral_modes = n_spectral_modes
        self.n_spectral = n_spectral if n_spectral is not None else self.n_spectral_modes

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

    For spectral transform output of shape ``(L, M)`` reduced to ``L``, with
    ``L = n_spectral_modes``, the weight at for total wavenumber *i* is:

        weight[i] = slope * i + y_intercept

    With the default parameters (``slope=1/n_spectral_modes``,
    ``y_intercept=1/n_spectral_modes``), higher wavenumbers receive higher
    weights.
    """

    def __init__(
        self,
        n_spectral_modes: int,
        n_spectral: int | None = None,
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
        n_spectral : int, optional
            Total number of spectral modes (length of the (flattened) spectral dimension). Default to n_spectral_modes
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
        super().__init__(n_spectral_modes=n_spectral_modes, n_spectral=n_spectral, norm=norm, **kwargs)
        self.slope = slope if slope is not None else 1.0 / self.n_spectral_modes
        self.y_intercept = y_intercept if y_intercept is not None else 1.0 / self.n_spectral_modes

    def get_scaling_values(self) -> torch.Tensor:

        if self.n_spectral_modes == self.n_spectral:
            # if n_spectral_modes == n_spectral, then the scaling values are simply a linear function of the frequency
            # the frequency is increasing with index
            LOGGER.info(
                "Spectral Scaling: Applying %s with n_spectral_modes=%d, slope=%s, y_intercept=%s.",
                self.__class__.__name__,
                self.n_spectral_modes,
                self.slope,
                self.y_intercept,
            )
            return self.slope * torch.arange(self.n_spectral, dtype=torch.float32) + self.y_intercept

        # if n_spectral_modes < n_spectral, then the scaling values are a linear function of frequency
        # the frequency is increasing with index, and the scaling values are repeated
        flat_indices = torch.arange(self.n_spectral, dtype=torch.float32)
        wavenumbers = flat_indices // self.n_spectral_modes

        LOGGER.info(
            "Spectral Scaling: Applying %s with n_spectral_modes=%d, loss tensor dimensions n_spectral=%d, slope=%s, y_intercept=%s.",
            self.__class__.__name__,
            self.n_spectral_modes,
            self.n_spectral,
            self.slope,
            self.y_intercept,
        )

        return self.slope * wavenumbers + self.y_intercept

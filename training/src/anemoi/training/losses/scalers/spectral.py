# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import torch

from anemoi.training.losses.scaler_tensor import ScalerDomain
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
    grid_domain = ScalerDomain.SPECTRAL

    def __init__(
        self,
        n_spectral_modes: int,
        spectral_dims: int | None = None,
        norm: str | None = None,
        **kwargs,
    ) -> None:
        """Initialise SpectralDimensionScaler to scale by 1/n_spectral_modes in the spectral dimension.

        Parameters
        ----------
        n_spectral_modes : int
            Number of total wavenumbers (L dimension) for SHT and total frequencies for 2D spectral transforms.
        spectral_dims : int, optional
            Length of the spectral dimension. Defaults to n_spectral_modes.
            In case of flattened spectral representation, the spectral dimension is higher than n_spectral_modes.
        norm : str, optional
            Type of normalization to apply.
            Options are None, unit-sum, unit-mean and l1.
        **kwargs : dict
            Additional keyword arguments (ignored).
        """
        super().__init__(norm=norm)
        del kwargs
        self.n_spectral_modes = n_spectral_modes
        self.spectral_dims = spectral_dims if spectral_dims is not None else self.n_spectral_modes

    def get_scaling_values(self, **_kwargs) -> torch.Tensor:
        """Return uniform scaling values (all entries equal to ``1 / n_spectral_modes``).

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

        return torch.ones(self.spectral_dims, dtype=torch.float32) / self.n_spectral_modes

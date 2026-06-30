# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import torch

from anemoi.training.losses.kcrps import AlmostFairKernelCRPS
from anemoi.training.losses.crps_fft_lpf import CRPSFFTLossLPF


class AlmostFairKernelCRPSLooped(AlmostFairKernelCRPS):
    """Memory-efficient AlmostFairKernelCRPS using loop-based ensemble variance.

    Avoids materializing the full (ens, ens) pairwise matrix in _kernel_crps.
    Mathematically equivalent to AlmostFairKernelCRPS.
    """

    def _kernel_crps(self, preds: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Kernel (ensemble) CRPS — loop version to reduce memory usage.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted ensemble, shape (batch_size, n_vars, latlon, ens_size)
        targets : torch.Tensor
            Ground truth, shape (batch_size, n_vars, latlon)
        alpha : float
            Factor for linear combination of fair and standard CRPS

        Returns
        -------
        kCRPS : torch.Tensor
            The point-wise kernel CRPS, shape (batch_size, n_vars, latlon).
        """
        ens_size = preds.shape[-1]
        assert ens_size > 1, "Ensemble size must be greater than 1."

        epsilon = (1.0 - alpha) / ens_size

        out_dtype = preds.real.dtype if preds.is_complex() else preds.dtype

        # MAE component: loop over ensemble to avoid (bs, vars, latlon, ens) broadcast
        mae = torch.zeros(preds.shape[:-1], device=preds.device, dtype=out_dtype)
        for i in range(ens_size):
            mae += torch.abs(preds[..., i] - targets)
        mae /= ens_size

        # Ensemble variance component: loop over pairs to avoid O(ens^2) memory
        ens_var = torch.zeros(preds.shape[:-1], device=preds.device, dtype=out_dtype)
        for i in range(ens_size):
            for j in range(i + 1, ens_size):
                ens_var += torch.abs(preds[..., i] - preds[..., j])

        coef = -(1.0 - epsilon) / (ens_size * (ens_size - 1))
        return mae + coef * ens_var

    @property
    def name(self) -> str:
        return f"afkcrps_loop{self.alpha:.2f}"

class CRPSFFTLossLoopedLPF(CRPSFFTLossLPF):
    """Memory-efficient CRPSFFTLossLPF using loop-based ensemble variance.

    Inherits all FFT logic from CRPSFFTLossLPF but overrides _kernel_crps
    with the memory-efficient loop version.
    """

    _kernel_crps = AlmostFairKernelCRPSLooped._kernel_crps

    @property
    def name(self) -> str:
        return f"CRPS-FFT-LPF-loop{self.alpha:.2f}"
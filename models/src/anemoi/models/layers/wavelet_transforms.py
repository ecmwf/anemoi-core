# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Literal
from typing import NamedTuple
from typing import Protocol
from typing import runtime_checkable

import torch
from torch import nn


class WaveletPyramid(NamedTuple):
    """Low-pass ``[N,C,H,W]`` and high-pass ``[N,C,6,Hj,Wj,2]`` coefficients."""

    lowpass: torch.Tensor
    highpasses: tuple[torch.Tensor, ...]


@runtime_checkable
class _BackendResult(Protocol):
    yl: torch.Tensor
    yh: list[torch.Tensor] | tuple[torch.Tensor, ...]


class DTCWT2D(nn.Module):
    """DTCWT with symmetric boundaries and a common coefficient contract.

    Parameters
    ----------
    J
        Positive number of decomposition levels.
    backend
        The supported ``pytorch_wavelets`` implementation.
    biort, qshift
        First-level and subsequent-level filters.

    Notes
    -----
    The optional ``pytorch_wavelets`` dependency is imported when the transform
    is constructed. Inputs must be float32 or float64 tensors, and outputs retain
    the input dtype.
    """

    def __init__(
        self,
        J: int,
        backend: Literal["pytorch_wavelets"] = "pytorch_wavelets",
        biort: str = "near_sym_b",
        qshift: str = "qshift_b",
    ) -> None:
        super().__init__()
        if isinstance(J, bool) or not isinstance(J, int) or J < 1:
            raise ValueError("J must be a positive integer")
        if backend != "pytorch_wavelets":
            raise ValueError(f"Unknown wavelet backend: {backend!r}")
        self.J = J
        self.backend = backend
        try:
            from pytorch_wavelets import DTCWTForward
        except ImportError as exc:
            raise ImportError("Install anemoi-models[wavelet] to use the pytorch_wavelets backend") from exc
        self.impl = DTCWTForward(J=J, biort=biort, qshift=qshift, mode="symmetric")

    def prepare(
        self,
        device: torch.device,
        grid_shape: tuple[int, int],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Materialize device-dependent resources eagerly, before compiled work."""
        self.to(device=device, dtype=dtype)
        with torch.no_grad():
            self(torch.zeros(1, 1, *grid_shape, device=device, dtype=dtype))

    def forward(self, data: torch.Tensor) -> WaveletPyramid:
        """Transform ``[N,C,H,W]`` fields, returning the full coefficient pyramid."""
        if data.ndim != 4 or any(size < 1 for size in data.shape):
            raise ValueError("DTCWT2D expects a nonempty [N,C,H,W] tensor")
        if data.dtype not in (torch.float32, torch.float64):
            raise TypeError("DTCWT2D inputs must be float32 or float64")
        # This boundary stays eager because wavelet filters may be initialized lazily.
        self.impl.to(device=data.device, dtype=data.dtype)
        output = self.impl(data)
        if isinstance(output, tuple) and len(output) == 2:
            lowpass, highpasses = output
        elif isinstance(output, _BackendResult):
            lowpass, highpasses = output.yl, output.yh
        else:
            raise TypeError("pytorch_wavelets DTCWT must return (yl, yh) or an object with yl/yh attributes")
        if not isinstance(lowpass, torch.Tensor) or lowpass.ndim != 4 or lowpass.shape[:2] != data.shape[:2]:
            raise ValueError("DTCWT low-pass coefficients must have shape [N,C,H,W]")
        if not isinstance(highpasses, (list, tuple)) or len(highpasses) != self.J:
            raise ValueError(f"DTCWT backend must return exactly J={self.J} high-pass levels")
        for level in highpasses:
            if (
                not isinstance(level, torch.Tensor)
                or level.ndim != 6
                or level.shape[:3] != (*data.shape[:2], 6)
                or level.shape[-1] != 2
                or min(level.shape[-3:-1]) < 1
            ):
                raise ValueError("DTCWT high-pass coefficients must have shape [N,C,6,Hj,Wj,2]")
        return WaveletPyramid(
            lowpass.to(dtype=data.dtype).contiguous(),
            tuple(level.to(dtype=data.dtype).contiguous() for level in highpasses),
        )

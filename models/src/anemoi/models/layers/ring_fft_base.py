# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Ring layout, input validation, dtype promotion and empty batches, shared by the FFT backends."""

from numbers import Integral

import torch
from torch import Tensor


def at_least_float32(x: Tensor) -> Tensor:
    """Promote to at least float32, or complex64 for complex tensors."""
    lowest = torch.complex64 if x.is_complex() else torch.float32
    return x.to(torch.promote_types(x.dtype, lowest))


def empty_transform(x: Tensor, shape: tuple[int, ...], dtype: torch.dtype) -> Tensor:
    """Return the transform of ``x`` when ``x`` holds no fields, as an empty tensor of the given shape and dtype.

    The FFT libraries reject a batch without fields, which a rank of a sharded grid gets when it holds no variables.
    The ring FFTs and the Cartesian transforms return this tensor for such a batch, without calling the libraries.
    The result must still depend on the input, since the backward pass exchanges gradients between all ranks and
    every rank has to take part. Adding the sum of the empty input, which is zero, keeps the result in the autograd
    graph.
    """
    return x.new_zeros(shape, dtype=dtype) + x.real.sum()


class RingFFTBase:
    """Fourier transforms along the latitude rings of a reduced grid.

    As for spherical harmonics, the coefficients are indexed by the order m, the number of waves around a ring.
    A ring of n points holds the orders 0 to n // 2.

    The two public transforms validate their input and promote half precision to float32, or to complex64 for
    coefficients, while float32 and float64 keep their precision. Each backend supplies ``_rfft`` and ``_irfft``.

    Parameters
    ----------
    lengths : tuple[int, ...]
        Number of points on each ring, in the order the rings are stored.
    """

    def __init__(self, lengths: tuple[int, ...]) -> None:
        if not lengths or any(not isinstance(n, Integral) or n < 1 for n in lengths):
            raise ValueError("Ring lengths must be positive integers")
        self.lengths = tuple(int(n) for n in lengths)
        self.nlat = len(self.lengths)
        self.points = sum(self.lengths)
        self.modes = max(self.lengths) // 2 + 1

    def rfft(self, x: Tensor, modes: int | None = None) -> Tensor:
        """Compute the Fourier coefficients ``[..., ring, modes]`` of real fields ``[..., points]``.

        By default, ``modes`` covers all orders that the longest ring holds. Orders that a ring cannot hold, above
        half its length, are zero.
        """
        if x.ndim < 1 or x.shape[-1] != self.points:
            raise ValueError(f"Expected {self.points} grid points")
        if not x.is_floating_point():
            raise TypeError("rfft expects a real floating-point tensor")
        modes = self.modes if modes is None else modes
        if not isinstance(modes, Integral) or modes < 1:
            raise ValueError("modes must be a positive integer")
        x = at_least_float32(x)
        if x.numel() == 0:
            return empty_transform(x, (*x.shape[:-1], self.nlat, modes), torch.promote_types(x.dtype, torch.complex64))
        return self._rfft(x, modes)

    def irfft(self, a: Tensor) -> Tensor:
        """Compute real fields ``[..., points]`` from Fourier coefficients ``[..., ring, modes]``.

        Orders above half the length of a ring are ignored, as in an inverse real FFT.
        """
        if a.ndim < 2 or a.shape[-2] != self.nlat or a.shape[-1] < 1:
            raise ValueError(f"Expected {self.nlat} rings and at least one order")
        if not a.is_complex():
            raise TypeError("irfft expects a complex tensor")
        a = at_least_float32(a)
        if a.numel() == 0:
            return empty_transform(a, (*a.shape[:-2], self.points), a.real.dtype)
        return self._irfft(a)

    def _rfft(self, x: Tensor, modes: int) -> Tensor:
        raise NotImplementedError

    def _irfft(self, a: Tensor) -> Tensor:
        raise NotImplementedError

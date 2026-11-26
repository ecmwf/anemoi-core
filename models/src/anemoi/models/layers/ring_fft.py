# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Batched PyTorch FFTs for equal-length latitude rings, with explicit adjoints."""

from collections import defaultdict
from numbers import Integral

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F


class RingFFT(Module):
    """Share packing and FFT operations between the forward, inverse and their adjoints.

    Both transforms use ``norm="forward"``. We only retain grid metadata, because
    the linear transforms do not need any saved forward activations for backward.
    Metadata is stored as nonpersistent integer buffers, so moving the parent
    SHT moves it too, and it is insensitive to dtype conversions.
    """

    def __init__(self, lengths: list[int]) -> None:
        super().__init__()
        if not lengths or any(not isinstance(n, Integral) or n < 1 for n in lengths):
            raise ValueError("Ring lengths must be positive integers")
        self.nlat = len(lengths)
        self.points = sum(lengths)
        self.modes = max(lengths) // 2 + 1
        by_length = defaultdict(list)
        offsets = np.cumsum([0, *lengths])
        # Build groups of rings of the same length
        # by_length maps ring length to list of ring indices of that length
        # groups is a tuple of (length, number of rings of that length) tuples
        for ring, n in enumerate(lengths):
            by_length[n].append(ring)
        self.groups = tuple((n, len(rings)) for n, rings in sorted(by_length.items()))

        grid_indices, spectrum_indices, mode_lengths, multiplicities = [], [], [], []
        for n, rings in sorted(by_length.items()):
            modes = n // 2 + 1
            # Interior frequencies represent both halves of the real spectrum.
            multiplicity = np.full(modes, 2, dtype=np.int64)
            multiplicity[0] = 1
            if n % 2 == 0:
                multiplicity[-1] = 1
            for ring in rings:
                grid_indices.append(np.arange(offsets[ring], offsets[ring + 1]))
                spectrum_indices.append(ring * self.modes + np.arange(modes))
                mode_lengths.append(np.full(modes, n, dtype=np.int64))
                multiplicities.append(multiplicity)
        for name, values in (
            ("grid_order", grid_indices),
            ("spectrum_order", spectrum_indices),
            ("mode_lengths", mode_lengths),
            ("multiplicity", multiplicities),
        ):
            self.register_buffer(name, torch.as_tensor(np.concatenate(values), dtype=torch.int64), persistent=False)

    def _pack_spectrum(self, x: Tensor) -> Tensor:
        # Match irfft: zero-pad missing modes and discard modes beyond Nyquist.
        x = F.pad(x[..., : self.modes], (0, max(0, self.modes - x.shape[-1])))
        return x.flatten(-2).index_select(-1, self.spectrum_order)

    def _unpack_spectrum(self, packed: Tensor) -> Tensor:
        result = packed.new_zeros((*packed.shape[:-1], self.nlat * self.modes))
        result.index_copy_(-1, self.spectrum_order, packed)
        return result.reshape(*packed.shape[:-1], self.nlat, self.modes)

    def _unpack_grid(self, packed: Tensor) -> Tensor:
        result = torch.empty_like(packed)
        result.index_copy_(-1, self.grid_order, packed)
        return result

    def _rfft_groups(self, packed: Tensor) -> Tensor:
        result = []
        offset = 0
        for n, count in self.groups:
            block = packed[..., offset : offset + count * n].reshape(*packed.shape[:-1], count, n)
            result.append(torch.fft.rfft(block, norm="backward").flatten(-2))
            offset += count * n
        return torch.cat(result, dim=-1)

    def _irfft_groups(self, packed: Tensor) -> Tensor:
        result = []
        offset = 0
        for n, count in self.groups:
            modes = n // 2 + 1
            block = packed[..., offset : offset + count * modes].reshape(*packed.shape[:-1], count, modes)
            result.append(torch.fft.irfft(block, n=n, norm="forward").flatten(-2))
            offset += count * modes
        return torch.cat(result, dim=-1)

    def rfft(self, x: Tensor) -> Tensor:
        """Transform real fields ``[..., grid]`` to padded spectra ``[..., ring, mode]``."""
        if x.ndim < 1 or x.shape[-1] != self.points:
            raise ValueError(f"Expected {self.points} grid points")
        if not x.is_floating_point():
            raise TypeError("rfft expects a real floating-point tensor")
        return _RingRFFT.apply(x, self)

    def irfft(self, x: Tensor) -> Tensor:
        """Transform spectra ``[..., ring, mode]`` to real fields ``[..., grid]``."""
        if x.ndim < 2 or x.shape[-2] != self.nlat or x.shape[-1] < 1:
            raise ValueError(f"Expected {self.nlat} rings and at least one mode")
        if not x.is_complex():
            raise TypeError("irfft expects a complex tensor")
        return _RingIRFFT.apply(x, self)


class _RingRFFT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, fft: RingFFT) -> Tensor:
        ctx.fft = fft
        # Rearrange input so rings of the same length are contiguous, and points are ordered by length of the ring they are on
        packed = x.index_select(-1, fft.grid_order)
        return fft._unpack_spectrum(fft._rfft_groups(packed) / fft.mode_lengths)

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor, None]:
        fft = ctx.fft
        # The adjoint uses 1/n at DC/even Nyquist, 1/(2*n) at interior modes.
        packed = fft._pack_spectrum(grad) / (fft.mode_lengths * fft.multiplicity)
        return fft._unpack_grid(fft._irfft_groups(packed)), None


class _RingIRFFT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, fft: RingFFT) -> Tensor:
        ctx.fft = fft
        ctx.input_modes = x.shape[-1]
        return fft._unpack_grid(fft._irfft_groups(fft._pack_spectrum(x)))

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor, None]:
        fft = ctx.fft
        # The adjoint doubles interior frequencies; DC/even Nyquist remains real
        # and is not doubled. Surplus input modes receive zero gradients.
        packed = fft._rfft_groups(grad.index_select(-1, fft.grid_order))
        spectrum = fft._unpack_spectrum(packed * fft.multiplicity)
        return F.pad(spectrum[..., : ctx.input_modes], (0, max(0, ctx.input_modes - fft.modes))), None

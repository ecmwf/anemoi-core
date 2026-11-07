# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Fourier transforms along the latitude rings of a reduced grid, with a few FFTs for all rings.

- A reduced grid has a different number of points on most rings, so plain FFTs need one call per ring
  length: more than a thousand small calls per transform at O1280.
- With m k = (m^2 + k^2 - (m - k)^2) / 2, a ring's Fourier sum becomes a convolution with the "chirp"
  exp(i pi j^2 / n) (Bluestein 1970). A convolution can be padded with zeros to any length of at least
  n + M - 1, where M is the number of orders, so rings of different lengths share one FFT length.
- Each field has its own complex transform, keeping its rounding and non-finite values independent of
  other fields in the batch.
- The rings are grouped by that padded length, a power of 2. Each group takes one batched FFT, one
  product and one inverse FFT.
- Each group keeps its chirp data between calls: the FFT of its chirp and the factors on its orders,
  about one complex value per padded point for each number of orders and direction in use.
- The cached kernel includes division by the padded length. The convolution's inverse FFT is unnormalized,
  saving a scaling pass on every call. The padded length is a power of 2, so this scaling is exact within
  the normal floating-point range.
- The chirp phases use j^2 mod 2 n in integers, so they stay exact on long rings. All chirp data is worked
  out in float64 and then rounded, because errors in these factors reach every value of the transform.
- The scaling matches ``norm="forward"``: the forward sum is divided by the ring length, the inverse is
  not. The gradients are the adjoints of the two sums.
"""

from functools import lru_cache
from numbers import Integral

import numpy as np
import torch
from torch import Tensor

# Fields go through in chunks that limit one padded buffer to this many complex values, unless a single
# field already exceeds the limit. Outputs, cached tables and FFT workspaces need additional memory.
_MAX_BUFFER_VALUES = 2**26


def _padded_length(needed: int) -> int:
    """The smallest power of 2 that is at least ``needed``."""
    return 1 << max(0, needed - 1).bit_length()


def _multiplicity(m: Tensor, n: Tensor) -> Tensor:
    # Orders 0 and n / 2 appear once in a real spectrum, all others twice.
    return torch.where((m == 0) | (2 * m == n), 1.0, 2.0)


def _complex_type(real: torch.dtype) -> torch.dtype:
    return torch.complex64 if real == torch.float32 else torch.complex128


def _phase(j: Tensor, n: Tensor, sign: float) -> Tensor:
    """exp(sign i pi j^2 / n) in float64, from j^2 mod 2 n."""
    angle = torch.pi * ((j * j) % (2 * n)).to(torch.float64) / n.to(torch.float64)
    return torch.polar(torch.ones_like(angle), sign * angle)


def _real_at_ends(coefficients: Tensor, ends: Tensor) -> Tensor:
    """Drop the imaginary part at orders 0 and n / 2 (``ends``), which are real for a real field."""
    return torch.where(ends, torch.complex(coefficients.real, torch.zeros_like(coefficients.real)), coefficients)


class _Group:
    """Rings that share a padded length, with their place in the padded buffer."""

    def __init__(self, size: int, rings: list[int], offset: int, lengths: Tensor, modes: int, taken: int) -> None:
        self.size = size
        # Orders worked out for the group: the most that any of its rings holds.
        self.taken = taken
        self.offset = offset
        self.count = len(rings)
        self.rings = torch.as_tensor(rings, dtype=torch.int64, device=lengths.device)
        self.n = lengths[self.rings][:, None]
        self.modes = torch.clamp(self.n // 2 + 1, max=modes)
        self.cache = {}

    def data(self, key: tuple, build):
        """Chirp data of this group, built the first time it is asked for and then kept."""
        if key not in self.cache:
            self.cache[key] = build()
        return self.cache[key]


@lru_cache(maxsize=None)
def shared_ring_fft(lengths: tuple[int, ...]) -> "RingFFT":
    """The ring transforms of a grid, shared by every transform on it."""
    return RingFFT(lengths)


class RingFFT:
    """Ring Fourier sums of a reduced grid, and their adjoints, with Bluestein's padding trick.

    The tables are built on each device the first time they are needed there, and kept. Use
    ``shared_ring_fft`` to share one instance between the transforms on a grid.

    Parameters
    ----------
    lengths : tuple[int, ...]
        Number of points on each ring, in the order the rings are stored.
    """

    def __init__(self, lengths: tuple[int, ...]) -> None:
        if not lengths or any(not isinstance(n, Integral) or n < 1 for n in lengths):
            raise ValueError("Ring lengths must be positive integers")
        self.lengths = tuple(int(n) for n in lengths)
        self.nlat = len(lengths)
        self.points = sum(self.lengths)
        self.modes = max(self.lengths) // 2 + 1
        self._tables = {}
        self._plans = {}
        self._chirps = {}

    def _table(self, device: torch.device) -> dict[str, Tensor]:
        """Ring lengths, and the ring and position of each point, on ``device``."""
        if device not in self._tables:
            lengths = np.asarray(self.lengths)
            self._tables[device] = {
                "ring_length": torch.as_tensor(lengths, dtype=torch.int64, device=device),
                "point_ring": torch.as_tensor(np.repeat(np.arange(self.nlat), lengths), device=device),
                "point_k": torch.as_tensor(np.concatenate([np.arange(n) for n in self.lengths]), device=device),
            }
        return self._tables[device]

    def _plan(self, modes: int, device: torch.device) -> tuple[list[_Group], Tensor, int]:
        """The groups of rings for ``modes`` orders, the place of each grid point in the padded buffer, and the
        buffer length.
        """
        key = (modes, device)
        if key not in self._plans:
            table = self._table(device)
            by_size = {}
            for ring, n in enumerate(self.lengths):
                by_size.setdefault(_padded_length(n + min(modes, n // 2 + 1) - 1), []).append(ring)
            groups, start, offset = [], torch.empty(self.nlat, dtype=torch.int64), 0
            for size, rings in sorted(by_size.items()):
                taken = max(min(modes, self.lengths[ring] // 2 + 1) for ring in rings)
                groups.append(_Group(size, rings, offset, table["ring_length"], modes, taken))
                start[rings] = offset + size * torch.arange(len(rings))
                offset += size * len(rings)
            self._plans[key] = (groups, start.to(device), offset)
        groups, start, offset = self._plans[key]
        # The place of each point is worked out on every call: kept, it would take 8 bytes per point for
        # every number of orders in use.
        table = self._table(device)
        return groups, start[table["point_ring"]] + table["point_k"], offset

    def _chirp(self, real: torch.dtype, device: torch.device) -> Tensor:
        """exp(-i pi k^2 / n) at each point, from k^2 mod 2 n, in the complex type that goes with ``real``."""
        key = (real, device)
        if key not in self._chirps:
            k = np.concatenate([np.arange(n) for n in self.lengths]).astype(np.int64)
            n = np.repeat(np.asarray(self.lengths), self.lengths)
            chirp = _phase(torch.as_tensor(k, device=device), torch.as_tensor(n, device=device), -1.0)
            self._chirps[key] = chirp.to(_complex_type(real))
        return self._chirps[key]

    @staticmethod
    def _chirp_kernel(group: _Group, ahead: Tensor, behind: Tensor, sign: float, real: torch.dtype) -> Tensor:
        """Normalized FFT of exp(sign i pi j^2 / n), at j = 0 .. ahead - 1 and size + j for j = -behind .. -1.

        Include the convolution's 1 / size normalization here so each call can use an unnormalized inverse FFT.
        """
        size = group.size
        p = torch.arange(size, device=group.n.device)[None, :]
        j = torch.where(p < ahead, p, torch.where(p >= size - behind, p - size, size))
        return torch.fft.fft((j < size) * _phase(j, group.n, sign), dim=-1, norm="forward").to(_complex_type(real))

    @staticmethod
    def _order_chirp(group: _Group, taken: int, weight: Tensor, sign: float, real: torch.dtype) -> Tensor:
        """weight * exp(sign i pi m^2 / n) for the orders m < taken that the ring holds, zero above."""
        m = torch.arange(taken, device=group.n.device)[None, :]
        return ((m < group.modes) * weight * _phase(m, group.n, sign)).to(_complex_type(real))

    @staticmethod
    def _ends(group: _Group, taken: int) -> Tensor:
        """Orders 0 and n / 2 of each ring of the group, ``[ring, taken]``."""
        m = torch.arange(taken, device=group.n.device)[None, :]
        return (m == 0) | (2 * m == group.n)

    def _inverse_weights(self, group: _Group, weight_by_length: bool, real: torch.dtype) -> Tensor:
        """Weight of each nonnegative order times exp(i pi m^2 / n), ``[ring, taken]``."""
        m = torch.arange(group.taken, device=group.n.device)[None, :]
        weight = 1.0 / group.n.to(torch.float64) if weight_by_length else _multiplicity(m, group.n).to(torch.float64)
        return self._order_chirp(group, group.taken, weight, 1.0, real)

    def _field_groups(self, batch: int, buffer: int) -> list[slice]:
        step = max(1, _MAX_BUFFER_VALUES // max(buffer, 1))
        return [slice(start, min(start + step, batch)) for start in range(0, batch, step)]

    def _forward_sums(self, x: Tensor, modes: int) -> Tensor:
        """(1 / n) sum over k of x[k] exp(-2 pi i m k / n) for m < modes, ``[batch, ring, modes]``."""
        real, batch = x.dtype, x.shape[0]
        groups, index, buffer_size = self._plan(modes, x.device)
        chirp = self._chirp(real, x.device)
        out = x.new_zeros((batch, self.nlat, modes), dtype=_complex_type(real))
        for fields in self._field_groups(batch, buffer_size):
            buffer = out.new_zeros((fields.stop - fields.start, buffer_size))
            buffer.index_copy_(1, index, x[fields] * chirp)
            for group in groups:
                taken = group.taken
                kernel = group.data(
                    ("forward kernel", real),
                    lambda: self._chirp_kernel(group, group.modes, group.n - 1, 1.0, real),
                )
                post = group.data(
                    ("forward post", real),
                    lambda: self._order_chirp(group, taken, 1.0 / group.n.to(torch.float64), -1.0, real),
                )
                ends = group.data(("ends",), lambda: self._ends(group, taken))
                block = buffer[:, group.offset : group.offset + group.size * group.count].view(
                    -1, group.count, group.size
                )
                sums = torch.fft.ifft(torch.fft.fft(block, dim=-1) * kernel, dim=-1, norm="forward")
                out[fields, group.rings, :taken] = _real_at_ends(sums[..., :taken] * post, ends)
        return out

    def _inverse_sums(self, a: Tensor, weight_by_length: bool) -> Tensor:
        """Sum over m of w(m) Re(a[m] exp(2 pi i m k / n)), ``[batch, points]``.

        w is the multiplicity of the order, or 1 / n with ``weight_by_length``. Orders above n / 2 are left
        out, as in an inverse real FFT.
        """
        real, batch, modes = a.real.dtype, a.shape[0], a.shape[-1]
        groups, index, buffer_size = self._plan(modes, a.device)
        chirp = self._chirp(real, a.device).conj()
        out = a.real.new_empty((batch, self.points))
        for fields in self._field_groups(batch, buffer_size):
            buffer = a.new_zeros((fields.stop - fields.start, buffer_size))
            for group in groups:
                taken = group.taken
                weights = group.data(
                    ("inverse weights", real, weight_by_length),
                    lambda: self._inverse_weights(group, weight_by_length, real),
                )
                kernel = group.data(
                    ("inverse kernel", real),
                    lambda: self._chirp_kernel(group, group.n, group.modes - 1, -1.0, real),
                )
                ends = group.data(("ends",), lambda: self._ends(group, taken))
                block = buffer[:, group.offset : group.offset + group.size * group.count].view(
                    -1, group.count, group.size
                )
                block[..., :taken] = _real_at_ends(a[fields, group.rings, :taken], ends) * weights
                block.copy_(torch.fft.ifft(torch.fft.fft(block, dim=-1) * kernel, dim=-1, norm="forward"))
            out[fields] = (buffer[:, index] * chirp).real
        return out

    def rfft(self, x: Tensor, modes: int | None = None) -> Tensor:
        """Fourier coefficients ``[..., ring, modes]`` of real fields ``[..., points]``.

        ``modes`` defaults to all orders the longest ring holds. Orders a ring cannot hold (above half its
        length) are zero.
        """
        if x.ndim < 1 or x.shape[-1] != self.points:
            raise ValueError(f"Expected {self.points} grid points")
        if not x.is_floating_point():
            raise TypeError("rfft expects a real floating-point tensor")
        modes = self.modes if modes is None else modes
        leading = x.shape[:-1]
        out = _RingRFFT.apply(x.reshape(-1, self.points), self, modes)
        return out.reshape(*leading, self.nlat, modes)

    def irfft(self, a: Tensor) -> Tensor:
        """Real fields ``[..., points]`` from Fourier coefficients ``[..., ring, modes]``.

        Orders above half a ring's length are ignored, as in an inverse real FFT.
        """
        if a.ndim < 2 or a.shape[-2] != self.nlat or a.shape[-1] < 1:
            raise ValueError(f"Expected {self.nlat} rings and at least one order")
        if not a.is_complex():
            raise TypeError("irfft expects a complex tensor")
        leading = a.shape[:-2]
        out = _RingIRFFT.apply(a.reshape(-1, *a.shape[-2:]), self)
        return out.reshape(*leading, self.points)


class _RingRFFT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, fft: RingFFT, modes: int) -> Tensor:
        ctx.fft = fft
        return fft._forward_sums(x, modes)

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor, None, None]:
        # The adjoint of (1 / n) sum x exp(-i phi) is (1 / n) Re sum g exp(i phi) over all orders it gave.
        return ctx.fft._inverse_sums(grad, weight_by_length=True), None, None


class _RingIRFFT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: Tensor, fft: RingFFT) -> Tensor:
        ctx.fft = fft
        ctx.modes = a.shape[-1]
        return fft._inverse_sums(a, weight_by_length=False)

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor, None]:
        # The adjoint of sum w Re(a exp(i phi)) is w sum g exp(-i phi), zero for orders the ring cannot hold.
        fft = ctx.fft
        sums = fft._forward_sums(grad, ctx.modes)
        m = torch.arange(ctx.modes, device=grad.device)
        n = fft._table(grad.device)["ring_length"][:, None]
        return sums * (n * _multiplicity(m[None, :], n)).to(grad.dtype), None

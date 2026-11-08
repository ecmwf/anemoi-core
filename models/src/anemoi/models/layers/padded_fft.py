# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Fourier transforms along the latitude rings of a reduced grid, with a few FFTs for all rings.

A reduced grid has a different number of points on most rings, so plain FFTs need one call per ring length, which
at O1280 means more than a thousand small calls per transform. For the order m, the number of waves around a ring,
and the point k on a ring of n points, the identity m k = (m^2 + k^2 - (m - k)^2) / 2 turns the Fourier sum into a
convolution with the chirp exp(i pi j^2 / n) (Bluestein 1970). This convolution can be padded with zeros to any
length of at least n + M - 1, where M is the number of orders, so rings of different lengths share one FFT length.
The rings are grouped by this padded length, a power of 2, and each group takes one batched FFT, one product and
one inverse FFT. Each field has its own complex transform, so its rounding and its non-finite values are
independent of the other fields in the batch.

Each group keeps its chirp data between calls, which consists of the FFT of its chirp and the factors on its
orders, about one complex value per padded point for each number of orders and direction in use. The cached
kernel includes the division by the padded length, so the inverse FFT of the convolution runs unnormalized and
saves a scaling pass on every call. Since the padded length is a power of 2, this scaling is exact within the
normal floating-point range. The chirp phases are computed from j^2 mod 2 n in integers and stay exact on long
rings. All chirp data is computed in float64 and then rounded, because errors in these factors reach every
value of the transform.

The scaling matches ``norm="forward"``, with the forward sum divided by the ring length and the inverse sum
left unscaled. The gradients are the adjoints of the two sums.
"""

import numpy as np
import torch
from torch import Tensor

from anemoi.models.layers.ring_fft_base import RingFFTBase

# Fields go through in chunks that limit one padded buffer to this many complex values, unless a single
# field already exceeds the limit. Outputs, cached tables and FFT workspaces need additional memory.
_MAX_BUFFER_VALUES = 2**26


def _padded_length(needed: int) -> int:
    """Return the smallest power of 2 that is at least ``needed``."""
    return 1 << max(0, needed - 1).bit_length()


def _multiplicity(m: Tensor, n: Tensor) -> Tensor:
    # Orders 0 and n / 2 appear once in a real spectrum, all others twice.
    return torch.where((m == 0) | (2 * m == n), 1.0, 2.0)


def _complex_type(real: torch.dtype) -> torch.dtype:
    return torch.complex128 if real == torch.float64 else torch.complex64


def _phase(j: Tensor, n: Tensor, sign: float) -> Tensor:
    """Compute exp(sign i pi j^2 / n) in float64 from j^2 mod 2 n."""
    angle = torch.pi * ((j * j) % (2 * n)).to(torch.float64) / n.to(torch.float64)
    return torch.polar(torch.ones_like(angle), sign * angle)


def _kept_parts(coefficients: Tensor, kept: Tensor) -> Tensor:
    """Zero the real or imaginary components excluded by ``kept``, including NaN and Inf."""
    return torch.view_as_complex(torch.where(kept, torch.view_as_real(coefficients), 0))


def _pack_points(x: Tensor, chirp: Tensor, index: Tensor, buffer_size: int) -> Tensor:
    """Multiply by the chirp and scatter into the zero padding, with real and imaginary parts on a final axis."""
    weighted = x[..., None] * chirp
    buffer = weighted.new_zeros((x.shape[0], buffer_size, 2))
    return buffer.index_copy_(1, index, weighted)


def _unpack_points(buffer: Tensor, chirp: Tensor, index: Tensor) -> Tensor:
    """Gather the grid points and take Re(value * conj(chirp)), with inputs split into real and imaginary parts.

    The split lets the compiler fuse the gather with the product.
    """
    if not torch.compiler.is_compiling():
        # Eager complex multiplication uses fewer kernels. Inductor needs real arithmetic to fuse the gather.
        return (torch.view_as_complex(buffer)[:, index] * torch.view_as_complex(chirp).conj()).real
    values = buffer[:, index]
    return values[..., 0] * chirp[:, 0] + values[..., 1] * chirp[:, 1]


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
        """Return chirp data of this group, which is built on first use and then kept."""
        if key not in self.cache:
            self.cache[key] = build()
        return self.cache[key]


class PaddedFFT(RingFFTBase):
    """Ring Fourier sums of a reduced grid and their adjoints, computed with Bluestein's padding trick.

    The tables are built on each device when they are first needed there and are kept afterwards. Transforms on
    the same grid share one instance through ``ring_fft.shared_ring_fft(lengths, "padded", False)``.

    Parameters
    ----------
    lengths : tuple[int, ...]
        Number of points on each ring, in the order the rings are stored.
    """

    def __init__(self, lengths: tuple[int, ...]) -> None:
        super().__init__(lengths)
        self._tables = {}
        self._plans = {}
        self._chirps = {}
        self._pack = _pack_points
        self._unpack = _unpack_points

    def compile_for_training(self) -> None:
        """Compile the packing and gathering of grid points, while the FFTs and the table construction stay eager.

        The shapes can vary between truncations and field chunks. CUDA graphs are disabled, so the compiled code
        keeps no graph input and output buffers. The chirp tables are shared with the eager code.
        """
        options = {"triton.cudagraphs": False}
        self._pack = torch.compile(_pack_points, fullgraph=True, dynamic=True, options=options)
        self._unpack = torch.compile(_unpack_points, fullgraph=True, dynamic=True, options=options)

    def _table(self, device: torch.device) -> dict[str, Tensor]:
        """Return the ring lengths, and the ring and position of each point, on ``device``."""
        if device not in self._tables:
            lengths = np.asarray(self.lengths)
            self._tables[device] = {
                "ring_length": torch.as_tensor(lengths, dtype=torch.int64, device=device),
                "point_ring": torch.as_tensor(np.repeat(np.arange(self.nlat), lengths), device=device),
                "point_k": torch.as_tensor(np.concatenate([np.arange(n) for n in self.lengths]), device=device),
            }
        return self._tables[device]

    def _plan(self, modes: int, device: torch.device) -> tuple[list[_Group], Tensor, int]:
        """Return the groups of rings for ``modes`` orders, the place of each grid point in the padded buffer and
        the buffer length.
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
        # The place of each point is worked out on every call, since keeping it would take 8 bytes per point
        # for every number of orders in use.
        table = self._table(device)
        return groups, start[table["point_ring"]] + table["point_k"], offset

    def _chirp(self, real: torch.dtype, device: torch.device) -> Tensor:
        """Return exp(-i pi k^2 / n) at each point from k^2 mod 2 n, in the complex type that matches ``real``."""
        key = (real, device)
        if key not in self._chirps:
            k = np.concatenate([np.arange(n) for n in self.lengths]).astype(np.int64)
            n = np.repeat(np.asarray(self.lengths), self.lengths)
            chirp = _phase(torch.as_tensor(k, device=device), torch.as_tensor(n, device=device), -1.0)
            self._chirps[key] = chirp.to(_complex_type(real))
        return self._chirps[key]

    @staticmethod
    def _chirp_kernel(group: _Group, ahead: Tensor, behind: Tensor, sign: float, real: torch.dtype) -> Tensor:
        """Return the normalized FFT of the chirp exp(sign i pi j^2 / n).

        The chirp is taken at j = 0 .. ahead - 1 and at size + j for j = -behind .. -1. The kernel includes the
        1 / size normalization of the convolution, so each call can use an unnormalized inverse FFT.
        """
        size = group.size
        p = torch.arange(size, device=group.n.device)[None, :]
        j = torch.where(p < ahead, p, torch.where(p >= size - behind, p - size, size))
        return torch.fft.fft((j < size) * _phase(j, group.n, sign), dim=-1, norm="forward").to(_complex_type(real))

    @staticmethod
    def _order_chirp(group: _Group, taken: int, weight: Tensor, sign: float, real: torch.dtype) -> Tensor:
        """Return weight * exp(sign i pi m^2 / n) at the orders m < taken that the ring holds, and zero above."""
        m = torch.arange(taken, device=group.n.device)[None, :]
        return ((m < group.modes) * weight * _phase(m, group.n, sign)).to(_complex_type(real))

    @staticmethod
    def _kept(group: _Group) -> Tensor:
        """Return which real and imaginary parts are kept at the orders m < taken of each ring, ``[ring, taken, 2]``.

        Orders that a ring cannot hold are dropped. At the orders 0 and n / 2, only the real part is kept, since a
        real field has no imaginary part there.
        """
        m = torch.arange(group.taken, device=group.n.device)[None, :]
        held = m < group.modes
        return torch.stack((held, held & (m != 0) & (2 * m != group.n)), dim=-1)

    def _inverse_weights(self, group: _Group, weight_by_length: bool, real: torch.dtype) -> Tensor:
        """Return the weight of each nonnegative order times exp(i pi m^2 / n), ``[ring, taken]``."""
        m = torch.arange(group.taken, device=group.n.device)[None, :]
        weight = 1.0 / group.n.to(torch.float64) if weight_by_length else _multiplicity(m, group.n).to(torch.float64)
        return self._order_chirp(group, group.taken, weight, 1.0, real)

    def _field_groups(self, batch: int, buffer: int) -> list[slice]:
        step = max(1, _MAX_BUFFER_VALUES // max(buffer, 1))
        return [slice(start, min(start + step, batch)) for start in range(0, batch, step)]

    def _forward_sums(self, x: Tensor, modes: int) -> Tensor:
        """Compute (1 / n) sum over k of x[k] exp(-2 pi i m k / n) for m < modes, ``[batch, ring, modes]``."""
        real, batch = x.dtype, x.shape[0]
        groups, index, buffer_size = self._plan(modes, x.device)
        chirp = self._chirp(real, x.device)
        out = x.new_zeros((batch, self.nlat, modes), dtype=_complex_type(real))
        for fields in self._field_groups(batch, buffer_size):
            buffer = torch.view_as_complex(self._pack(x[fields], torch.view_as_real(chirp), index, buffer_size))
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
                kept = group.data(("kept",), lambda: self._kept(group))
                block = buffer[:, group.offset : group.offset + group.size * group.count].view(
                    -1, group.count, group.size
                )
                sums = torch.fft.ifft(torch.fft.fft(block, dim=-1) * kernel, dim=-1, norm="forward")
                out[fields, group.rings, :taken] = _kept_parts(sums[..., :taken] * post, kept)
        return out

    def _inverse_sums(self, a: Tensor, weight_by_length: bool) -> Tensor:
        """Compute the sum over m of w(m) Re(a[m] exp(2 pi i m k / n)), ``[batch, points]``.

        Here, w is the multiplicity of the order, or 1 / n with ``weight_by_length``. Orders above n / 2 are left
        out, as in an inverse real FFT.
        """
        real, batch, modes = a.real.dtype, a.shape[0], a.shape[-1]
        groups, index, buffer_size = self._plan(modes, a.device)
        chirp = self._chirp(real, a.device)
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
                kept = group.data(("kept",), lambda: self._kept(group))
                block = buffer[:, group.offset : group.offset + group.size * group.count].view(
                    -1, group.count, group.size
                )
                block[..., :taken] = _kept_parts(a[fields, group.rings, :taken], kept) * weights
                block.copy_(torch.fft.ifft(torch.fft.fft(block, dim=-1) * kernel, dim=-1, norm="forward"))
            out[fields] = self._unpack(torch.view_as_real(buffer), torch.view_as_real(chirp), index)
        return out

    def _rfft(self, x: Tensor, modes: int) -> Tensor:
        leading = x.shape[:-1]
        out = _RingRFFT.apply(x.reshape(-1, self.points), self, modes)
        return out.reshape(*leading, self.nlat, modes)

    def _irfft(self, a: Tensor) -> Tensor:
        leading = a.shape[:-2]
        out = _RingIRFFT.apply(a.reshape(-1, *a.shape[-2:]), self)
        return out.reshape(*leading, self.points)


class _RingRFFT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, fft: PaddedFFT, modes: int) -> Tensor:
        ctx.fft = fft
        # The backward pass below gives the gradient, so the input is detached, and the compiler then builds
        # one version for inputs with and without gradients.
        return fft._forward_sums(x.detach(), modes)

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor, None, None]:
        # The adjoint of (1 / n) sum x exp(-i phi) is (1 / n) Re sum g exp(i phi) over all orders it gave.
        return ctx.fft._inverse_sums(grad, weight_by_length=True), None, None


class _RingIRFFT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: Tensor, fft: PaddedFFT) -> Tensor:
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

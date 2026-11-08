# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Native FFTs grouped by ring length, optional CUDA graph capture, and backend selection."""

import gc
import logging
from collections import defaultdict
from functools import lru_cache
from functools import partial
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.cuda.graphs import make_graphed_callables

from anemoi.models.layers.padded_fft import PaddedFFT
from anemoi.models.layers.ring_fft_base import RingFFTBase

LOGGER = logging.getLogger(__name__)

FFTBackend = Literal["ring", "padded"]


def validate_fft_options(backend: str, graphed: bool = False) -> None:
    """Validate the backend selection before any tables or graph wrappers are built."""
    if backend not in ("ring", "padded"):
        raise ValueError(f"Unknown fft_backend={backend!r}; use 'ring' or 'padded'")
    if backend == "padded" and graphed:
        raise ValueError("CUDA graphs require fft_backend='ring'; padded FFTs do not support graph capture")


def shared_ring_fft(
    lengths: tuple[int, ...] | list[int], backend: FFTBackend = "ring", graphed: bool = False
) -> "NativeRingFFT | PaddedFFT | GraphedRingFFT":
    """Return the ring FFTs of one grid with the chosen backend, shared by all transforms on that grid.

    For a given grid, backend and graph setting, equivalent positional and keyword calls return the same
    instance. ``shared_ring_fft.cache_clear()`` clears the factory cache.
    """
    validate_fft_options(backend, graphed)
    return _ring_fft(tuple(lengths), backend, bool(graphed))


@lru_cache(maxsize=None)
def _ring_fft(
    lengths: tuple[int, ...], backend: FFTBackend, graphed: bool
) -> "NativeRingFFT | PaddedFFT | GraphedRingFFT":
    if backend == "padded":
        return PaddedFFT(lengths)
    if graphed:
        return GraphedRingFFT(lengths)
    return NativeRingFFT(lengths)


shared_ring_fft.cache_clear = _ring_fft.cache_clear
shared_ring_fft.cache_info = _ring_fft.cache_info


class NativeRingFFT(RingFFTBase):
    """Real FFTs on the rings of a reduced grid, with one batched FFT for all rings of the same length.

    Both directions are scaled as with ``norm="forward"``. The backward pass uses explicit adjoints and the grid
    metadata, so the input fields are not saved. Each ring is transformed at its own length, and the requested
    number of orders determines how many coefficients are kept.
    """

    def __init__(self, lengths: tuple[int, ...]) -> None:
        super().__init__(lengths)
        by_length = defaultdict(list)
        offsets = np.cumsum([0, *self.lengths])
        for ring, n in enumerate(self.lengths):
            by_length[n].append(ring)
        self.groups = tuple((n, len(rings)) for n, rings in sorted(by_length.items()))
        grid_order, spectrum_ring, spectrum_mode, mode_lengths, multiplicity = [], [], [], [], []
        for n, rings in sorted(by_length.items()):
            modes = n // 2 + 1
            weights = np.full(modes, 2, dtype=np.int64)
            weights[0] = 1
            if n % 2 == 0:
                weights[-1] = 1
            for ring in rings:
                grid_order.append(np.arange(offsets[ring], offsets[ring + 1]))
                spectrum_ring.append(np.full(modes, ring, dtype=np.int64))
                spectrum_mode.append(np.arange(modes))
                mode_lengths.append(np.full(modes, n, dtype=np.int64))
                multiplicity.append(weights)
        self._grid_order = np.concatenate(grid_order)
        self._spectrum_ring = np.concatenate(spectrum_ring)
        self._spectrum_mode = np.concatenate(spectrum_mode)
        self._mode_lengths = np.concatenate(mode_lengths)
        self._multiplicity = np.concatenate(multiplicity)
        self._tables = {}
        self._spectra = {}

    def _table(self, device: torch.device) -> dict[str, Tensor]:
        if device not in self._tables:
            self._tables[device] = {
                "grid_order": torch.as_tensor(self._grid_order, device=device),
                "mode_lengths": torch.as_tensor(self._mode_lengths, device=device),
                "multiplicity": torch.as_tensor(self._multiplicity, device=device),
            }
        return self._tables[device]

    def _spectrum_indices(self, modes: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Return the positions of the kept orders in the packed spectra and in [ring, order]."""
        key = (modes, device)
        if key not in self._spectra:
            kept = np.flatnonzero(self._spectrum_mode < modes)
            order = self._spectrum_ring[kept] * modes + self._spectrum_mode[kept]
            self._spectra[key] = (torch.as_tensor(kept, device=device), torch.as_tensor(order, device=device))
        return self._spectra[key]

    def _pack_spectrum(self, x: Tensor) -> Tensor:
        kept, order = self._spectrum_indices(x.shape[-1], x.device)
        values = x.flatten(-2).index_select(-1, order)
        if x.shape[-1] >= self.modes:
            return values
        packed = x.new_zeros((*x.shape[:-2], len(self._spectrum_mode)))
        return packed.index_copy_(-1, kept, values)

    def _unpack_spectrum(self, packed: Tensor, modes: int) -> Tensor:
        kept, order = self._spectrum_indices(modes, packed.device)
        out = packed.new_zeros((*packed.shape[:-1], self.nlat * modes))
        values = packed if modes >= self.modes else packed.index_select(-1, kept)
        out.index_copy_(-1, order, values)
        return out.reshape(*packed.shape[:-1], self.nlat, modes)

    def _unpack_grid(self, packed: Tensor) -> Tensor:
        out = torch.empty_like(packed)
        return out.index_copy_(-1, self._table(packed.device)["grid_order"], packed)

    def _rfft_groups(self, packed: Tensor) -> Tensor:
        results, offset = [], 0
        for n, count in self.groups:
            block = packed[..., offset : offset + count * n].reshape(*packed.shape[:-1], count, n)
            results.append(torch.fft.rfft(block, norm="backward").flatten(-2))
            offset += count * n
        return torch.cat(results, dim=-1)

    def _irfft_groups(self, packed: Tensor) -> Tensor:
        results, offset = [], 0
        for n, count in self.groups:
            modes = n // 2 + 1
            block = packed[..., offset : offset + count * modes].reshape(*packed.shape[:-1], count, modes)
            results.append(torch.fft.irfft(block, n=n, norm="forward").flatten(-2))
            offset += count * modes
        return torch.cat(results, dim=-1)

    def _rfft(self, x: Tensor, modes: int) -> Tensor:
        return _NativeRFFT.apply(x, self, modes)

    def _irfft(self, a: Tensor) -> Tensor:
        return _NativeIRFFT.apply(a, self)


class _NativeRFFT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, fft: NativeRingFFT, modes: int) -> Tensor:
        ctx.fft = fft
        table = fft._table(x.device)
        packed = x.index_select(-1, table["grid_order"])
        return fft._unpack_spectrum(fft._rfft_groups(packed) / table["mode_lengths"], modes)

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor, None, None]:
        fft = ctx.fft
        table = fft._table(grad.device)
        # The real inverse counts each interior order twice, once for each half of the spectrum, so the adjoint
        # halves them.
        packed = fft._pack_spectrum(grad) / (table["mode_lengths"] * table["multiplicity"])
        return fft._unpack_grid(fft._irfft_groups(packed)), None, None


class _NativeIRFFT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: Tensor, fft: NativeRingFFT) -> Tensor:
        ctx.fft, ctx.modes = fft, a.shape[-1]
        return fft._unpack_grid(fft._irfft_groups(fft._pack_spectrum(a)))

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor, None]:
        fft = ctx.fft
        table = fft._table(grad.device)
        packed = fft._rfft_groups(grad.index_select(-1, table["grid_order"]))
        return fft._unpack_spectrum(packed * table["multiplicity"], ctx.modes), None


class GraphedRingFFT(RingFFTBase):
    """Native ring FFTs, captured in CUDA graphs in up to three latitude bands and then replayed.

    A capture is made on first use for each direction, number of orders, shape, dtype and device, and for inputs
    with and without gradients. It keeps its own input, output and gradient buffers and its memory pool for the
    life of the process. The backward pass does not support gradients of gradients.

    A replay runs the cuFFT plans that its capture was made with. PyTorch keeps these plans in its plan cache and
    frees the least recently used ones when the cache is full, and a replay after one of its plans has been freed
    is unsafe. The cache, sized with ``training.cufft_plan_cache_size``, should therefore hold every plan of the
    run and should not be cleared.
    """

    def __init__(self, lengths: tuple[int, ...]) -> None:
        super().__init__(lengths)
        LOGGER.warning(
            "CUDA graphs for ring FFTs replay cuFFT plans from PyTorch's plan cache; a replay after the cache has "
            "freed one of them is unsafe. Set training.cufft_plan_cache_size above the number of plans the run "
            "needs."
        )
        bands = min(3, self.nlat)
        self.ffts, self.grid_slices = [], []
        offset = 0
        for band in range(bands):
            start, end = band * self.nlat // bands, (band + 1) * self.nlat // bands
            fft = shared_ring_fft(self.lengths[start:end], "ring", False)
            self.ffts.append(fft)
            self.grid_slices.append(slice(offset, offset + fft.points))
            offset += fft.points
        self._captures = {}

    def _run(self, x: Tensor, inverse: bool, modes: int | None = None) -> Tensor:
        if x.device.type != "cuda":
            raise RuntimeError("CUDA graphs require CUDA input")
        requires_grad = x.requires_grad and torch.is_grad_enabled()
        key = (inverse, modes, tuple(x.shape), x.dtype, x.device, requires_grad)
        if inverse:
            inputs = list(x.split([fft.nlat for fft in self.ffts], dim=-2))
            functions = tuple(fft.irfft for fft in self.ffts)
        else:
            inputs = [x[..., part] for part in self.grid_slices]
            functions = tuple(partial(fft.rfft, modes=modes) for fft in self.ffts)
        if key not in self._captures:
            # Collect unreachable objects before capture, since their destructors may call CUDA.
            gc_enabled = gc.isenabled()
            gc.disable()
            try:
                gc.collect()
                # The capture's buffers are made outside inference mode, so that a capture made during
                # validation can still take new inputs in training.
                with (
                    torch.inference_mode(False),
                    torch.cuda.device(x.device),
                    torch.amp.autocast("cuda", enabled=torch.is_autocast_enabled(), cache_enabled=False),
                ):
                    samples = tuple((torch.zeros_like(band, requires_grad=requires_grad),) for band in inputs)
                    self._captures[key] = make_graphed_callables(functions, samples)
            finally:
                if gc_enabled:
                    gc.enable()
        # Joining the bands copies the result, so a later replay cannot overwrite what this call returned.
        return torch.cat([fn(band) for fn, band in zip(self._captures[key], inputs)], dim=-1 if inverse else -2)

    def _rfft(self, x: Tensor, modes: int) -> Tensor:
        return self._run(x, inverse=False, modes=modes)

    def _irfft(self, a: Tensor) -> Tensor:
        return self._run(a, inverse=True)

# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Gaussian smoothing over a fixed neighbour list.

    y[n] = sum_k w[n, k] * x[idx[n, k]] / sum_k w[n, k],   w[n, k] = exp(-d[n, k]**2 / (2 * sigma**2))

``d`` is the great-circle distance to the neighbour, computed from the unit vectors of the points
in the forward and the backward pass. No weights are stored, so one neighbour list
(:class:`anemoi.models.layers.graph_provider.NeighbourListProvider`) serves any ``sigma``.

Backends: ``"triton"`` (:mod:`anemoi.models.triton.gaussian_smoothing`) and ``"torch"``, which
runs on any device, eagerly in small bands or compiled (:class:`CompiledGaussianSmoothing`).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from anemoi.models.layers.projection_backend import PROJECTION_BACKENDS
from anemoi.models.layers.projection_backend import ProjectionBackend
from anemoi.models.layers.projection_backend import accumulation_dtype

# Neighbour values gathered at once by the eager torch backend; bounds its extra memory.
_TORCH_GATHER_ELEMENTS = 2**26


def squared_neighbour_distances(
    unit_vectors: Tensor, neighbours: Tensor, points: slice | Tensor, dtype: torch.dtype
) -> Tensor:
    """Squared great-circle distances in radians from each point in ``points`` to its neighbours."""
    return _squared_angles(unit_vectors[points], unit_vectors[neighbours[points].long()], dtype)


def _squared_angles(here: Tensor, there: Tensor, dtype: torch.dtype) -> Tensor:
    """Squared great-circle distances from ``here`` (n, 3) to ``there`` (n, k, 3)."""
    chord = (there.to(dtype) - here.to(dtype)[:, None, :]).square().sum(dim=-1).sqrt()
    angle = 2.0 * torch.asin(torch.clamp(chord * 0.5, max=1.0))
    return angle * angle


def _normalised_weights(
    unit_vectors: Tensor, neighbours: Tensor, points: slice, sigma: float, dtype: torch.dtype
) -> Tensor:
    """Gaussian weights of each point's neighbours, normalised to sum to one."""
    weights = torch.exp(-squared_neighbour_distances(unit_vectors, neighbours, points, dtype) / (2.0 * sigma**2))
    return weights / weights.sum(dim=1, keepdim=True)


def _rows_per_band(batch: int, num_neighbours: int, channels: int, num_points: int) -> int:
    return max(1, min(num_points, _TORCH_GATHER_ELEMENTS // max(1, batch * num_neighbours * channels)))


class _TorchGaussianSmoothing(torch.autograd.Function):
    """Eager torch backend, in bands of rows to bound the memory."""

    @staticmethod
    def forward(ctx, x: Tensor, neighbours: Tensor, unit_vectors: Tensor, sigma: float) -> Tensor:
        batch, num_points, channels = x.shape
        acc_dtype = accumulation_dtype(x.dtype)
        rows = _rows_per_band(batch, neighbours.shape[1], channels, num_points)
        out = torch.empty_like(x)
        for start in range(0, num_points, rows):
            band = slice(start, start + rows)
            weights = _normalised_weights(unit_vectors, neighbours, band, sigma, acc_dtype)
            gathered = x[:, neighbours[band].long()].to(acc_dtype)  # (batch, band rows, neighbours, channels)
            out[:, band] = torch.einsum("rk,brkc->brc", weights, gathered).to(x.dtype)
        ctx.save_for_backward(neighbours, unit_vectors)
        ctx.sigma = sigma
        ctx.shape = x.shape
        ctx.dtype = x.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[Tensor, None, None, None]:
        neighbours, unit_vectors = ctx.saved_tensors
        batch, num_points, channels = ctx.shape
        acc_dtype = accumulation_dtype(ctx.dtype)
        rows = _rows_per_band(batch, neighbours.shape[1], channels, num_points)
        grad_x = torch.zeros(ctx.shape, dtype=acc_dtype, device=grad_out.device)
        for start in range(0, num_points, rows):
            band = slice(start, start + rows)
            weights = _normalised_weights(unit_vectors, neighbours, band, ctx.sigma, acc_dtype)
            sent = weights[None, :, :, None] * grad_out[:, band, None, :].to(acc_dtype)
            grad_x.index_add_(1, neighbours[band].long().reshape(-1), sent.flatten(1, 2))
        return grad_x.to(ctx.dtype), None, None, None


def _band_weights(band_neighbours: Tensor, band_vectors: Tensor, unit_vectors: Tensor, scale: Tensor) -> Tensor:
    """Unnormalised weights for a band of points; ``scale`` is ``1 / (2 * sigma**2)``."""
    return torch.exp(-_squared_angles(band_vectors, unit_vectors[band_neighbours.long()], scale.dtype) * scale)


def _smooth_band(
    x: Tensor, band_neighbours: Tensor, band_vectors: Tensor, unit_vectors: Tensor, scale: Tensor
) -> Tensor:
    """Compiled forward pass for one band of points."""
    weights = _band_weights(band_neighbours, band_vectors, unit_vectors, scale)
    gathered = x[:, band_neighbours.long()].to(scale.dtype)  # (batch, band points, neighbours, channels)
    sums = (weights[None, :, :, None] * gathered).sum(dim=2)
    return (sums / weights.sum(dim=1)[None, :, None]).to(x.dtype)


def _smooth_band_backward(
    grad_x: Tensor,
    grad_band: Tensor,
    band_neighbours: Tensor,
    band_vectors: Tensor,
    unit_vectors: Tensor,
    scale: Tensor,
) -> None:
    """Compiled backward pass for one band of points; adds into ``grad_x``."""
    weights = _band_weights(band_neighbours, band_vectors, unit_vectors, scale)
    scaled = grad_band.to(scale.dtype) / weights.sum(dim=1)[None, :, None]
    sent = weights[None, :, :, None] * scaled[:, :, None, :]
    grad_x.index_add_(1, band_neighbours.long().reshape(-1), sent.flatten(1, 2))


# Points per call of a compiled pass. A whole O1280 grid in one call fails in torch.compile
# ("XBLOCK too large"), and the band also bounds the temporary weights.
_COMPILED_BAND_POINTS = 2**19


class CompiledGaussianSmoothing:
    """Compiled forward and backward passes for the torch backend.

    The gather is fused into the weighted sum, so the gathered values are never stored. ``sigma``
    is passed as a tensor, so all scales share one compiled graph.

    Parameters
    ----------
    **compile_options
        Passed to ``torch.compile``.
    """

    def __init__(self, **compile_options) -> None:
        self.forward = torch.compile(_smooth_band, **compile_options)
        self.backward = torch.compile(_smooth_band_backward, **compile_options)


class _CompiledTorchGaussianSmoothing(torch.autograd.Function):
    """Torch backend with the passes of :class:`CompiledGaussianSmoothing`."""

    @staticmethod
    def forward(
        ctx, x: Tensor, neighbours: Tensor, unit_vectors: Tensor, sigma: float, passes: CompiledGaussianSmoothing
    ) -> Tensor:
        acc_dtype = accumulation_dtype(x.dtype)
        scale = torch.full((), 1.0 / (2.0 * sigma**2), dtype=acc_dtype, device=x.device)
        out = torch.empty_like(x)
        if x.numel() > 0:
            for start in range(0, x.shape[1], _COMPILED_BAND_POINTS):
                band = slice(start, start + _COMPILED_BAND_POINTS)
                out[:, band] = passes.forward(x, neighbours[band], unit_vectors[band], unit_vectors, scale)
        ctx.save_for_backward(neighbours, unit_vectors, scale)
        ctx.passes = passes
        ctx.dtype = x.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[Tensor, None, None, None, None]:
        neighbours, unit_vectors, scale = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_x = torch.zeros(grad_out.shape, dtype=scale.dtype, device=grad_out.device)
        if grad_x.numel() > 0:
            for start in range(0, grad_out.shape[1], _COMPILED_BAND_POINTS):
                band = slice(start, start + _COMPILED_BAND_POINTS)
                ctx.passes.backward(
                    grad_x, grad_out[:, band], neighbours[band], unit_vectors[band], unit_vectors, scale
                )
        return grad_x.to(ctx.dtype), None, None, None, None


def gaussian_smooth(
    x: Tensor,
    neighbours: Tensor,
    unit_vectors: Tensor,
    sigma: float,
    backend: ProjectionBackend = "triton",
    compiled: CompiledGaussianSmoothing | None = None,
) -> Tensor:
    """Smooth ``x`` with a Gaussian of width ``sigma`` over each point's neighbours.

    Parameters
    ----------
    x : Tensor
        Field of shape ``(..., num_points, channels)``.
    neighbours : Tensor
        Neighbours of each point, shape ``(num_points, num_neighbours)``, integer.
    unit_vectors : Tensor
        Position of each point on the unit sphere, shape ``(num_points, 3)``.
    sigma : float
        Width of the Gaussian, in radians.
    backend : str
        ``"triton"`` (GPU) or ``"torch"`` (any device).
    compiled : CompiledGaussianSmoothing, optional
        Compiled passes for the torch backend; eager if not given.

    Returns
    -------
    Tensor
        The smoothed field, with the shape and dtype of ``x``.
    """
    if backend not in PROJECTION_BACKENDS:
        raise ValueError(f"Gaussian smoothing backend must be one of {PROJECTION_BACKENDS}, got '{backend}'.")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    num_points = neighbours.shape[0]
    if x.shape[-2] != num_points or unit_vectors.shape != (num_points, 3):
        raise ValueError(
            f"Expected a field of {num_points} points and unit vectors of shape ({num_points}, 3), "
            f"got a field of shape {tuple(x.shape)} and unit vectors of shape {tuple(unit_vectors.shape)}."
        )

    shape = x.shape
    # Explicit sizes so that a field with zero channels reshapes too.
    flat = x.reshape(math.prod(shape[:-2]), shape[-2], shape[-1]).contiguous()
    if backend == "triton":
        if not flat.is_cuda:
            raise ValueError("The triton backend of Gaussian smoothing needs the field on a GPU; use backend 'torch'.")
        from anemoi.models.triton.gaussian_smoothing import gaussian_smooth_triton

        out = gaussian_smooth_triton(flat, neighbours, unit_vectors, float(sigma))
    elif compiled is not None:
        out = _CompiledTorchGaussianSmoothing.apply(flat, neighbours, unit_vectors, float(sigma), compiled)
    else:
        out = _TorchGaussianSmoothing.apply(flat, neighbours, unit_vectors, float(sigma))
    return out.reshape(shape)

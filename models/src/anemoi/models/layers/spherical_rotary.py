# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Rotary position embeddings from the positions of points on the sphere.

Attention compares queries and keys by their content. Turning each query and key by angles that depend
on where its point lies makes the score of a pair depend also on where the key lies relative to the
query, which graph transformers get from their edge features. A model component switches it on with::

    rotary_embeddings:
      max_frequency: 100
      backend: triton

It works with every attention implementation (scaled dot product, flash and neighbourhood attention)
and for self attention on one set of nodes as well as cross attention between two.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor
from torch import nn

ROTARY_BACKENDS = ("triton", "torch")


def rotary_angles(coords: Tensor, head_dim: int, max_frequency: float) -> Tensor:
    """Rotation angles of rotary position embeddings from the 3D positions of points on the sphere.

    Each point's position on the unit sphere, ``(x, y, z) = (cos lat cos lon, cos lat sin lon, sin lat)``,
    turns ``n = head_dim // 6`` channel pairs by ``w * x``, ``n`` pairs by ``w * y`` and ``n`` pairs by
    ``w * z``, with the same ``n`` frequencies ``w`` spread evenly on a log scale from 1 to
    ``max_frequency``. After rotating queries and keys, the score of a pair depends on the straight
    line from the query to the key, ``(x_q - x_k, y_q - y_k, z_q - z_k)``, at every frequency; the
    channels left over are not rotated. A frequency ``w`` repeats every ``2 pi / w`` Earth radii, about
    ``40,000 km / w``, so ``max_frequency`` 100 reaches down to offsets of a few hundred km. The
    frequencies do not depend on the grid, so a model keeps its embeddings when it is moved to another
    resolution.

    Parameters
    ----------
    coords : Tensor
        Latitude and longitude of each point in radians, shape ``(num_points, 2)``.
    head_dim : int
        Number of channels per head; at least 6.
    max_frequency : float
        Highest frequency, in radians per Earth radius; at least 1.

    Returns
    -------
    Tensor
        Angles in radians, in float64, shape ``(num_points, 3 * (head_dim // 6))``: the ``x`` angles
        first, then ``y``, then ``z``.
    """
    per_axis = head_dim // 6
    if per_axis == 0:
        raise ValueError(f"Rotary embeddings need a head dimension of at least 6, got {head_dim}.")
    if max_frequency < 1:
        raise ValueError(f"The highest rotary frequency must be at least 1, got {max_frequency}.")
    lat, lon = coords[:, 0].double(), coords[:, 1].double()
    xyz = torch.stack([torch.cos(lat) * torch.cos(lon), torch.cos(lat) * torch.sin(lon), torch.sin(lat)], dim=1)
    frequencies = torch.logspace(0.0, math.log10(max_frequency), per_axis, dtype=torch.float64)
    return (xyz[:, :, None] * frequencies[None, None, :]).flatten(1)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Turn channel ``i`` and channel ``head_dim // 2 + i`` of every point by the angle whose cosine and sine are given.

    The PyTorch version of the turn, against which the Triton kernel is tested. ``x`` has shape
    ``(..., points, head_dim)``; ``cos`` and ``sin`` have shape ``(points, n)`` for the first ``n``
    pairs; the other channels pass through unchanged. The turn is worked out in float32, or in float64
    for float64 inputs, and the result is returned in that precision.
    """
    n = cos.shape[-1]
    half = x.shape[-1] // 2
    dtype = torch.promote_types(x.dtype, torch.float32)
    x, cos, sin = x.to(dtype), cos.to(dtype), sin.to(dtype)
    first, second = x[..., :n], x[..., half : half + n]
    turned_first = first * cos - second * sin
    turned_second = first * sin + second * cos
    return torch.cat([turned_first, x[..., n:half], turned_second, x[..., half + n :]], dim=-1)


class SphericalRotaryEmbedding(nn.Module):
    """Turns queries and keys by the rotary angles of their points (see :func:`rotary_angles`).

    Holds the cosines and sines of the angles of the query points and of the key points, in the order
    in which the nodes are stored. They move with the module to the GPU but are not part of the saved
    weights; one module serves all attention layers of a model component.

    ``backend`` chooses how the turn is computed: ``"triton"`` runs one GPU kernel per tensor (see
    :mod:`anemoi.models.triton.spherical_rotary`), ``"torch"`` runs :func:`apply_rotary`, which works on
    any device and can be listed in the ``compile`` section of the model configuration.
    """

    def __init__(
        self,
        query_coords: Tensor,
        key_coords: Optional[Tensor],
        head_dim: int,
        max_frequency: float,
        backend: str = "triton",
    ) -> None:
        """Set up the tables of one model component.

        Parameters
        ----------
        query_coords : Tensor
            Latitude and longitude of the query nodes in radians, shape ``(num_queries, 2)``.
        key_coords : Tensor, optional
            Latitude and longitude of the key nodes. Leave out for self attention, where the queries
            and keys are the same nodes.
        head_dim : int
            Number of channels per head; at least 6.
        max_frequency : float
            Highest frequency in radians per Earth radius, see :func:`rotary_angles`.
        backend : str
            ``"triton"`` or ``"torch"``.
        """
        super().__init__()
        if backend not in ROTARY_BACKENDS:
            raise ValueError(f"Rotary embeddings backend must be one of {ROTARY_BACKENDS}, got '{backend}'.")
        self.backend = backend
        self.head_dim = head_dim
        self.max_frequency = float(max_frequency)
        self.shared = key_coords is None
        points = {"query": query_coords} if self.shared else {"query": query_coords, "key": key_coords}
        for name, coords in points.items():
            angles = rotary_angles(coords, head_dim, self.max_frequency)
            self.register_buffer(f"{name}_cos", torch.cos(angles).float().contiguous(), persistent=False)
            self.register_buffer(f"{name}_sin", torch.sin(angles).float().contiguous(), persistent=False)

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, max_frequency={self.max_frequency}, backend={self.backend!r}"

    def _turn(self, x: Tensor, cos: Tensor, sin: Tensor, name: str) -> Tensor:
        if x.shape[-1] != self.head_dim or x.shape[-2] != cos.shape[0]:
            raise ValueError(
                f"Rotary embeddings are set up for {cos.shape[0]} {name} points of {self.head_dim} channels, "
                f"got shape {tuple(x.shape)}."
            )
        if self.backend == "torch":
            return apply_rotary(x, cos, sin).to(x.dtype)
        from anemoi.models.triton.spherical_rotary import spherical_rotary

        return spherical_rotary(x, cos, sin)

    def forward(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        """Queries and keys of shape ``(..., points, head_dim)``, points in node order, turned and in their own dtype."""
        key_cos, key_sin = (self.query_cos, self.query_sin) if self.shared else (self.key_cos, self.key_sin)
        return self._turn(query, self.query_cos, self.query_sin, "query"), self._turn(key, key_cos, key_sin, "key")


def build_spherical_rotary(
    config: Optional[dict],
    head_dim: int,
    query_coords: Optional[Tensor],
    key_coords: Optional[Tensor] = None,
) -> Optional[SphericalRotaryEmbedding]:
    """The rotary embeddings of a model component, or None if its ``rotary_embeddings`` section is empty.

    Parameters
    ----------
    config : dict, optional
        ``max_frequency`` and optionally ``backend`` (``"triton"``, the default, or ``"torch"``).
    head_dim : int
        Number of channels per attention head.
    query_coords : Tensor, optional
        Latitude and longitude of the query nodes in radians.
    key_coords : Tensor, optional
        Latitude and longitude of the key nodes; leave out for self attention.

    Returns
    -------
    SphericalRotaryEmbedding or None
        The module shared by the attention layers of the component.
    """
    if config is None:
        return None
    unknown = set(config) - {"max_frequency", "backend"}
    if unknown:
        raise ValueError(f"Unknown rotary_embeddings settings {sorted(unknown)}; use max_frequency and backend.")
    if "max_frequency" not in config:
        raise ValueError("rotary_embeddings needs a max_frequency.")
    if query_coords is None:
        raise ValueError("Rotary embeddings need the coordinates of the graph nodes.")
    return SphericalRotaryEmbedding(
        query_coords, key_coords, head_dim, float(config["max_frequency"]), config.get("backend", "triton")
    )

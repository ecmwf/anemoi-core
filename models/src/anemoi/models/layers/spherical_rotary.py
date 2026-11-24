# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Rotary position embeddings from positions on the sphere.

Rotating queries and keys by angles that depend on their positions makes attention scores depend on
where a key lies relative to its query, the role edge features play in graph transformers. Enable
it in a model component with::

    rotary_embeddings:
      max_frequency: 10000
      backend: triton

Works with every attention implementation, in self and cross attention.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor
from torch import nn

ROTARY_BACKENDS = ("triton", "torch")


def rotary_angles(coords: Tensor, head_dim: int, max_frequency: float) -> Tensor:
    """Rotation angles of rotary embeddings from the 3D positions of points on the unit sphere.

    With ``(x, y, z) = (cos lat cos lon, cos lat sin lon, sin lat)``, ``n = head_dim // 6`` channel pairs
    turn by ``w * x``, ``n`` by ``w * y`` and ``n`` by ``w * z``, for ``n`` frequencies ``w`` spaced
    evenly on a log scale from 1 to ``max_frequency``. Scores then depend on the offset from query to
    key; leftover channels are not rotated. A frequency ``w`` repeats every ``40,000 km / w``. The
    frequencies do not depend on the grid, so the embeddings carry over to other resolutions.

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
        Angles in radians, float64, ``(num_points, 3 * (head_dim // 6))``: ``x`` angles, then ``y``, then ``z``.
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
    """Rotate channels ``i`` and ``head_dim // 2 + i`` of every point by the given angles, in PyTorch.

    ``x`` is ``(..., points, head_dim)``; ``cos`` and ``sin`` are ``(points, n)`` for the first ``n``
    pairs; other channels pass through. Computed and returned in float32, or float64 for float64 inputs.
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
    """Rotates queries and keys by the rotary angles of their points (see :func:`rotary_angles`).

    Holds cos and sin per query and key point, in node order, as non-persistent buffers. One module is
    shared by all attention layers of a component. ``backend`` is ``"triton"`` (GPU kernel) or
    ``"torch"`` (:func:`apply_rotary`, any device, can be compiled).
    """

    def __init__(
        self,
        query_coords: Tensor,
        key_coords: Optional[Tensor],
        head_dim: int,
        max_frequency: float,
        backend: str = "triton",
    ) -> None:
        """Build the tables of one model component.

        Parameters
        ----------
        query_coords : Tensor
            Latitude and longitude of the query nodes in radians, ``(num_queries, 2)``.
        key_coords : Tensor, optional
            Same for the key nodes; None for self attention.
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
        """Rotate queries and keys, ``(..., points, head_dim)`` in node order."""
        key_cos, key_sin = (self.query_cos, self.query_sin) if self.shared else (self.key_cos, self.key_sin)
        return self._turn(query, self.query_cos, self.query_sin, "query"), self._turn(key, key_cos, key_sin, "key")

    def turn_queries(self, query: Tensor, points: slice) -> Tensor:
        """Rotate the queries of a run of query points."""
        return self._turn(query, self.query_cos[points], self.query_sin[points], "query")

    def turn_keys(self, key: Tensor, points: slice | Tensor) -> Tensor:
        """Rotate the keys of a run of key points, or of the key points with indices ``points``."""
        key_cos, key_sin = (self.query_cos, self.query_sin) if self.shared else (self.key_cos, self.key_sin)
        return self._turn(key, key_cos[points], key_sin[points], "key")


def build_spherical_rotary(
    config: Optional[dict],
    head_dim: int,
    query_coords: Optional[Tensor],
    key_coords: Optional[Tensor] = None,
) -> Optional[SphericalRotaryEmbedding]:
    """Build the rotary embeddings of a model component; None if not configured.

    Parameters
    ----------
    config : dict, optional
        ``max_frequency`` and optional ``backend`` (``"triton"``, default, or ``"torch"``).
    head_dim : int
        Number of channels per attention head.
    query_coords : Tensor, optional
        Latitude and longitude of the query nodes in radians.
    key_coords : Tensor, optional
        Same for the key nodes; None for self attention.

    Returns
    -------
    SphericalRotaryEmbedding or None
        Shared by the attention layers of the component.
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

# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Node positions of octahedral reduced Gaussian grids, for tests that need points on the sphere."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class OctahedralPoints:
    """The points of the grid O``n``, row by row from north to south, each row from longitude 0 eastwards."""

    n: int

    @property
    def row_lengths(self) -> list[int]:
        north = [4 * i + 20 for i in range(self.n)]
        return north + north[::-1]

    @property
    def num_points(self) -> int:
        return sum(self.row_lengths)

    @property
    def coords(self) -> Tensor:
        """Latitude and longitude of every point in radians, float64, ``(num_points, 2)``."""
        zeros, _ = np.polynomial.legendre.leggauss(2 * self.n)
        latitudes = np.arcsin(zeros)[::-1]
        lat = [latitude for latitude, length in zip(latitudes, self.row_lengths) for _ in range(length)]
        lon = [2 * math.pi * j / length for length in self.row_lengths for j in range(length)]
        return torch.tensor([lat, lon], dtype=torch.float64).T.contiguous()


def octahedral(n: int) -> OctahedralPoints:
    return OctahedralPoints(n)

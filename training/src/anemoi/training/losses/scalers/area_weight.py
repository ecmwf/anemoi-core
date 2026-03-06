# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Area-weight scalers computed from lat/lon coordinates.

These scalers replace ``GraphNodeAttributeScaler`` for the area-weight use
case.  They source their coordinate data from the dataset/datamodule rather
than from the graph, removing a direct dependency on ``torch_geometric``.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from scipy.spatial import SphericalVoronoi

from anemoi.training.losses.scalers.base_scaler import BaseScaler
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.masks import BaseMask
from anemoi.training.utils.masks import NoOutputMask

LOGGER = logging.getLogger(__name__)


def _latlon_deg_to_cartesian(latlons_deg: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Convert lat/lon in degrees to 3-D Cartesian coordinates on a sphere.

    Parameters
    ----------
    latlons_deg : np.ndarray
        Array of shape ``(N, 2)`` with latitude and longitude in **degrees**.
    radius : float
        Radius of the sphere.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 3)`` with ``(x, y, z)`` coordinates.
    """
    latlons_rad = np.deg2rad(latlons_deg)
    lat, lon = latlons_rad[:, 0], latlons_rad[:, 1]
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return np.stack((x, y, z), axis=-1)


class AreaWeightScaler(BaseScaler):
    """Scaler that computes spherical Voronoi area weights from lat/lon coordinates.

    Unlike ``GraphNodeAttributeScaler``, this scaler does **not** depend on
    ``torch_geometric`` or on a pre-built graph.  It receives its coordinate
    data via the ``latlons`` keyword argument passed by ``create_scalers``.

    Parameters
    ----------
    latlons : np.ndarray
        Array of shape ``(N, 2)`` with latitude and longitude of each grid
        point, in **degrees**.
    output_mask : BaseMask | None
        Optional output mask applied to the scaling values.
    radius : float
        Radius of the sphere used for Voronoi tessellation (default: 1.0).
    fill_value : float
        Value assigned to degenerate Voronoi cells (default: 0.0).
    norm : str | None
        Normalisation mode (see ``BaseScaler``).
    """

    scale_dims: TensorDim = TensorDim.GRID

    def __init__(
        self,
        latlons: np.ndarray,
        output_mask: type[BaseMask] | None = None,
        radius: float = 1.0,
        fill_value: float = 0.0,
        norm: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(norm=norm)
        del kwargs
        if latlons is None:
            msg = (
                "AreaWeightScaler requires latlons from the datamodule. "
                "Ensure the datamodule exposes a latlons property for this dataset."
            )
            raise ValueError(msg)
        self.output_mask = output_mask if output_mask is not None else NoOutputMask()
        self.radius = radius
        self.fill_value = fill_value
        self._area_weights = self._compute_area_weights(latlons)

    def _compute_area_weights(self, latlons_deg: np.ndarray) -> torch.Tensor:
        """Compute spherical Voronoi cell areas from lat/lon in degrees.

        Parameters
        ----------
        latlons_deg : np.ndarray
            Coordinates of shape ``(N, 2)`` in **degrees**.

        Returns
        -------
        torch.Tensor
            1-D tensor of area weights with length ``N``.
        """
        points = _latlon_deg_to_cartesian(latlons_deg, self.radius)
        centre = np.array([0.0, 0.0, 0.0])
        sv = SphericalVoronoi(points, self.radius, centre)

        # Identify degenerate (empty) regions
        mask = np.array([bool(region) for region in sv.regions])
        sv.regions = [region for region in sv.regions if region]

        areas = sv.calculate_areas()

        null_nodes = (~mask).sum()
        if null_nodes > 0:
            LOGGER.warning(
                "AreaWeightScaler: filling %d (%.2f%%) degenerate cells with value %f",
                null_nodes,
                100 * null_nodes / len(mask),
                self.fill_value,
            )

        result = np.full(points.shape[0], self.fill_value)
        result[mask] = areas

        LOGGER.debug(
            "Computed %d area weights, total (unscaled) = %.2f",
            len(result),
            result.sum(),
        )
        return torch.from_numpy(result).float()

    def get_scaling_values(self) -> torch.Tensor:
        """Return the area weights, masked by the output mask."""
        return self.output_mask.apply(self._area_weights, dim=0, fill_value=0.0)


class ReweightedAreaWeightScaler(AreaWeightScaler):
    """Area-weight scaler with reweighting for a subset of nodes.

    Nodes identified by ``cutout_mask`` are rescaled so that their combined
    weight equals ``weight_frac_of_total`` of the total weight.

    Parameters
    ----------
    latlons : np.ndarray
        Lat/lon coordinates in degrees, shape ``(N, 2)``.
    cutout_mask : np.ndarray | torch.Tensor | None
        Boolean mask identifying the subset of nodes to reweight.
    weight_frac_of_total : float
        Target weight fraction for the masked nodes.
    output_mask : BaseMask | None
        Optional output mask.
    radius : float
        Sphere radius.
    fill_value : float
        Fill for degenerate cells.
    norm : str | None
        Normalisation mode.
    """

    def __init__(
        self,
        latlons: np.ndarray,
        cutout_mask: np.ndarray | torch.Tensor | None = None,
        weight_frac_of_total: float = 0.5,
        output_mask: type[BaseMask] | None = None,
        radius: float = 1.0,
        fill_value: float = 0.0,
        norm: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            latlons=latlons,
            output_mask=output_mask,
            radius=radius,
            fill_value=fill_value,
            norm=norm,
            **kwargs,
        )
        self.weight_frac_of_total = weight_frac_of_total
        if cutout_mask is not None:
            if isinstance(cutout_mask, np.ndarray):
                cutout_mask = torch.from_numpy(cutout_mask)
            self._scaling_mask = cutout_mask.bool()
        else:
            self._scaling_mask = None

    def _reweight(self, values: torch.Tensor) -> torch.Tensor:
        """Reweight values so that masked nodes sum to the target fraction."""
        if self._scaling_mask is None:
            return values
        mask = self._scaling_mask
        unmasked_sum = torch.sum(values[~mask])
        if mask.sum() == 0 or unmasked_sum == 0:
            return values
        weight_per_masked_node = (
            self.weight_frac_of_total / (1 - self.weight_frac_of_total) * unmasked_sum / mask.sum()
        )
        values = values.clone()
        values[mask] = weight_per_masked_node
        LOGGER.info(
            "Reweighted %d cutout nodes so their sum equals %.3f of total",
            mask.sum(),
            self.weight_frac_of_total,
        )
        return values

    def get_scaling_values(self) -> torch.Tensor:
        """Return reweighted area weights."""
        values = self.output_mask.apply(self._area_weights, dim=0, fill_value=0.0)
        return self._reweight(values)

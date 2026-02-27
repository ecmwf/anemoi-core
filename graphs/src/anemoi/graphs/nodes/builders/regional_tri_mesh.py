# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

from __future__ import annotations

import logging
import math

import numpy as np
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs import EARTH_RADIUS
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder

LOGGER = logging.getLogger(__name__)


class RegionalTriMeshNodes(BaseNodeBuilder):
    """Regional triangular-mesh nodes defined from reference data coordinates.

    The builder creates a staggered lat/lon mesh over the reference-node domain
    bounding box (optionally expanded by a margin). Consecutive latitude rows are
    offset by half a longitudinal step. Additionally, each row inserts midpoint
    nodes between neighboring row nodes, which adds the coincident triangle-height
    point inside each diamond cell. The resulting node cloud can be triangulated
    with ``anemoi.graphs.edges.TriangulationEdges``.

    Parameters
    ----------
    reference_node_name : str
        Node set used to infer the regional domain bounds.
    spacing_km : float, optional
        Approximate mesh spacing in kilometers.
    margin_km : float, optional
        Margin added around the inferred domain in kilometers.
    mask_attr_name : str | None, optional
        Optional boolean attribute on the reference nodes to restrict which
        points define the domain bounds.
    name : str
        Name of the hidden node set in the graph.
    """

    def __init__(
        self,
        reference_node_name: str,
        spacing_km: float,
        name: str,
        margin_km: float = 0.0,
        mask_attr_name: str | None = None,
    ) -> None:
        super().__init__(name)

        assert isinstance(reference_node_name, str) and reference_node_name, "reference_node_name must be a string."
        assert isinstance(spacing_km, (int, float)) and spacing_km > 0, "spacing_km must be positive."
        assert isinstance(margin_km, (int, float)) and margin_km >= 0, "margin_km must be non-negative."

        self.reference_node_name = reference_node_name
        self.spacing_km = float(spacing_km)
        self.margin_km = float(margin_km)
        self.mask_attr_name = mask_attr_name
        self._reference_coords_rad: np.ndarray | None = None

        self.hidden_attributes = BaseNodeBuilder.hidden_attributes | {
            "reference_node_name",
            "spacing_km",
            "margin_km",
            "mask_attr_name",
        }

    def register_nodes(self, graph: HeteroData) -> HeteroData:
        assert (
            self.reference_node_name in graph.node_types
        ), f'Reference node "{self.reference_node_name}" not found in graph.'

        reference_coords = graph[self.reference_node_name].x

        if self.mask_attr_name is not None:
            assert (
                self.mask_attr_name in graph[self.reference_node_name].node_attrs()
            ), f'Mask attribute "{self.mask_attr_name}" not found in reference node "{self.reference_node_name}".'
            mask = graph[self.reference_node_name][self.mask_attr_name].squeeze()
            reference_coords = reference_coords[mask]

        self._reference_coords_rad = reference_coords.detach().cpu().numpy().astype(np.float64)

        if self._reference_coords_rad.shape[0] < 3:
            raise ValueError(
                f"RegionalTriMeshNodes needs at least 3 reference points, got {self._reference_coords_rad.shape[0]}."
            )

        return super().register_nodes(graph)

    @staticmethod
    def _normalize_lon(lon_rad: np.ndarray) -> np.ndarray:
        return (lon_rad + np.pi) % (2 * np.pi) - np.pi

    def get_coordinates(self) -> torch.Tensor:
        if self._reference_coords_rad is None:
            raise RuntimeError("RegionalTriMeshNodes requires register_nodes to run before get_coordinates.")

        lat = self._reference_coords_rad[:, 0]
        lon = np.unwrap(self._reference_coords_rad[:, 1])

        lat_margin = self.margin_km / EARTH_RADIUS

        lat_min = max(-0.5 * math.pi + 1e-8, float(lat.min() - lat_margin))
        lat_max = min(0.5 * math.pi - 1e-8, float(lat.max() + lat_margin))

        lat_mid = 0.5 * (lat_min + lat_max)
        cos_lat_mid = max(abs(math.cos(lat_mid)), 1e-3)

        lon_margin = self.margin_km / (EARTH_RADIUS * cos_lat_mid)
        lon_min = float(lon.min() - lon_margin)
        lon_max = float(lon.max() + lon_margin)

        # Equilateral triangle row spacing under local planar approximation.
        lat_step = (math.sqrt(3.0) / 2.0) * (self.spacing_km / EARTH_RADIUS)
        lon_step = self.spacing_km / (EARTH_RADIUS * cos_lat_mid)

        lats = np.arange(lat_min, lat_max + 0.5 * lat_step, lat_step, dtype=np.float64)
        lons = np.arange(lon_min, lon_max + 0.5 * lon_step, lon_step, dtype=np.float64)

        if lats.size < 2 or lons.size < 2:
            raise ValueError(
                "RegionalTriMeshNodes produced fewer than 2 points in one axis. "
                f"Try lowering spacing_km (current={self.spacing_km})."
            )

        lon_min_ext = lon_min - 0.5 * lon_step
        lon_max_ext = lon_max + 0.5 * lon_step

        rows: list[np.ndarray] = []
        for row_idx, lat_value in enumerate(lats):
            row_offset = 0.5 * lon_step if row_idx % 2 else 0.0
            row_lons = np.arange(lon_min_ext + row_offset, lon_max_ext + 0.5 * lon_step, lon_step, dtype=np.float64)
            row = np.stack(
                [
                    np.full_like(row_lons, lat_value, dtype=np.float64),
                    self._normalize_lon(row_lons),
                ],
                axis=1,
            )
            rows.append(row)

            # Add midpoint nodes on each row segment to place one node at the
            # coincident height-foot position inside each diamond cell.
            if row_lons.size >= 2:
                row_mid_lons = 0.5 * (row_lons[:-1] + row_lons[1:])
                row_mid = np.stack(
                    [
                        np.full_like(row_mid_lons, lat_value, dtype=np.float64),
                        self._normalize_lon(row_mid_lons),
                    ],
                    axis=1,
                )
                rows.append(row_mid)

        coords_rad = np.concatenate(rows, axis=0)

        LOGGER.info(
            "RegionalTriMeshNodes created %d nodes from domain lat=[%.3f, %.3f] rad, lon=[%.3f, %.3f] rad "
            "with triangular spacing=%.1f km (lat_step=%.6f rad, lon_step=%.6f rad) and margin=%.1f km.",
            coords_rad.shape[0],
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            self.spacing_km,
            lat_step,
            lon_step,
            self.margin_km,
        )

        return torch.from_numpy(coords_rad).to(dtype=torch.float32)

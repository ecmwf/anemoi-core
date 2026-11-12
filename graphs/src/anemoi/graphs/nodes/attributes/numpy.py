# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.nodes.attributes.base_attributes import BaseNodeAttribute


class CutOutNumpyAttribute(BaseNodeAttribute):
    """Join regional and global static arrays onto cutout nodes by coordinates."""

    def __init__(
        self,
        regional_values: str | Path,
        regional_coordinates: str | Path,
        global_values: str | Path,
        global_coordinates: str | Path,
        mask_node_attr_name: str = "cutout_mask",
        coordinate_tolerance_degrees: float = 1e-3,
        dtype: str = "float32",
    ) -> None:
        super().__init__(norm=None, dtype=dtype)
        self.regional_values = Path(regional_values)
        self.regional_coordinates = Path(regional_coordinates)
        self.global_values = Path(global_values)
        self.global_coordinates = Path(global_coordinates)
        self.mask_node_attr_name = mask_node_attr_name
        self.coordinate_tolerance_degrees = coordinate_tolerance_degrees

    @staticmethod
    def _load_coordinates(path: Path) -> np.ndarray:
        with np.load(path) as coordinates:
            latitude = np.asarray(coordinates["latitudes"], dtype=np.float64)
            longitude = np.asarray(coordinates["longitudes"], dtype=np.float64)
        return np.deg2rad(np.column_stack((latitude, longitude)))

    @staticmethod
    def _unit_sphere(coordinates: np.ndarray) -> np.ndarray:
        latitude, longitude = coordinates[:, 0], coordinates[:, 1]
        cos_latitude = np.cos(latitude)
        return np.column_stack(
            (
                cos_latitude * np.cos(longitude),
                cos_latitude * np.sin(longitude),
                np.sin(latitude),
            )
        )

    def _match(self, target_coordinates: np.ndarray, values_path: Path, coordinates_path: Path) -> np.ndarray:
        values = np.load(values_path, mmap_mode="r")
        source_coordinates = self._load_coordinates(coordinates_path)
        if values.ndim != 2 or len(values) != len(source_coordinates):
            raise ValueError(
                f"Static values {values_path} have shape {values.shape}, but coordinates "
                f"{coordinates_path} contain {len(source_coordinates)} points."
            )

        distance, index = cKDTree(self._unit_sphere(source_coordinates)).query(
            self._unit_sphere(target_coordinates), workers=-1
        )
        angular_tolerance = np.deg2rad(self.coordinate_tolerance_degrees)
        chord_tolerance = 2.0 * np.sin(angular_tolerance / 2.0)
        if np.any(distance > chord_tolerance):
            worst = float(np.rad2deg(2.0 * np.arcsin(np.clip(distance.max() / 2.0, 0.0, 1.0))))
            raise ValueError(
                f"Could not match all cutout nodes to {coordinates_path}; worst separation is {worst:.6g} degrees."
            )
        return np.asarray(values[index], dtype=np.float32)

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        if self.mask_node_attr_name not in nodes:
            raise ValueError(f"Node attribute '{self.mask_node_attr_name}' must be built before this attribute.")

        target_coordinates = nodes.x.detach().cpu().numpy().astype(np.float64, copy=False)
        regional_mask = nodes[self.mask_node_attr_name].detach().cpu().numpy().astype(bool, copy=False).reshape(-1)
        if len(regional_mask) != len(target_coordinates):
            raise ValueError(
                f"Node attribute '{self.mask_node_attr_name}' contains {len(regional_mask)} entries, "
                f"but the graph has {len(target_coordinates)} nodes."
            )
        output: np.ndarray | None = None
        sources = (
            (regional_mask, self.regional_values, self.regional_coordinates),
            (~regional_mask, self.global_values, self.global_coordinates),
        )
        for selection, values_path, coordinates_path in sources:
            if not np.any(selection):
                continue
            matched = self._match(target_coordinates[selection], values_path, coordinates_path)
            if output is None:
                output = np.empty((len(target_coordinates), matched.shape[1]), dtype=np.float32)
            elif output.shape[1] != matched.shape[1]:
                raise ValueError("Regional and global static arrays must have the same feature dimension.")
            output[selection] = matched

        if output is None:
            raise ValueError("Cannot build a static node attribute for an empty node set.")
        return torch.from_numpy(output)

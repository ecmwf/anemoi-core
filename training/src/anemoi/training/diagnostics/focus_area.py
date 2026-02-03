from abc import ABC
from abc import abstractmethod
from typing import Any

import numpy as np


class SpatialMask(ABC):
    def __init__(self, tag: str | None) -> None:
        self.focus_mask: np.ndarray | None = None
        self.tag = "_" + tag

    @abstractmethod
    def verify_mask(self, graph_data: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def compute_mask(self, graph_data: dict[str, Any], latlons: np.ndarray) -> None:
        pass

    def apply(self, graph_data: dict[str, Any], latlons: np.ndarray, *fields: np.ndarray) -> tuple[np.ndarray, ...]:
        self.compute_mask(graph_data, latlons)

        # Slicing logic for NumPy arrays
        masked_latlons = latlons[self.focus_mask]
        masked_fields = [f[..., self.focus_mask, :] for f in fields]

        return masked_latlons, *masked_fields


class NoOpSpatialMask(SpatialMask):
    def __init__(self) -> None:
        super().__init__(tag="")

    def verify_mask(self, _graph_data: dict[str, Any]) -> None:
        pass

    def compute_mask(self, _graph_data: dict[str, Any], _latlons: np.ndarray) -> None:
        pass

    def apply(self, _graph_data: dict[str, Any], latlons: np.ndarray, *fields: np.ndarray) -> tuple[np.ndarray, ...]:
        # Prefix _graph_data to satisfy the linter
        return latlons, *fields


class NodeAttributeSpatialMask(SpatialMask):
    def __init__(self, node_attribute_name: str, name=None) -> None:
        tag = f"_{node_attribute_name}" if name is None else name
        super().__init__(tag)
        self.node_attribute_name = node_attribute_name
        self.mask_name = node_attribute_name

    def verify_mask(self, graph_data: dict[str, Any]) -> None:
        assert self.node_attribute_name in graph_data["data"], (
            f"Spatial mask '{self.node_attribute_name}' not found in graph data. "
            f"Available masks: {list(graph_data['data'].keys())}"
        )

    def compute_mask(self, graph_data: dict[str, Any], latlons: np.ndarray) -> None:
        self.verify_mask(graph_data)
        self.mask_attr_name_idxs = graph_data["data"][self.node_attribute_name]
        self.focus_mask = np.zeros(latlons.shape[0], dtype=bool)
        self.focus_mask[self.mask_attr_name_idxs.squeeze()] = True


class BoundingBoxSpatialMask(SpatialMask):
    def __init__(self, bbox: tuple[float, float, float, float], name: str = None) -> None:
        tag = f"_bbox_lat-{bbox[0]}-{bbox[2]}_lon-{bbox[1]}-{bbox[3]}" if name is None else name

        super().__init__(tag)
        self.bbox = bbox
        # Internal logical validation
        lat_min, lon_min, lat_max, lon_max = self.bbox
        assert lat_min < lat_max and lon_min < lon_max

    def verify_mask(self, _graph_data: dict[str, Any]) -> None:
        pass

    def compute_mask(self, _graph_data: dict[str, Any], latlons: np.ndarray) -> None:
        (lat_min, lon_min, lat_max, lon_max) = self.bbox
        lat, lon = latlons[:, 0], latlons[:, 1]
        self.focus_mask = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)


def build_spatial_mask(
    node_attribute_name: str | None = None,
    latlon_bbox: tuple[float, float, float, float] | None = None,
    name: str | None = None,
) -> Any:
    if node_attribute_name is not None:
        return NodeAttributeSpatialMask(node_attribute_name, name)
    if latlon_bbox is not None:
        return BoundingBoxSpatialMask(latlon_bbox, name)
    return NoOpSpatialMask()

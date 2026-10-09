# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses.scalers.base_scaler import BaseScaler
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)

# Latitude bands used for the logged summary (hard edges, degrees).
BANDS = {"NH": (20.0, 90.0), "Tropics": (-20.0, 20.0), "SH": (-90.0, -20.0)}


def _band_masks(latitudes: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "NH": latitudes > BANDS["NH"][0],
        "Tropics": (latitudes >= BANDS["Tropics"][0]) & (latitudes <= BANDS["Tropics"][1]),
        "SH": latitudes < BANDS["SH"][1],
    }


class ObsDensityScaler(BaseScaler):
    """Static per-cell, per-variable loss weights from an offline observation-density file.

    The file is built by ``obs_density.py weights`` from the training period of the dataset. For each
    variable it holds an extra multiplicative weight per grid cell (inverse observation density, raised
    to ``alpha`` and capped), normalised so that the expected loss magnitude of each variable is
    unchanged. With ``alpha=1`` the NH, Tropics and SH bands receive loss shares equal to their area
    (or cell) shares; with ``alpha=0`` the weights are 1 everywhere.

    The weights are only meaningful together with the node weights they were built for: build the file
    with ``--areas uniform`` when ``area_weight`` is ``UniformWeights``, and with ``--areas voronoi``
    for area weights.

    The scaler spans (GRID, VARIABLE), so it can only be used in training losses: validation metrics
    may not be scaled over the variable dimension.
    """

    scale_dims: tuple[TensorDim, ...] = (TensorDim.GRID, TensorDim.VARIABLE)

    def __init__(
        self,
        data_indices: IndexCollection,
        graph_data: HeteroData,
        nodes_name: str,
        weights_path: str,
        variables: list[str] | None = None,
        default_weight: float = 1.0,
        rename: dict[str, str] | None = None,
        check_coordinates: bool = True,
        coordinate_tolerance_deg: float = 1e-3,
        node_weights_attribute: str | None = "area_weight",
        norm: str | None = None,
        **kwargs,
    ) -> None:
        """Initialise ObsDensityScaler.

        Parameters
        ----------
        data_indices : IndexCollection
            Collection of data indices.
        graph_data : HeteroData
            Graph; ``graph_data[nodes_name].x`` holds the node [lat, lon] in radians.
        nodes_name : str
            Name of the data nodes in the graph.
        weights_path : str
            Path to the ``.npz`` file written by ``obs_density.py weights``.
        variables : list[str], optional
            Variables to weight; fnmatch patterns such as ``"t_*"`` are allowed. Default: every
            variable in the file.
        default_weight : float, optional
            Weight for loss variables that are not in the file or not selected, by default 1.0.
        rename : dict[str, str], optional
            Extra mapping from file variable names to training variable names.
        check_coordinates : bool, optional
            Compare the file's latitudes/longitudes with the graph nodes, by default True.
        coordinate_tolerance_deg : float, optional
            Largest allowed coordinate difference in degrees, by default 1e-3.
        node_weights_attribute : str, optional
            Graph node attribute used as the loss node weights; only used to log the band shares the
            weights reach and to check the file's area method. By default ``area_weight``.
        norm : None
            Must be None: the file already keeps each variable's expected loss magnitude, and the
            base-class norms would normalise over the whole (grid, variable) tensor.
        """
        if norm is not None:
            msg = (
                f"{self.__class__.__name__} does not support norm={norm!r}: the weights file already keeps "
                "each variable's expected loss magnitude."
            )
            raise ValueError(msg)
        super().__init__(norm=norm)
        del kwargs

        self.data_indices = data_indices
        self.weights_path = str(weights_path)
        self.patterns = list(variables) if variables is not None else None
        self.default_weight = float(default_weight)

        self._load(Path(self.weights_path), rename or {})

        nodes = graph_data[nodes_name]
        self.n_grid = int(nodes.num_nodes)
        if self.n_grid != self.file_weights.shape[1]:
            msg = (
                f"{self.__class__.__name__}: {self.weights_path} has {self.file_weights.shape[1]} grid cells, "
                f"but graph nodes '{nodes_name}' have {self.n_grid}."
            )
            raise ValueError(msg)
        if check_coordinates:
            self._check_coordinates(nodes, nodes_name, coordinate_tolerance_deg)

        self.node_weights = None
        if node_weights_attribute is not None and node_weights_attribute in nodes.node_attrs():
            self.node_weights = nodes[node_weights_attribute].squeeze().detach().cpu().double().numpy()
            self._check_area_method(node_weights_attribute)

    def _load(self, path: Path, rename: dict[str, str]) -> None:
        if not path.is_file():
            msg = f"{self.__class__.__name__}: weights file {path} does not exist."
            raise FileNotFoundError(msg)
        try:
            with np.load(path, allow_pickle=False) as npz:
                contents = {key: npz[key] for key in npz.files}
        except (OSError, ValueError) as e:
            msg = f"{self.__class__.__name__}: cannot read weights file {path}: {e}"
            raise ValueError(msg) from e

        missing = {"variables", "weights", "latitudes", "longitudes"} - contents.keys()
        if missing:
            msg = f"{self.__class__.__name__}: weights file {path} lacks {sorted(missing)}."
            raise KeyError(msg)

        names = [rename.get(str(name), str(name)) for name in contents["variables"]]
        weights = np.asarray(contents["weights"], dtype=np.float32)
        if weights.ndim != 2 or weights.shape[0] != len(names):
            msg = (
                f"{self.__class__.__name__}: weights in {path} must have shape (n_variables={len(names)}, "
                f"n_grid), got {weights.shape}."
            )
            raise ValueError(msg)
        if not np.isfinite(weights).all() or (weights < 0).any():
            msg = f"{self.__class__.__name__}: weights in {path} must be finite and non-negative."
            raise ValueError(msg)

        self.file_names = names
        self.file_index = {name: i for i, name in enumerate(names)}
        self.file_weights = weights
        self.file_valid_fraction = contents.get("valid_fraction")
        self.file_latitudes = np.asarray(contents["latitudes"], dtype=np.float64)
        self.file_longitudes = np.asarray(contents["longitudes"], dtype=np.float64)
        self.file_meta = {
            key: contents[key].item()
            for key in ("mode", "alpha", "max_weight", "areas_method", "first_date", "last_date", "n_dates")
            if key in contents
        }

    def _check_coordinates(self, nodes: HeteroData, nodes_name: str, tolerance_deg: float) -> None:
        coords = getattr(nodes, "x", None)
        if coords is None or coords.ndim != 2 or coords.shape[1] != 2:
            LOGGER.warning(
                "%s: graph nodes '%s' have no (n, 2) [lat, lon] attribute 'x'; skipping the coordinate check.",
                self.__class__.__name__,
                nodes_name,
            )
            return
        latlon = np.rad2deg(coords.detach().cpu().double().numpy())
        dlat = np.abs(latlon[:, 0] - self.file_latitudes)
        dlon = np.abs((latlon[:, 1] - self.file_longitudes + 180.0) % 360.0 - 180.0)
        worst = float(max(dlat.max(), dlon.max()))
        if worst > tolerance_deg:
            first = int(np.argmax(np.maximum(dlat, dlon) > tolerance_deg))
            msg = (
                f"{self.__class__.__name__}: grid of {self.weights_path} does not match graph nodes '{nodes_name}' "
                f"(max difference {worst:.4g} deg > {tolerance_deg:g}; first at cell {first}: file "
                f"({self.file_latitudes[first]:.4f}, {self.file_longitudes[first]:.4f}), graph "
                f"({latlon[first, 0]:.4f}, {latlon[first, 1]:.4f})). Different dataset, grid, cutout or ordering?"
            )
            raise ValueError(msg)

    def _check_area_method(self, attribute: str) -> None:
        method = self.file_meta.get("areas_method")
        if method is None:
            return
        uniform_nodes = bool(np.ptp(self.node_weights) <= 1e-6 * np.abs(self.node_weights).max())
        if (method == "uniform") != uniform_nodes:
            LOGGER.warning(
                "%s: %s was built with areas=%s, but the graph node attribute '%s' is %s. The band shares "
                "reached will differ from the file's targets; rebuild with --areas %s.",
                self.__class__.__name__,
                self.weights_path,
                method,
                attribute,
                "uniform" if uniform_nodes else "not uniform",
                "uniform" if uniform_nodes else "voronoi",
            )

    def _selected(self, name: str) -> bool:
        if self.patterns is None:
            return True
        return any(fnmatchcase(name, pattern) for pattern in self.patterns)

    def get_scaling_values(self, **_kwargs) -> torch.Tensor:
        """Get the (grid, variable) weights in the loss-tensor (data.output) layout."""
        output_full = self.data_indices.data.output.full.tolist()
        index_to_name = {idx: name for name, idx in self.data_indices.data.output.name_to_index.items()}
        values = np.full((self.n_grid, len(output_full)), self.default_weight, dtype=np.float32)

        weighted, not_in_file = [], []
        for position, raw_idx in enumerate(output_full):
            name = index_to_name[raw_idx]
            if not self._selected(name):
                continue
            if name not in self.file_index:
                not_in_file.append(name)
                continue
            values[:, position] = self.file_weights[self.file_index[name]]
            weighted.append(name)

        self._log_summary(weighted, not_in_file)
        return torch.from_numpy(values)

    def _log_summary(self, weighted: list[str], not_in_file: list[str]) -> None:
        meta = ", ".join(f"{key}={value}" for key, value in self.file_meta.items())
        LOGGER.info(
            "%s: %s (%s); %d loss variables weighted, the rest use default_weight=%g.",
            self.__class__.__name__,
            self.weights_path,
            meta,
            len(weighted),
            self.default_weight,
        )
        LOGGER.info("%s: weighted variables: %s", self.__class__.__name__, weighted)
        if self.patterns is not None:
            unmatched = [p for p in self.patterns if not any(fnmatchcase(n, p) for n in weighted + not_in_file)]
            if unmatched:
                LOGGER.warning("%s: patterns matching no loss variable: %s", self.__class__.__name__, unmatched)
        if not_in_file:
            LOGGER.warning(
                "%s: selected loss variables missing from the weights file (use default_weight): %s",
                self.__class__.__name__,
                not_in_file,
            )
        if not weighted or self.file_valid_fraction is None:
            return

        # Band shares of the expected loss mass f * node_weight * w, before and after weighting.
        node_weights = self.node_weights if self.node_weights is not None else np.ones(self.n_grid)
        masks = _band_masks(self.file_latitudes)
        LOGGER.info(
            "%s: expected loss share NH/Tropics/SH before -> after, and band-mean weight:",
            self.__class__.__name__,
        )
        for name in weighted:
            row = self.file_index[name]
            mass = self.file_valid_fraction[row].astype(np.float64) * node_weights
            w = self.file_weights[row].astype(np.float64)
            if mass.sum() <= 0:
                continue
            before = [mass[m].sum() / mass.sum() for m in masks.values()]
            after = [(mass * w)[m].sum() / (mass * w).sum() for m in masks.values()]
            band_w = [(mass * w)[m].sum() / max(mass[m].sum(), 1e-30) for m in masks.values()]
            LOGGER.info(
                "  %-12s %s -> %s  w %s",
                name,
                "/".join(f"{x:.2f}" for x in before),
                "/".join(f"{x:.2f}" for x in after),
                "/".join(f"{x:.2f}" for x in band_w),
            )

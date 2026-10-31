# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Reduced diagnostics of the learned semi-Lagrangian transport inside an ADR processor.

``GridSampler.diag_hook`` and ``KNNSampler.diag_hook`` are called once per forward pass with
the departure points of every head, in radians. :class:`NSLRecorder` reduces them to a handful of
maps per layer, so a rollout costs megabytes rather than gigabytes. It records both the hidden
latitude-longitude grid of a ``RegularLatLonGeometry``, where a map has shape (nlat, nlon), and
the hidden mesh of a ``MeshGeometry``, where it has one entry per node. ``nsl_plots`` draws
latitude-longitude grids only, so :meth:`NSLRecorder.for_plotting` puts the maps of a mesh on a
regular grid at the resolution of the mesh (:func:`on_plotting_grid`); the trajectory-folding
map, which compares neighbouring grid cells, exists on the grid only.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from dataclasses import field

import numpy as np
import torch

from anemoi.graphs import EARTH_RADIUS  # km
from anemoi.models.layers.processor import EARTH_ROTATION_RATE
from anemoi.models.layers.semi_lagrangian import GridSampler
from anemoi.models.layers.semi_lagrangian import KNNSampler

# The names under which the owners of a sampler hold the length of a layer's sub-step and the
# per-channel advection gate: ``SemiLagrangianAdvection.time_step`` and ``Advection.share`` now,
# ``advection_share`` on the block in older checkpoints.
TIME_STEP_ATTRIBUTES = ("time_step",)
GATE_ATTRIBUTES = ("share", "advection_share")


# ---------------------------------------------------------------- geometry
def haversine(lat1, lon1, lat2, lon2):
    """Great-circle angle (radians) between points given in radians."""
    a = torch.sin((lat2 - lat1) / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin((lon2 - lon1) / 2) ** 2
    return 2 * torch.asin(a.clamp(0, 1).sqrt())


def displacement_east_north(lat_a, lon_a, lat_d, lon_d):
    """Motion vector (departure -> arrival) in km, in the local east/north frame at the arrival point."""
    dist = haversine(lat_a, lon_a, lat_d, lon_d) * EARTH_RADIUS
    dlon = lon_d - lon_a
    bearing = torch.atan2(  # initial bearing arrival -> departure
        torch.sin(dlon) * torch.cos(lat_d),
        torch.cos(lat_a) * torch.sin(lat_d) - torch.sin(lat_a) * torch.cos(lat_d) * torch.cos(dlon),
    )
    return -dist * torch.sin(bearing), -dist * torch.cos(bearing), dist


def latlon_to_cartesian(lat, lon):
    return torch.stack([torch.cos(lat) * torch.cos(lon), torch.cos(lat) * torch.sin(lon), torch.sin(lat)], dim=-1)


def _signed_cell_area(p):
    """Signed area proxy of each grid cell on the sphere; p: (..., H, W, 3). Periodic in lon."""
    dy = p[..., 1:, :, :] - p[..., :-1, :, :]
    dx = (torch.roll(p, -1, dims=-2) - p)[..., :-1, :, :]
    return (torch.cross(dx, dy, dim=-1) * p[..., :-1, :, :]).sum(-1)


def fold_ratio(lat_a, lon_a, lat_d, lon_d):
    """Departure-cell area / arrival-cell area. <= 0 means neighbouring trajectories crossed.

    The first and last rows are dropped, so the output is (..., H-2, W).
    """
    ratio = _signed_cell_area(latlon_to_cartesian(lat_d, lon_d)) / _signed_cell_area(latlon_to_cartesian(lat_a, lon_a))
    return ratio[..., 1:, :]


@dataclass
class LayerDiag:
    step: int
    layer: int
    dt_seconds: float | None  # length of this layer's sub-step, None if the layer is untimed
    # The maps below have the shape of the hidden nodes: (nlat, nlon) on a grid, (nodes,) on a mesh.
    east_km: np.ndarray  # head-mean motion vector, east component
    north_km: np.ndarray
    dist_km: np.ndarray  # head-mean displacement length
    dist_per_mode: np.ndarray  # (heads,) area-weighted mean displacement of each head
    alpha: np.ndarray | None  # (C,) advection blend weights in [0, 1], None if the block has no gate
    fold_fraction: np.ndarray | None  # (nlat-2, nlon) fraction of heads whose mapping folds, grid only
    probe_departures: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (heads, 2) lat/lon deg


class NSLRecorder:
    """Context manager collecting reduced NSL diagnostics during forward passes.

    Example
    -------
    >>> with NSLRecorder(model, probes={"natl": (45.0, -40.0)}) as rec:  # doctest: +SKIP
    ...     for state in runner.run(input_states=input_state, lead_time=12):
    ...         pass
    >>> rec.diags  # one LayerDiag per (step, layer)  # doctest: +SKIP
    """

    def __init__(self, model, probes=None, fold_check=True, chunk=64, batch_index=0):
        self.modules = [m for m in model.modules() if isinstance(m, (GridSampler, KNNSampler))]
        if not self.modules:
            raise ValueError(
                "No GridSampler or KNNSampler modules found in the model. This checkpoint does not "
                "use an ADR/semi-Lagrangian processor, so there is nothing to record.",
            )
        # On a mesh the nodes are not laid out in rows and columns, so neighbouring trajectories
        # cannot be compared cell by cell.
        self.on_grid = isinstance(self.modules[0], GridSampler)
        # The sampler only moves values. The length of the sub-step it moves them over and the
        # per-channel advection gate live on its owners: ``SemiLagrangianAdvection.time_step``
        # and, one level up, ``Advection.share``.
        owners = {id(child): parent for parent in model.modules() for child in parent.children()}
        self._time_steps = [_from_owners(m, owners, TIME_STEP_ATTRIBUTES) for m in self.modules]
        self._gates = [_from_owners(m, owners, GATE_ATTRIBUTES) for m in self.modules]

        self.probes = probes or {}
        self.fold_check = fold_check and self.on_grid
        self.chunk = chunk
        self.b = batch_index
        self.diags: list[LayerDiag] = []
        self._step = -1

        # The samplers read the nodes from the layout they are given, so take them from the one
        # geometry the processor shares with its blocks.
        geometry = next(g for g in model.modules() if hasattr(g, "lat") and hasattr(g, "sampler"))
        if self.on_grid:
            self.nlat, self.nlon = geometry.nlat, geometry.nlon
            self.lat_deg = np.rad2deg(geometry.lat[:, 0].detach().cpu().numpy())  # descending, no pole
            self.lon_deg = np.rad2deg(geometry.lon[0, :].detach().cpu().numpy()) % 360
            self.spacing_deg = 180.0 / self.nlat
        else:
            # One entry per hidden node, in graph order, rather than the rows and columns of a grid.
            self.nlat = self.nlon = None
            self.lat_deg = np.rad2deg(geometry.lat.detach().cpu().numpy())
            self.lon_deg = np.rad2deg(geometry.lon.detach().cpu().numpy()) % 360
            # Typical distance between neighbouring nodes, which sets the plotting grid.
            chord = float(geometry.spacing.detach().median())
            self.spacing_deg = float(np.degrees(2 * np.arcsin(chord / 2)))
        self._probe_idx = {name: self._nearest(lat, lon) for name, (lat, lon) in self.probes.items()}

    def _nearest(self, lat, lon):
        dlon = np.deg2rad((self.lon_deg - lon % 360 + 180) % 360 - 180)
        if self.on_grid:
            i = int(np.argmin(np.abs(self.lat_deg - lat)))
            return i, int(np.argmin(np.abs(dlon)))
        # The nodes are anywhere on the sphere, so take the one at the smallest great-circle angle.
        lat_rad, node_lat = np.deg2rad(lat), np.deg2rad(self.lat_deg)
        cosine = np.sin(lat_rad) * np.sin(node_lat) + np.cos(lat_rad) * np.cos(node_lat) * np.cos(dlon)
        return (int(np.argmax(cosine)),)

    def probe_index(self, name):
        """Where a probe sits on the hidden nodes: row and column on a grid, node index on a mesh."""
        index = self._probe_idx[name]
        return index if self.on_grid else index[0]

    @property
    def num_nodes(self):
        return self.lat_deg.size

    def for_plotting(self, num_neighbours=4):
        """The diagnostics as ``nsl_plots`` draws them: on a latitude-longitude grid.

        On the hidden grid they are returned as recorded. On a mesh every map is put on a regular
        grid at the resolution of the mesh (see :func:`on_plotting_grid`). Returns the diagnostics
        and the grid's latitudes (descending, no pole) and longitudes in degrees.
        """
        if self.on_grid:
            return self.diags, self.lat_deg, self.lon_deg
        return on_plotting_grid(self.diags, self.lat_deg, self.lon_deg, self.spacing_deg, num_neighbours)

    def __enter__(self):
        for layer, m in enumerate(self.modules):
            m.diag_hook = functools.partial(self._capture, layer)
        return self

    def __exit__(self, *exc):
        for m in self.modules:
            m.diag_hook = None

    @torch.no_grad()
    def _capture(self, layer, module, layout, lat_d, lon_d):
        """Hook called by ``GridSampler.sample`` and ``KNNSampler.sample``.

        lat_d, lon_d hold the departure points in radians, with shape
        (batch * local heads, nlat, nlon) on a grid and (batch * local heads, nodes) on a mesh.
        """
        if layer == 0:
            self._step += 1

        heads = module.num_heads
        if lat_d.shape[0] % heads:
            raise RuntimeError(
                f"Expected a whole number of batches of {heads} heads, got {lat_d.shape[0]} slices. "
                "Sharding the model across GPUs splits the heads; record on a single device.",
            )
        sl = slice(self.b * heads, (self.b + 1) * heads)
        lat_d, lon_d = lat_d[sl].float(), lon_d[sl].float()

        # The arrival points have the shape of the hidden nodes, which the departure points carry
        # after the head dimension: (nlat, nlon) on a grid, (nodes,) on a mesh.
        lat_a = layout.lat.float()
        lon_a = layout.lon.float()
        C = lat_d.shape[0]
        # Area weight of each point: a grid cell narrows towards the poles; a node of a mesh
        # stands for an area that grows with the distance to its neighbours.
        w = torch.cos(lat_a).clamp_min(0) if self.on_grid else layout.spacing.float().square()
        w = w / w.sum()

        e_sum = torch.zeros_like(lat_a)
        n_sum, d_sum = torch.zeros_like(e_sum), torch.zeros_like(e_sum)
        fold_cnt = torch.zeros(self.nlat - 2, self.nlon, device=lat_d.device) if self.fold_check else None
        per_mode = []

        for c0 in range(0, C, self.chunk):  # chunk over heads to bound memory
            ld, od = lat_d[c0 : c0 + self.chunk], lon_d[c0 : c0 + self.chunk]
            e, n, d = displacement_east_north(lat_a, lon_a, ld, od)
            e_sum += e.sum(0)
            n_sum += n.sum(0)
            d_sum += d.sum(0)
            per_mode.append((d * w).flatten(1).sum(-1))
            if self.fold_check:
                fold_cnt += (fold_ratio(lat_a, lon_a, ld, od) <= 0).sum(0)

        # `time_step` is in units of 1 / EARTH_ROTATION_RATE and covers one layer's sub-step,
        # not the whole model step.
        time_step = self._time_steps[layer]
        gate = self._gates[layer]

        to_np = lambda t: t.cpu().numpy()  # noqa: E731
        diag = LayerDiag(
            step=self._step,
            layer=layer,
            dt_seconds=None if time_step is None else float(time_step) / EARTH_ROTATION_RATE,
            east_km=to_np(e_sum / C),
            north_km=to_np(n_sum / C),
            dist_km=to_np(d_sum / C),
            dist_per_mode=to_np(torch.cat(per_mode)),
            alpha=None if gate is None else to_np(torch.sigmoid(gate.detach().float())),
            fold_fraction=to_np(fold_cnt / C) if self.fold_check else None,
        )
        for name, index in self._probe_idx.items():
            point = (slice(None), *index)
            diag.probe_departures[name] = np.rad2deg(
                torch.stack([lat_d[point], lon_d[point]], dim=-1).cpu().numpy(),
            )
        self.diags.append(diag)


def _from_owners(module, owners, names):
    """The first of ``names`` found on the module or one of its ancestors, or None."""
    while module is not None:
        for name in names:
            if hasattr(module, name):
                return getattr(module, name)
        module = owners.get(id(module))
    return None


# ---------------------------------------------------------------- mesh -> grid
def plotting_grid(spacing_deg):
    """Latitudes (descending, no pole) and longitudes in degrees of a regular grid of about that spacing."""
    nlat = max(2, int(round(180.0 / spacing_deg)))
    step = 180.0 / nlat
    return 90.0 - step * (np.arange(nlat) + 0.5), step * np.arange(2 * nlat)


def nearest_grid_index(lat, lon, lat_deg, lon_deg):
    """Row and column of the grid point nearest to a point given in degrees."""
    dlon = (np.asarray(lon_deg) - lon % 360 + 180) % 360 - 180
    return int(np.argmin(np.abs(np.asarray(lat_deg) - lat))), int(np.argmin(np.abs(dlon)))


def _unit_sphere(lat_deg, lon_deg):
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def on_plotting_grid(diags, node_lat_deg, node_lon_deg, spacing_deg, num_neighbours=4):
    """Diagnostics recorded on a mesh, put on a regular latitude-longitude grid.

    Every map with one entry per node (the head-mean motion vector and displacement length) is
    read at each grid point as the inverse-distance weighted mean of the ``num_neighbours``
    nearest nodes, found on the unit sphere. The grid has about the spacing of the mesh, so
    nothing is invented between the nodes and little is lost. The east and north components are
    those of the local frame at each node, which neighbouring nodes share closely enough away
    from the poles. Per-head statistics, gates and probe clouds are kept as they are; the folding
    map has no meaning here and stays None.

    Returns the gridded diagnostics and the grid's latitudes and longitudes in degrees.
    """
    from scipy.spatial import cKDTree

    lat_deg, lon_deg = plotting_grid(spacing_deg)
    grid_lat, grid_lon = np.meshgrid(lat_deg, lon_deg, indexing="ij")
    distance, index = cKDTree(_unit_sphere(node_lat_deg, node_lon_deg)).query(
        _unit_sphere(grid_lat.ravel(), grid_lon.ravel()), k=num_neighbours
    )
    distance, index = np.atleast_2d(distance.T).T, np.atleast_2d(index.T).T
    weight = 1.0 / np.maximum(distance, 1e-6)
    weight /= weight.sum(axis=1, keepdims=True)

    def regrid(values):
        return (np.asarray(values)[index] * weight).sum(axis=1).reshape(lat_deg.size, lon_deg.size)

    gridded = [
        LayerDiag(
            step=d.step,
            layer=d.layer,
            dt_seconds=d.dt_seconds,
            east_km=regrid(d.east_km),
            north_km=regrid(d.north_km),
            dist_km=regrid(d.dist_km),
            dist_per_mode=d.dist_per_mode,
            alpha=d.alpha,
            fold_fraction=None,
            probe_departures=dict(d.probe_departures),
        )
        for d in diags
    ]
    return gridded, lat_deg, lon_deg


def column_mean_wind(u, v, levels_hpa):
    """Pressure-thickness-weighted vertical mean of (L, ...) wind fields."""
    p = np.asarray(levels_hpa, dtype=float)
    order = np.argsort(p)
    dp = np.gradient(p[order])
    wts = (dp / dp.sum()).reshape(-1, *([1] * (u.ndim - 1)))
    return (wts * u[order]).sum(0), (wts * v[order]).sum(0)


def regrid_to_hidden(values, lat_deg, lon_deg, hidden_lat_deg, hidden_lon_deg):
    """Nearest-neighbour map of fields on an arbitrary grid onto the hidden latitude-longitude grid.

    ``values`` has shape (..., npoints) on the grid given by ``lat_deg``/``lon_deg`` (one entry per
    point, so reduced Gaussian grids such as O96 work too). Returns (..., nlat, nlon). Neighbours
    are found on the unit sphere, which keeps the meridian and the poles well behaved.
    """
    from scipy.spatial import cKDTree

    def unit_sphere(lat, lon):
        lat, lon = np.deg2rad(lat), np.deg2rad(lon)
        return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)

    tree = cKDTree(unit_sphere(np.asarray(lat_deg), np.asarray(lon_deg)))
    grid_lat, grid_lon = np.meshgrid(hidden_lat_deg, hidden_lon_deg, indexing="ij")
    _, idx = tree.query(unit_sphere(grid_lat.ravel(), grid_lon.ravel()))
    return np.asarray(values)[..., idx].reshape(*np.shape(values)[:-1], len(hidden_lat_deg), len(hidden_lon_deg))

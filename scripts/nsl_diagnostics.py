# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Reduced diagnostics of the learned semi-Lagrangian transport inside an ADR processor.

``SemiLagrangianLayer.diag_hook`` is called once per forward pass with the departure points of
every head, in ``grid_sample``'s coordinates of the padded globe. :class:`NSLRecorder` turns those
back into latitudes and longitudes and reduces them to a handful of maps per layer, so a rollout
costs megabytes rather than gigabytes. The hidden nodes are always a regular latitude-longitude
grid here, so every map has shape (nlat, nlon).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from dataclasses import field

import numpy as np
import torch

from anemoi.graphs import EARTH_RADIUS  # km
from anemoi.models.layers.semi_lagrangian import EARTH_ROTATION_RATE
from anemoi.models.layers.semi_lagrangian import SemiLagrangianLayer
from anemoi.models.layers.semi_lagrangian import latlon_cell_centres


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


def departure_latlon(grid_x, grid_y, nlat, nlon, pad):
    """Undo ``SemiLagrangianLayer.departure_points``: sampling coordinates -> radians.

    ``grid_x``/``grid_y`` are the column and row coordinates handed to ``grid_sample``, in [-1, 1]
    over the globe padded by ``pad`` points on every side.
    """
    col = (grid_x + 1) * (nlon + 2 * pad - 1) / 2 - pad
    row = (grid_y + 1) * (nlat + 2 * pad - 1) / 2 - pad
    lon = col * (2 * torch.pi / nlon)
    lat = torch.pi / 2 - (row + 0.5) * (torch.pi / nlat)
    return lat, lon


# ---------------------------------------------------------------- records
@dataclass
class LayerDiag:
    step: int
    layer: int
    dt_seconds: float | None  # length of this layer's sub-step, None if the layer is untimed
    # The maps below have the shape of the hidden grid: (nlat, nlon).
    east_km: np.ndarray  # head-mean motion vector, east component
    north_km: np.ndarray
    dist_km: np.ndarray  # head-mean displacement length
    dist_per_mode: np.ndarray  # (heads,) area-weighted mean displacement of each head
    alpha: np.ndarray | None  # (C,) advection blend weights in [0, 1], None if the block has no gate
    fold_fraction: np.ndarray | None  # (nlat-2, nlon) fraction of heads whose mapping folds
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
        self.modules = [m for m in model.modules() if isinstance(m, SemiLagrangianLayer)]
        if not self.modules:
            raise ValueError(
                "No SemiLagrangianLayer modules found in the model. This checkpoint does not "
                "use an ADR/semi-Lagrangian processor, so there is nothing to record.",
            )
        # `advection_share` is the per-channel advection gate and lives on the owning
        # ADRProcessorBlock, not on the semi-Lagrangian layer itself.
        owners = {id(child): parent for parent in model.modules() for child in parent.children()}
        self._gates = [getattr(owners.get(id(m)), "advection_share", None) for m in self.modules]

        self.probes = probes or {}
        self.fold_check = fold_check
        self.chunk = chunk
        self.b = batch_index
        self.diags: list[LayerDiag] = []
        self._step = -1

        # Every layer builds its departure points on the same hidden grid, which each one carries
        # itself; the layout is rebuilt per forward pass and is not passed to the hook.
        self.nlat, self.nlon = self.modules[0].nlat, self.modules[0].nlon
        shapes = {(m.nlat, m.nlon) for m in self.modules}
        if len(shapes) > 1:
            raise ValueError(f"The semi-Lagrangian layers use different hidden grids: {sorted(shapes)}.")
        lat, lon = latlon_cell_centres(self.nlat, self.nlon)
        self._lat_a = lat.view(self.nlat, self.nlon).float()
        self._lon_a = lon.view(self.nlat, self.nlon).float()
        self.lat_deg = np.rad2deg(self._lat_a[:, 0].numpy())  # descending, no pole
        self.lon_deg = np.rad2deg(self._lon_a[0, :].numpy()) % 360
        self._probe_idx = {name: self._nearest(lat, lon) for name, (lat, lon) in self.probes.items()}

    def _nearest(self, lat, lon):
        dlon = np.deg2rad((self.lon_deg - lon % 360 + 180) % 360 - 180)
        i = int(np.argmin(np.abs(self.lat_deg - lat)))
        return i, int(np.argmin(np.abs(dlon)))

    def probe_index(self, name):
        """Row and column of a probe on the hidden grid."""
        return self._probe_idx[name]

    def __enter__(self):
        for layer, m in enumerate(self.modules):
            m.diag_hook = functools.partial(self._capture, layer)
        return self

    def __exit__(self, *exc):
        for m in self.modules:
            m.diag_hook = None

    @torch.no_grad()
    def _capture(self, layer, module, grid_x, grid_y):
        """Hook called by ``SemiLagrangianLayer.sample``.

        ``grid_x``/``grid_y`` hold the departure points in ``grid_sample``'s coordinates of the
        padded globe, with shape (batch * local heads, nlat, nlon).
        """
        if layer == 0:
            self._step += 1

        heads = module.num_heads
        if grid_x.shape[0] % heads:
            raise RuntimeError(
                f"Expected a whole number of batches of {heads} heads, got {grid_x.shape[0]} slices. "
                "Sharding the model across GPUs splits the heads; record on a single device.",
            )
        sl = slice(self.b * heads, (self.b + 1) * heads)
        lat_d, lon_d = departure_latlon(
            grid_x[sl].float(),
            grid_y[sl].float(),
            module.nlat,
            module.nlon,
            module.pad,
        )

        # The arrival points have the shape of the hidden grid, which the departure points carry
        # after the head dimension.
        lat_a = self._lat_a.to(lat_d.device)
        lon_a = self._lon_a.to(lat_d.device)
        C = lat_d.shape[0]
        w = torch.cos(lat_a).clamp_min(0)
        w = w / w.sum()  # area weight of each point

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
        # not the whole model step. A warp layer applies its displacements directly and has none.
        time_step = getattr(module, "time_step", None)
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
        for name, (i, j) in self._probe_idx.items():
            diag.probe_departures[name] = np.rad2deg(
                torch.stack([lat_d[:, i, j], lon_d[:, i, j]], dim=-1).cpu().numpy(),
            )
        self.diags.append(diag)


# ---------------------------------------------------------------- reference wind
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

# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Plots of the reduced NSL diagnostics collected by :mod:`nsl_diagnostics`.

Displacements are recorded per ADR layer, and one layer covers a sub-step of the model step
(``model timestep / num_layers``), so every conversion to a speed uses the layer's own
``dt_seconds`` rather than a single global time step.
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from cartopy.geodesic import Geodesic
from matplotlib.animation import FuncAnimation
from scipy.interpolate import RegularGridInterpolator

PC, GEO = ccrs.PlateCarree(), ccrs.Geodetic()


def _ms(km, dt_seconds):
    """Displacement over one layer sub-step (km) -> equivalent speed (m/s)."""
    return km * 1e3 / dt_seconds


def _select(diags, step):
    return sorted((d for d in diags if d.step == step), key=lambda d: d.layer)


def _dt(diags):
    """The sub-step shared by a set of layers, for annotations that are not per-layer."""
    dts = {d.dt_seconds for d in diags if d.dt_seconds is not None}
    if len(dts) != 1:
        raise ValueError(f"Expected one sub-step across the selected layers, got {dts or 'none'}.")
    return dts.pop()


def vector_corr(u1, v1, u2, v2, lat_deg):
    w = np.cos(np.deg2rad(lat_deg))[:, None]
    num = (w * (u1 * u2 + v1 * v2)).sum()
    return num / np.sqrt((w * (u1**2 + v1**2)).sum() * (w * (u2**2 + v2**2)).sum())


# 1. Latent wind maps -----------------------------------------------------------
def plot_latent_winds(diags, lat_deg, lon_deg, era5_uv=None, layers=(0, 3, 7), step=0, path=None):
    panels = [
        (f"Layer {d.layer + 1}: head-mean latent wind", _ms(d.east_km, d.dt_seconds), _ms(d.north_km, d.dt_seconds))
        for d in _select(diags, step)
        if d.layer in layers
    ]

    fig, axes = plt.subplots(
        len(panels) // 2,
        2,
        figsize=(10 * 2, 4.3 * len(panels) / 2),
        subplot_kw={"projection": ccrs.Robinson()},
    )
    for ax, (title, u, v) in zip(axes.ravel(), panels):
        spd = np.hypot(u, v)
        mesh = ax.pcolormesh(
            lon_deg,
            lat_deg,
            spd,
            transform=PC,
            cmap="viridis",
            shading="auto",
            vmin=0,
            vmax=np.nanpercentile(spd, 99),
        )
        ax.streamplot(
            lon_deg,
            lat_deg[::-1],
            u[::-1],
            v[::-1],
            transform=PC,
            density=1.6,
            color="white",
            linewidth=0.5,
            arrowsize=0.6,
        )
        ax.coastlines(lw=0.4, color="0.85")

        if era5_uv is not None and title.startswith("Layer"):
            corr_with_physical_winds = vector_corr(u, v, *era5_uv, lat_deg)
            title += f"   (vector corr. with reference = {corr_with_physical_winds:.2f})"

        ax.set_title(title, fontsize=10)
        fig.colorbar(mesh, ax=ax, shrink=0.7, label="m/s")
    _finish(fig, path)


def plot_physical_winds(lat_deg, lon_deg, era5_uv=None, step=0, path=None):
    fig, ax = plt.subplots(figsize=(10, 4.3), subplot_kw={"projection": ccrs.Robinson()})
    if era5_uv is not None:
        u, v = era5_uv
        spd = np.hypot(u, v)
        mesh = ax.pcolormesh(
            lon_deg,
            lat_deg,
            spd,
            transform=PC,
            cmap="viridis",
            shading="auto",
            vmin=0,
            vmax=np.nanpercentile(spd, 99),
        )
        ax.streamplot(
            lon_deg,
            lat_deg[::-1],
            u[::-1],
            v[::-1],
            transform=PC,
            density=1.6,
            color="white",
            linewidth=0.5,
            arrowsize=0.6,
        )
        ax.coastlines(lw=0.4, color="0.85")
        fig.colorbar(mesh, ax=ax, shrink=0.7, label="m/s")
    _finish(fig, path)


# 2. Departure-point cloud -----------------------------------------------------
def plot_departure_cloud(diags, probe, probe_latlon, ubar=None, wave_speed=50.0, step=0, path=None):
    lat0, lon0 = probe_latlon
    sel = _select(diags, step)
    dt = _dt(sel)
    cmap = plt.get_cmap("plasma", len(sel))
    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"projection": ccrs.Orthographic(lon0, lat0)})
    for d in sel:
        pts = d.probe_departures[probe]
        ax.scatter(
            pts[:, 1],
            pts[:, 0],
            s=3,
            alpha=0.5,
            color=cmap(d.layer),
            transform=PC,
            label=f"layer {d.layer + 1}",
        )
    ax.plot(lon0, lat0, marker="*", ms=15, mec="k", mfc="gold", transform=PC, label="arrival point")

    geod, center = Geodesic(), (lon0, lat0)
    if ubar is not None:  # expected centre: back-trajectory along the reference column-mean wind
        az = np.degrees(np.arctan2(-ubar[0], -ubar[1]))
        res = np.asarray(geod.direct([lon0, lat0], az, np.hypot(*ubar) * dt))
        center = (res[0, 0], res[0, 1])
        ax.plot(*center, "kx", ms=10, transform=PC, label="reference back-trajectory (−Δt · ū)")
    ring = np.asarray(geod.circle(center[0], center[1], wave_speed * dt, n_samples=180))
    ax.plot(
        ring[:, 0],
        ring[:, 1],
        "k--",
        lw=1,
        transform=GEO,
        label=f"{wave_speed:.0f} m/s × {dt / 3600:.2f} h ({wave_speed * dt / 1e3:.0f} km)",
    )
    ax.set_extent([lon0 - 35, lon0 + 35, max(lat0 - 25, -90), min(lat0 + 25, 90)], crs=PC)
    ax.coastlines(lw=0.5)
    ax.gridlines(lw=0.3)
    ax.legend(loc="lower left", fontsize=7, markerscale=3)
    ax.set_title(f"NSL departure points for all heads at '{probe}'")
    _finish(fig, path)


# 3. Mode statistics -----------------------------------------------------------
def plot_mode_statistics(diags, wave_speed: float = 50.0, step=0, path=None):
    sel = _select(diags, step)
    dt = _dt(sel)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    a1.violinplot([d.dist_per_mode for d in sel], showmedians=True)
    a1.axhline(wave_speed * dt / 1e3, ls="--", c="k", lw=1, label=f"{wave_speed:.0f} m/s × {dt / 3600:.2f} h")
    a1.set_xticks(range(1, len(sel) + 1), [str(d.layer + 1) for d in sel])
    a1.set(
        xlabel="ADR layer",
        ylabel="area-mean displacement per head (km)",
        title="How far each transported head travels per sub-step",
    )
    a1.legend()
    gated = [d for d in sel if d.alpha is not None]
    for d in gated:
        a2.hist(d.alpha, bins=40, histtype="step", label=f"layer {d.layer + 1}")
    if gated:
        a2.legend(fontsize=7, ncol=2)
    else:
        a2.text(0.5, 0.5, "no advection gate on these blocks", ha="center", transform=a2.transAxes)
    a2.set(
        xlabel="α  (0 = bypasses advection, 1 = fully advected)",
        ylabel="latent channels",
        title="Advection blend gates",
    )
    _finish(fig, path)


# 4. Information-flow trajectories ---------------------------------------------
def back_trajectories(diags, lat_deg, lon_deg, seeds, step=0):
    """Chain per-layer head-mean back-traces from the last layer to the first."""
    lat_up = lat_deg[::-1]
    lon_ext = np.append(lon_deg, lon_deg[0] + 360)

    def interp(f):
        f = f[::-1]
        return RegularGridInterpolator(
            (lat_up, lon_ext),
            np.concatenate([f, f[:, :1]], axis=1),
            bounds_error=False,
            fill_value=None,
        )

    layers = [(interp(d.east_km), interp(d.north_km)) for d in reversed(_select(diags, step))]
    geod, paths = Geodesic(), []
    for lat0, lon0 in seeds:
        pts = [(lon0, lat0)]
        for fe, fn in layers:
            lo, la = pts[-1]
            q = np.array([[np.clip(la, lat_up[0], lat_up[-1]), lo % 360]])
            e, n = fe(q)[0] * 1e3, fn(q)[0] * 1e3
            res = np.asarray(geod.direct([lo, la], np.degrees(np.arctan2(-e, -n)), np.hypot(e, n)))
            pts.append((res[0, 0], res[0, 1]))
        paths.append(np.array(pts))
    return paths


def plot_trajectories(paths, path=None):
    fig, ax = plt.subplots(figsize=(11, 5.5), subplot_kw={"projection": ccrs.Robinson()})
    ax.set_global()
    ax.coastlines(lw=0.4, color="0.6")
    for p in paths:
        ax.plot(p[:, 0], p[:, 1], lw=0.9, color="tab:blue", transform=GEO)
        ax.plot(p[0, 0], p[0, 1], "o", ms=2.5, color="tab:red", transform=PC)
    ax.set_title("Chained NSL back-traces through all ADR layers (red = arrival point)")
    _finish(fig, path)


# 5. Trajectory-crossing map ----------------------------------------------------
def plot_fold_map(diags, layer, lat_deg, lon_deg, step=0, path=None):
    d = next(x for x in diags if x.step == step and x.layer == layer)
    if d.fold_fraction is None:
        raise ValueError("No fold diagnostics recorded; construct NSLRecorder with fold_check=True.")
    fig, ax = plt.subplots(figsize=(10, 4.8), subplot_kw={"projection": ccrs.Robinson()})
    mesh = ax.pcolormesh(lon_deg, lat_deg[1:-1], d.fold_fraction, transform=PC, cmap="magma_r", shading="auto", vmin=0)
    ax.coastlines(lw=0.4)
    fig.colorbar(mesh, ax=ax, shrink=0.7, label="fraction of heads with crossing trajectories")
    ax.set_title(f"Layer {layer + 1}: trajectory folding (global mean {d.fold_fraction.mean():.2%})")
    _finish(fig, path)


# 6. Rollout animation ------------------------------------------------------------
def animate_latent_speed(diags, layer, lat_deg, lon_deg, step_hours=6, path="latent_speed.mp4", fps=4):
    frames = sorted((d for d in diags if d.layer == layer), key=lambda d: d.step)
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.Robinson()})
    ax.coastlines(lw=0.4, color="0.85")
    speed = lambda d: np.hypot(_ms(d.east_km, d.dt_seconds), _ms(d.north_km, d.dt_seconds))  # noqa: E731
    spd0 = speed(frames[0])
    mesh = ax.pcolormesh(
        lon_deg,
        lat_deg,
        spd0,
        transform=PC,
        cmap="viridis",
        shading="auto",
        vmin=0,
        vmax=np.nanpercentile(spd0, 99),
    )
    fig.colorbar(mesh, ax=ax, shrink=0.7, label="m/s")

    def update(k):
        d = frames[k]
        mesh.set_array(speed(d))
        ax.set_title(f"Layer {layer + 1} latent wind speed, lead time +{step_hours * (d.step + 1):g} h")
        return (mesh,)

    FuncAnimation(fig, update, frames=len(frames), blit=False).save(path, fps=fps)
    plt.close(fig)


def _finish(fig, path):
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

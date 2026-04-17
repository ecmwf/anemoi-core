#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SWD multi-échelle : réel (zarr) vs fake (dossier NetCDF)
- AROME vs AROME : même pool de dates, splitté en deux moitiés
- AROME vs Inférence : même pool AROME vs NetCDF
"""

import os
import glob
import numpy as np
import zarr
import netCDF4 as nc
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.interpolate import griddata
from math import log2

# ── Config ─────────────────────────────────────────────────────────────────────

ZARR_PATH  = '/project/home/p200177/DE_371/datasets/arome-an-oper-titan-0p025-2020-2024-1h-v4-eurw-precip6/arome-an-oper-titan-0p025-2020-2024-1h-v4-eurw-precip6/arome-an-oper-titan-0p025-2020-2024-1h-v4-eurw-precip6.zarr'
NETCDF_DIR = '/project/home/p200177/DE_371/avritj/experiments_anemoi/inference/netcdf_full_training_angelique/'
OUTPUT_DIR = '/project/home/p200177/DE_371/angeliquebonamy/anemoi/inferences/final_training/'

VARS_TO_USE = ['10u', '10v', '2t']

DATE_START = np.datetime64('2024-04-02')
DATE_END   = np.datetime64('2024-10-03')

MAX_SAMPLES=5
GRID_SIZE      = 256
RESOLUTIONS    = [256, 128, 64, 32]
DIR_REPEATS    = 4
DIRS_PER_REPEAT = 128

# ── Pyramide laplacienne ────────────────────────────────────────────────────────

gaussian_filter = np.float32([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1],
]) / 256.0


def pyr_down(minibatch):
    assert minibatch.ndim == 4
    return scipy.ndimage.convolve(
        minibatch, gaussian_filter[np.newaxis, np.newaxis, :, :], mode="mirror"
    )[:, :, ::2, ::2]


def pyr_up(minibatch):
    assert minibatch.ndim == 4
    S = minibatch.shape
    if log2(S[2]) - round(log2(S[2])) != 0:
        res = np.zeros((S[0], S[1], S[2]*2-1, S[3]*2-1), minibatch.dtype)
    else:
        res = np.zeros((S[0], S[1], S[2]*2, S[3]*2), minibatch.dtype)
    res[:, :, ::2, ::2] = minibatch
    return scipy.ndimage.convolve(
        res, gaussian_filter[np.newaxis, np.newaxis, :, :] * 4.0, mode="mirror"
    )


def generate_laplacian_pyramid(minibatch, num_levels):
    pyramid = [np.float32(minibatch)]
    for i in range(1, num_levels):
        pyramid.append(pyr_down(pyramid[-1]))
        pyramid[-2] -= pyr_up(pyramid[-1])
    return pyramid


# ── SWD ────────────────────────────────────────────────────────────────────────

def get_descriptors_for_minibatch(minibatch, nhood_size, nhoods_per_image):
    S = minibatch.shape
    assert len(S) == 4
    N = nhoods_per_image * S[0]
    H = nhood_size // 2
    nhood, chan, x, y = np.ogrid[0:N, 0:S[1], -H:H+1, -H:H+1]
    img = nhood // nhoods_per_image
    x = x + np.random.randint(H, S[3]-H, size=(N, 1, 1, 1))
    y = y + np.random.randint(H, S[2]-H, size=(N, 1, 1, 1))
    idx = ((img * S[1] + chan) * S[2] + y) * S[3] + x
    return minibatch.flat[idx]


def finalize_descriptors(desc):
    if isinstance(desc, list):
        desc = np.concatenate(desc, axis=0)
    assert desc.ndim == 4
    desc -= np.mean(desc, axis=(0, 2, 3), keepdims=True)
    desc /= np.std(desc,  axis=(0, 2, 3), keepdims=True) + 1e-8
    return desc.reshape(desc.shape[0], -1)


def sliced_wasserstein(A, B, dir_repeats, dirs_per_repeat):
    assert A.ndim == 2 and A.shape == B.shape
    results = []
    for _ in range(dir_repeats):
        dirs = np.random.randn(A.shape[1], dirs_per_repeat)
        dirs /= np.sqrt(np.sum(np.square(dirs), axis=0, keepdims=True))
        dirs = dirs.astype(np.float32)
        projA = np.sort(np.matmul(A, dirs), axis=0)
        projB = np.sort(np.matmul(B, dirs), axis=0)
        results.append(np.mean(np.abs(projA - projB)))
    return np.mean(results)


# ── Interpolation ───────────────────────────────────────────────────────────────

def build_grid_coords(lats, lons, grid_size=GRID_SIZE):
    lat_grid = np.linspace(lats.min(), lats.max(), grid_size)
    lon_grid = np.linspace(lons.min(), lons.max(), grid_size)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    points_src = np.stack([lons, lats], axis=1)
    return points_src, lon_mesh, lat_mesh


def interpolate_to_grid(values, points_src, lon_mesh, lat_mesh):
    grid = griddata(points_src, values, (lon_mesh, lat_mesh), method='linear')
    if np.isnan(grid).any():
        grid_near = griddata(points_src, values, (lon_mesh, lat_mesh), method='nearest')
        grid[np.isnan(grid)] = grid_near[np.isnan(grid)]
    return grid.astype(np.float32)


# ── Chargement ─────────────────────────────────────────────────────────────────

def load_zarr_date(ds_zarr, idx, indices_vars, vars_to_use,
                   points_src, lon_mesh, lat_mesh):
    """Charge 1 date du zarr → (1, C, H, W)."""
    channels = []
    for varname in vars_to_use:
        d = ds_zarr['data'].oindex[[idx], indices_vars[varname], 0, :][0]
        grid = interpolate_to_grid(d, points_src, lon_mesh, lat_mesh)
        channels.append(grid)
    return np.stack(channels, axis=0)[np.newaxis]  # (1, C, H, W)


def load_netcdf_as_grid(nc_path, vars_to_use, points_src, lon_mesh, lat_mesh):
    """Charge tous les samples d'un NetCDF → liste de (1, C, H, W)."""
    ds = nc.Dataset(nc_path, "r")
    n_times = ds.dimensions['time'].size
    images = []
    for t in range(1, n_times):   # on saute t=0 comme dans ton code original
        channels = []
        for var in vars_to_use:
            d = np.array(ds.variables[var][t]).reshape(-1)
            grid = interpolate_to_grid(d, points_src, lon_mesh, lat_mesh)
            channels.append(grid)
        images.append(np.stack(channels, axis=0)[np.newaxis])  # (1, C, H, W)
    ds.close()
    return images   # liste de (1, C, H, W)


# ── SWD multi-échelle ───────────────────────────────────────────────────────────

def compute_swd_multiscale(images_A, images_B, nhood_size=7, nhoods_per_image=128):
    """
    images_A, images_B : listes de np.array (1, C, H, W)
    Retourne un array SWD par résolution (×1e3).
    """
    num_levels = len(RESOLUTIONS)
    desc_A = [[] for _ in range(num_levels)]
    desc_B = [[] for _ in range(num_levels)]

    for img in images_A:
        for lod, level in enumerate(generate_laplacian_pyramid(img, num_levels)):
            desc_A[lod].append(get_descriptors_for_minibatch(level, nhood_size, nhoods_per_image))

    for img in images_B:
        for lod, level in enumerate(generate_laplacian_pyramid(img, num_levels)):
            desc_B[lod].append(get_descriptors_for_minibatch(level, nhood_size, nhoods_per_image))

    swd_per_level = []
    for lod in range(num_levels):
        dA = finalize_descriptors(desc_A[lod])
        dB = finalize_descriptors(desc_B[lod])
        n  = min(dA.shape[0], dB.shape[0])
        swd = sliced_wasserstein(dA[:n], dB[:n], DIR_REPEATS, DIRS_PER_REPEAT) * 1e3
        swd_per_level.append(swd)
        print(f"  Résolution {RESOLUTIONS[lod]:4d} : SWD×1e3 = {swd:.4f}")

    return np.array(swd_per_level)


# ── Plot ────────────────────────────────────────────────────────────────────────

def plot_swd(swd_values, labels, output_dir, filename="swd.png"):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ['cyan', 'royalblue']

    for swd, label, color in zip(swd_values, labels, colors):
        ax.plot(range(len(RESOLUTIONS)), swd,
                label=label, linewidth=2.5, color=color, marker='o')

    ax.set_title("Multiscale Sliced Wasserstein Distance", fontsize=16)
    ax.set_ylabel("Distance (×1e3)", fontsize=14)
    ax.set_xlabel("Résolution (points de grille)", fontsize=14)
    ax.set_xticks(range(len(RESOLUTIONS)))
    ax.set_xticklabels([str(r) for r in RESOLUTIONS], fontsize=14)
    ax.tick_params(direction='in', length=12, width=2)
    ax.set_yscale("log")
    ax.legend(fontsize=14, frameon=False)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"Figure sauvegardée : {out_path}")
    plt.close()


# ── Run ─────────────────────────────────────────────────────────────────────────

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Ouverture zarr ──────────────────────────────────────────────────────────
    ds_zarr      = zarr.open(ZARR_PATH, mode='r')
    dates        = ds_zarr['dates'][:]
    variables    = list(ds_zarr.attrs['variables'])
    indices_vars = {v: variables.index(v) for v in VARS_TO_USE}
    lats         = ds_zarr['latitudes'][:]
    lons         = ds_zarr['longitudes'][:]
    points_src, lon_mesh, lat_mesh = build_grid_coords(lats, lons)

    # ── Chargement des images fake (NetCDF inférence) ───────────────────────────
    print("\n── Chargement fake (NetCDF inférence) ──")
    nc_files = sorted(glob.glob(os.path.join(NETCDF_DIR, "*.nc")))[:5]
    print(f"{len(nc_files)} fichiers NetCDF trouvés")

    images_fake = []
    for path in nc_files:
        imgs = load_netcdf_as_grid(path, VARS_TO_USE, points_src, lon_mesh, lat_mesh)
        images_fake.extend(imgs)
        print(f"  {os.path.basename(path)} : {len(imgs)} samples")

    N_fake = len(images_fake)
    print(f"\nTotal samples fake : {N_fake}")

    # ── Sélection de 2×N_fake dates AROME dans la même fenêtre temporelle ──────
    # On tire 2*N_fake dates pour pouvoir splitter en deux moitiés égales
    # → moitié A sert de "réel" pour AROME vs AROME
    # → moitié B sert de "réel" pour AROME vs Inférence
    print("\n── Sélection des dates AROME ──")
    mask          = (dates >= DATE_START) & (dates <= DATE_END)
    indices_dispo = np.where(mask)[0]
    # ── Limitation à 500 samples ────────────────────────────────────────────────
    MAX_TOTAL = MAX_SAMPLES

    # N_fake doit être <= MAX_TOTAL / 2 pour garder le split propre
    N_fake = min(len(nc_files) * 1000000, MAX_TOTAL // 2)  # on ajuste ensuite proprement

    n_needed = 2 * N_fake

    assert len(indices_dispo) >= n_needed, (
        f"Pas assez de dates dispo ({len(indices_dispo)}) pour {n_needed} tirages"
    )
    # n_needed = 2 * N_fake
    # assert len(indices_dispo) >= n_needed, (
    #     f"Pas assez de dates dispo ({len(indices_dispo)}) pour {n_needed} tirages"
    # )

    rng           = np.random.default_rng(42)
    indices_dates = np.sort(rng.choice(indices_dispo, size=n_needed, replace=False))

    # Split en deux moitiés de taille N_fake
    indices_A = indices_dates[:N_fake]   # moitié A  → référence AROME vs AROME
    indices_B = indices_dates[N_fake:]   # moitié B  → réel pour AROME vs Inférence

    print(f"  Moitié A (ref)  : {len(indices_A)} dates")
    print(f"  Moitié B (real) : {len(indices_B)} dates")

    # ── Chargement AROME moitié A ───────────────────────────────────────────────
    print("\n── Interpolation AROME moitié A ──")
    images_real_A = []
    for i, idx in enumerate(indices_A):
        img = load_zarr_date(ds_zarr, idx, indices_vars, VARS_TO_USE,
                             points_src, lon_mesh, lat_mesh)
        images_real_A.append(img)
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(indices_A)} chargées")

    # ── Chargement AROME moitié B ───────────────────────────────────────────────
    print("\n── Interpolation AROME moitié B ──")
    images_real_B = []
    for i, idx in enumerate(indices_B):
        img = load_zarr_date(ds_zarr, idx, indices_vars, VARS_TO_USE,
                             points_src, lon_mesh, lat_mesh)
        images_real_B.append(img)
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(indices_B)} chargées")

    # ── SWD AROME vs AROME (baseline) ──────────────────────────────────────────
    print("\n── Calcul SWD AROME vs AROME ──")
    swd_ref = compute_swd_multiscale(images_real_A, images_real_B)
    np.save(os.path.join(OUTPUT_DIR, "swd_arome_vs_arome.npy"), swd_ref)
    print(f"  Moyenne : {swd_ref.mean():.4f}")

    # ── SWD AROME vs Inférence ──────────────────────────────────────────────────
    print("\n── Calcul SWD AROME vs Inférence ──")
    swd_inf = compute_swd_multiscale(images_real_B, images_fake)
    np.save(os.path.join(OUTPUT_DIR, "swd_arome_vs_inference.npy"), swd_inf)
    print(f"  Moyenne : {swd_inf.mean():.4f}")

    # ── Plot comparatif ─────────────────────────────────────────────────────────
    plot_swd(
        swd_values=[swd_ref, swd_inf],
        labels=["AROME vs AROME (baseline)", "AROME vs Inférence"],
        output_dir=OUTPUT_DIR,
        filename="swd_comparison.png"
    )


run()
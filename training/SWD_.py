#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SWD multi-échelle : réel (zarr) vs fake (dossier NetCDF)
Interpolation sur grille 512x512 + pyramide laplacienne (512, 256, 128, 64)
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
OUTPUT_DIR = '/project/home/p200177/DE_371/angeliquebonamy/anemoi/inferences/final_training/scores_190K_steps/'
AROME_scores = '/project/home/p200177/DE_371/angeliquebonamy/anemoi/inferences/scores_arome/'

VARS_TO_USE = ['10u', '10v', '2t']

DATE_START = np.datetime64('2024-04-02')
DATE_END   = np.datetime64('2024-10-03')
N_SAMPLES  = 500
CHUNK      = 10    # dates par chunk (réduit pour éviter OOM)

GRID_SIZE      = 256                 # grille carrée cible
RESOLUTIONS    = [256, 128, 64,32] #[512, 256, 128, 64]  # échelles de la pyramide
DIR_REPEATS    = 4
DIRS_PER_REPEAT = 128

# # ── Pyramide laplacienne (code original inchangé) ───────────────────────────────

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


# ── SWD (code original inchangé) ───────────────────────────────────────────────

def get_descriptors_for_minibatch(minibatch, nhood_size, nhoods_per_image):
    S = minibatch.shape   # (B, C, H, W)
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


# ── Interpolation non structuré → grille ────────────────────────────────────────

def build_grid_coords(lats, lons, grid_size=GRID_SIZE):
    """Prépare les coordonnées de la grille cible (fait une seule fois)."""
    lat_grid = np.linspace(lats.min(), lats.max(), grid_size)
    lon_grid = np.linspace(lons.min(), lons.max(), grid_size)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    points_src = np.stack([lons, lats], axis=1)
    return points_src, lon_mesh, lat_mesh


def interpolate_to_grid(values, points_src, lon_mesh, lat_mesh):
    """
    Interpole des points non structurés sur une grille régulière.
    NaN de bord remplis avec 'nearest'.
    """
    print("points_src:", points_src.shape)
    print("values:", values.shape)
    grid = griddata(points_src, values, (lon_mesh, lat_mesh), method='linear')
    if np.isnan(grid).any():
        grid_near = griddata(points_src, values, (lon_mesh, lat_mesh), method='nearest')
        grid[np.isnan(grid)] = grid_near[np.isnan(grid)]
    return grid.astype(np.float32)


# ── Chargement ─────────────────────────────────────────────────────────────────

def remove_nan_rows(data):
    mask = np.isfinite(data).all(axis=1)
    return data[mask]


def load_zarr_date(ds_zarr, idx, indices_vars, vars_to_use,
                   points_src, lon_mesh, lat_mesh):
    """
    Charge 1 date du zarr et retourne une image (C, H, W) interpolée sur grille.
    """
    channels = []
    for varname in vars_to_use:
        d = ds_zarr['data'].oindex[[idx], indices_vars[varname], 0, :][0]
        grid = interpolate_to_grid(d, points_src, lon_mesh, lat_mesh)
        channels.append(grid)
    return np.stack(channels, axis=0)   # (C, H, W)


# def load_netcdf_as_grid(nc_path, vars_to_use, points_src, lon_mesh, lat_mesh):
#     """
#     Charge un fichier NetCDF et retourne une image (1, C, H, W).
#     """
#     ds = nc.Dataset(nc_path, "r")
#     print(ds)
    
#     channels = []
#     for var in vars_to_use:
#         d = np.array(ds.variables[var][:][0]) #.flatten()
#         print(f"{var} shape brut:", d.shape)  # 👈 IMPORTANT

#         d = d.flatten()
#         print(f"{var} shape flatten:", d.shape)

#         grid = interpolate_to_grid(d, points_src, lon_mesh, lat_mesh)
#         channels.append(grid)
   

    
#     ds.close()
#     return np.stack(channels, axis=0)[np.newaxis]  # (1, C, H, W)

def load_netcdf_as_grid(nc_path, vars_to_use, points_src, lon_mesh, lat_mesh): #pour charger avec 100 samples ds le netcdf
    ds = nc.Dataset(nc_path, "r")
    all_images = []

    for t in range(1,100): #ds.dimensions['time'].size):  # 101
        channels = []

        for var in vars_to_use:
            d = np.array(ds.variables[var][t])  # 👈 UNE TIME
            values = d.reshape(-1)

            grid = interpolate_to_grid(values, points_src, lon_mesh, lat_mesh)
            channels.append(grid)

        img = np.stack(channels, axis=0)  # (C, H, W)
        all_images.append(img)

    ds.close()
    return np.stack(all_images, axis=0)  # (101, C, H, W)


# ── Pipeline SWD multi-échelle ─────────────────────────────────────────────────

def compute_swd_multiscale(images_real, images_fake, nhood_size=7, nhoods_per_image=128):
    """
    images_real : list de np.array (1, C, H, W)
    images_fake : list de np.array (1, C, H, W)
    Retourne une SWD par résolution (×1e3).
    """
    num_levels = len(RESOLUTIONS)

    # Accumulateurs de descripteurs par niveau
    desc_real = [[] for _ in range(num_levels)]
    desc_fake = [[] for _ in range(num_levels)]

    for img in images_real:
        pyramid = generate_laplacian_pyramid(img, num_levels)
        for lod, level in enumerate(pyramid):
            desc_real[lod].append(
                get_descriptors_for_minibatch(level, nhood_size, nhoods_per_image)
            )

    for img in images_fake:
        pyramid = generate_laplacian_pyramid(img, num_levels)
        for lod, level in enumerate(pyramid):
            desc_fake[lod].append(
                get_descriptors_for_minibatch(level, nhood_size, nhoods_per_image)
            )

    # Calcul SWD par niveau
    swd_per_level = []
    for lod in range(num_levels):
        dr = finalize_descriptors(desc_real[lod])
        df = finalize_descriptors(desc_fake[lod])
        # Alignement des tailles
        n = min(dr.shape[0], df.shape[0])
        dr = dr[:n]
        df = df[:n]
        swd = sliced_wasserstein(dr, df, DIR_REPEATS, DIRS_PER_REPEAT) * 1e3
        swd_per_level.append(swd)
        print(f"  Résolution {RESOLUTIONS[lod]:4d} : SWD×1e3 = {swd:.4f}")

    return np.array(swd_per_level)


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_swd(swd_values, labels, output_dir):
    """
    swd_values : list de np.array (4,) — un par expérience
    labels     : list de str
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(swd_values)))

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
    out_path = os.path.join(output_dir, "swd.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"Figure sauvegardée : {out_path}")
    plt.close()


# ── Run ────────────────────────────────────────────────────────────────────────

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nc_files = sorted(glob.glob(os.path.join(NETCDF_DIR, "*.nc")))[:5]
    print(f"{len(nc_files)} fichiers NetCDF trouvés")

    # Ouverture zarr
    ds_zarr      = zarr.open(ZARR_PATH, mode='r')
    dates        = ds_zarr['dates'][:]
    variables    = list(ds_zarr.attrs['variables'])
    indices_vars = {v: variables.index(v) for v in VARS_TO_USE}
    lats         = ds_zarr['latitudes'][:]
    lons         = ds_zarr['longitudes'][:]
    print(f"Indices variables : {indices_vars}")

    # Grille d'interpolation (calculée une seule fois)
    points_src, lon_mesh, lat_mesh = build_grid_coords(lats, lons)

    # Sélection des dates
    mask          = (dates >= DATE_START) & (dates <= DATE_END)
    indices_dispo = np.where(mask)[0]
    n             = min(N_SAMPLES, len(indices_dispo))
    rng           = np.random.default_rng(42)
    indices_dates = np.sort(rng.choice(indices_dispo, size=n, replace=False))
    print(f"{n} dates sélectionnées")

    # ── Chargement des images (par chunks pour limiter RAM) ───────────────────
    print("\n── Interpolation réel (zarr) ──")
    images_real = []
    for i, idx in enumerate(indices_dates[:CHUNK*5]):  # on prend 50 dates max pour la démo
        img = load_zarr_date(ds_zarr, idx, indices_vars, VARS_TO_USE,
                             points_src, lon_mesh, lat_mesh)
        images_real.append(img[np.newaxis])  # (1, C, H, W)
        if (i+1) % 10 == 0:
            print(f"  {i+1} dates interpolées")

    print("\n── Interpolation fake (NetCDF) ──")
    images_fake = []
    for path in nc_files:
        img = load_netcdf_as_grid(path, VARS_TO_USE, points_src, lon_mesh, lat_mesh)
        images_fake.append(img)
        print(f"  {os.path.basename(path)} interpolé")

    # ── Calcul SWD multi-échelle ──────────────────────────────────────────────
    print("\n── Calcul SWD multi-échelle ──")
    swd = compute_swd_multiscale(images_real, images_fake)
    # swd = compute_swd_multiscale(images_real, images_real)


    print(f"\nRésultats :")
    for res, val in zip(RESOLUTIONS, swd):
        print(f"  {res:4d} pts : {val:.4f}")
    print(f"  Moyenne : {swd.mean():.4f}")

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    npy_path = os.path.join(OUTPUT_DIR, "swd_multiscale_arome.npy")
    np.save(npy_path, swd)
    print(f"SWD sauvegardée : {npy_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_swd([swd], ["AROME vs AROME"], OUTPUT_DIR)


run()

# inf = np.load(f'{OUTPUT_DIR}swd_multiscale_inf.npy')
# ar = np.load(f'{AROME_scores}swd_multiscale_arome.npy')

# plot_swd(
#     swd_values=[inf, ar],
#     labels=["Inference", "AROME"],
#     output_dir=OUTPUT_DIR
# )
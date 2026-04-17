#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Histogrammes bivariés : réel (zarr) vs fake (dossier NetCDF)
Variables : 10u, 10v, 2t
NetCDF : time[0] = analyse ignorée, time[1:] = 100 échantillons générés
"""

import os
import glob
import numpy as np
import zarr
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from itertools import combinations_with_replacement

# ── Config ─────────────────────────────────────────────────────────────────────

ZARR_PATH  = '/project/home/p200177/DE_371/datasets/arome-an-oper-titan-0p025-2020-2024-1h-v4-eurw-precip6/arome-an-oper-titan-0p025-2020-2024-1h-v4-eurw-precip6/arome-an-oper-titan-0p025-2020-2024-1h-v4-eurw-precip6.zarr'
NETCDF_DIR = '/project/home/p200177/DE_371/avritj/experiments_anemoi/inference/netcdf_full_training_angelique/'
OUTPUT_DIR = './output/'

VARS_TO_USE = ['10u', '10v', '2t']
DATE_START  = np.datetime64('2024-04-02')
DATE_END    = np.datetime64('2024-10-03')
N_SAMPLES   = 5
CHUNK       = 20
N_BINS      = 100
N_LEVELS    = 15


# ── Filtre NaN ─────────────────────────────────────────────────────────────────

def remove_nan_rows(data):
    mask = np.isfinite(data).all(axis=1)
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        print(f"  [NaN] {n_dropped}/{data.shape[0]} points supprimés")
    return data[mask]


# ── Histogrammes bivariés ───────────────────────────────────────────────────────

def var2Var_hist_counts(data, bins):
    """
    data : np.array (N, C)
    bins : int        -> calcule les bins, retourne (counts, Bins)
           np.ndarray -> réutilise les bins fournis, retourne (counts, Bins)
    """
    data = remove_nan_rows(data)

    channels    = data.shape[1]
    var_couples = list(combinations_with_replacement(range(channels), 2))
    ncouples    = channels * (channels - 1) // 2

    if isinstance(bins, int):
        nbins  = bins
        Bins   = [None] * channels
        counts = np.zeros((ncouples, nbins, nbins))
    else:
        Bins   = bins
        nbins  = Bins.shape[1] - 1
        counts = np.zeros((ncouples, nbins, nbins))

    k = 0
    for i, j in var_couples:
        if i != j:
            if isinstance(bins, int):
                c, bx, by = np.histogram2d(data[:, i], data[:, j],
                                            bins=nbins, density=False)
                counts[k] = c
                Bins[i]   = bx
                Bins[j]   = by
            else:
                c, _, _ = np.histogram2d(data[:, i], data[:, j],
                                         bins=[Bins[i], Bins[j]], density=False)
                counts[k] = c
            k += 1

    if isinstance(bins, int):
        Bins = np.array(Bins)

    return counts, Bins


def normalize_counts(counts, bins):
    ncouples = counts.shape[0]
    density  = np.zeros_like(counts, dtype=float)
    pairs    = [(0, 1), (0, 2), (1, 2)]
    total    = counts.sum(axis=(1, 2), keepdims=True)
    for k, (i, j) in enumerate(pairs[:ncouples]):
        dx = np.diff(bins[i])[:, None]
        dy = np.diff(bins[j])[None, :]
        density[k] = counts[k] / (total[k] * dx * dy + 1e-30)
    return density


def define_levels(bivariates_r, bivariates_f, nlevels):
    levels = []
    for i in range(bivariates_r.shape[0]):
        combined = np.concatenate([bivariates_r[i].ravel(), bivariates_f[i].ravel()])
        pos = np.sort(combined[combined > 0])
        levels.append(np.logspace(np.log10(pos[0]), np.log10(pos[-1]), nlevels))
    return levels


def plot2D_histo(bivariates_r, bivariates_f, bins_r, levels, output_dir, add_name):
    pairs    = [(0, 1, "10u", "10v"), (0, 2, "10u", "2t"), (1, 2, "10v", "2t")]
    ncouples = bivariates_r.shape[0]

    fig, axs = plt.subplots(1, ncouples, figsize=(5 * ncouples, 5))
    if ncouples == 1:
        axs = [axs]

    cs_last = None
    for i, (xi, yi, xlabel, ylabel) in enumerate(pairs[:ncouples]):
        log_lvl = np.log10(levels[i])
        z_r = np.where(bivariates_r[i] > 0, np.log10(bivariates_r[i]), np.nan)
        z_f = np.where(bivariates_f[i] > 0, np.log10(bivariates_f[i]), np.nan)

        cs = axs[i].contourf(bins_r[xi][:-1], bins_r[yi][:-1], z_r.T,
                              cmap="plasma", levels=log_lvl, extend="both")
        axs[i].contour(bins_r[xi][:-1], bins_r[yi][:-1], z_f.T,
                       cmap="Greys", levels=log_lvl, linewidths=1.2)
        axs[i].set_xlabel(xlabel, fontsize="large", fontweight="bold")
        axs[i].set_ylabel(ylabel, fontsize="large", fontweight="bold")
        axs[i].set_title(f"{xlabel} vs {ylabel}", fontsize="medium")
        cs_last = cs

    fig.subplots_adjust(right=0.87)
    cbax = fig.add_axes([0.90, 0.1, 0.02, 0.8])
    cb   = fig.colorbar(cs_last, cax=cbax)
    cb.ax.tick_params(labelsize=10)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cb.set_label("log₁₀(Densité)", fontweight="bold", fontsize="large",
                 rotation=270, labelpad=18)
    fig.suptitle(f"Bivariés — réel zarr (plasma) vs fake NetCDF (gris)\n{add_name}",
                 fontsize=12, fontweight="bold")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"bivariate_{add_name}.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Figure sauvegardée : {out_path}")
    plt.close()


# ── Chargement NetCDF ──────────────────────────────────────────────────────────

def load_netcdf_file(nc_path, vars_to_use):
    """
    Charge un fichier NetCDF.
    time[0]  = analyse → ignorée
    time[1:] = 100 échantillons générés → on prend tout et on aplatit
    Retourne (N_samples * N_points, C)
    """
    ds = nc.Dataset(nc_path, "r")
    arrays = []
    for var in vars_to_use:
        if var not in ds.variables:
            raise KeyError(f"Variable '{var}' absente de {nc_path}")
        data = np.array(ds.variables[var][1:])  # (100, 665679) — on saute time[0]
        arrays.append(data.flatten())            # (100 * 665679,)
    ds.close()
    result = np.stack(arrays, axis=1)            # (100 * 665679, C)
    print(f"  {os.path.basename(nc_path)} : {result.shape[0]} points "
          f"({np.array(nc.Dataset(nc_path).variables[vars_to_use[0]][1:]).shape[0]} échantillons)")
    return result


# ── Chargement zarr ─────────────────────────────────────────────────────────────

def load_zarr_chunk(ds_zarr, idx_chunk, indices_vars, vars_to_use):
    arrays = []
    for varname in vars_to_use:
        d = ds_zarr['data'].oindex[idx_chunk, indices_vars[varname], 0, :]
        arrays.append(d.reshape(-1))
    return remove_nan_rows(np.stack(arrays, axis=1).astype(np.float32))


# ── Pipeline principal ─────────────────────────────────────────────────────────

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nc_files = sorted(glob.glob(os.path.join(NETCDF_DIR, "*.nc")))[:5]
    print(f"{len(nc_files)} fichiers NetCDF trouvés")

    # Ouverture zarr
    ds_zarr      = zarr.open(ZARR_PATH, mode='r')
    dates        = ds_zarr['dates'][:]
    variables    = list(ds_zarr.attrs['variables'])
    indices_vars = {v: variables.index(v) for v in VARS_TO_USE}
    print(f"Indices variables zarr : {indices_vars}")

    # Sélection des dates
    mask          = (dates >= DATE_START) & (dates <= DATE_END)
    indices_dispo = np.where(mask)[0]
    n             = min(N_SAMPLES, len(indices_dispo))
    rng           = np.random.default_rng(42)
    indices_dates = np.sort(rng.choice(indices_dispo, size=n, replace=False))
    print(f"{n} dates sélectionnées sur {len(indices_dispo)} disponibles")

    # ── 1. Bins sur le premier chunk zarr ─────────────────────────────────────
    print("\n── Calcul des bins (premier chunk zarr) ──")
    data_first = load_zarr_chunk(ds_zarr, indices_dates[:CHUNK], indices_vars, VARS_TO_USE)
    _, bins_r  = var2Var_hist_counts(data_first, N_BINS)
    ncouples   = len(VARS_TO_USE) * (len(VARS_TO_USE) - 1) // 2
    print(f"Bins calculés sur {data_first.shape[0]} points valides")

    # ── 2. Accumulation zarr (réel) ───────────────────────────────────────────
    print("\n── Accumulation réel (zarr) ──")
    counts_r = np.zeros((ncouples, N_BINS, N_BINS))
    for start in range(0, len(indices_dates), CHUNK):
        idx_chunk = indices_dates[start:start + CHUNK]
        chunk = load_zarr_chunk(ds_zarr, idx_chunk, indices_vars, VARS_TO_USE)
        c, _  = var2Var_hist_counts(chunk, bins_r)
        counts_r += c
        print(f"  chunk {start}–{start + len(idx_chunk)} accumulé")

    # ── 3. Accumulation NetCDF (fake, time[1:] seulement) ────────────────────
    print("\n── Accumulation fake (NetCDF, échantillons générés) ──")
    counts_f = np.zeros((ncouples, N_BINS, N_BINS))
    for path in nc_files:
        data = load_netcdf_file(path, VARS_TO_USE)
        c, _ = var2Var_hist_counts(data, bins_r)
        counts_f += c

    # ── 4. Normalisation + plot ───────────────────────────────────────────────
    bivariates_r = normalize_counts(counts_r, bins_r)
    bivariates_f = normalize_counts(counts_f, bins_r)

    levels = define_levels(bivariates_r, bivariates_f, N_LEVELS)
    plot2D_histo(bivariates_r, bivariates_f, bins_r, levels, OUTPUT_DIR, "zarr_vs_netcdf")

    # ── 5. Sauvegarde ─────────────────────────────────────────────────────────
    npz_path = os.path.join(OUTPUT_DIR, "bivariates_zarr_vs_netcdf.npz")
    np.savez(npz_path, real=bivariates_r, fake=bivariates_f, bins=bins_r)
    print(f"Données sauvegardées : {npz_path}")


run()
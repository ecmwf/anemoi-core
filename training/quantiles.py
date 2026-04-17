import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import xarray as xr
import netCDF4 as nc
import glob
from tqdm import tqdm


# DOSSIER = '/project/home/p200177/DE_371/angeliquebonamy/anemoi/inferences/quantiles'

# ZARR_PATH = '/project/home/p200177/DE_371/datasets/arome-an-oper-titan-0p025-2020-2024-1h-v4-eurw-precip6/arome-an-oper-titan-0p025-2020-2024-1h-v4-eurw-precip6/arome-an-oper-titan-0p025-2020-2024-1h-v4-eurw-precip6.zarr'

# ds = zarr.open(ZARR_PATH, mode='r')
# dates = ds['dates'][:]  # array de datetime64

# # Filtrer les dates voulues
# date_start = np.datetime64('2024-04-02')
# date_end   = np.datetime64('2024-10-03')

# mask = (dates >= date_start) & (dates <= date_end)
# indices_disponibles = np.where(mask)[0]
# print(f"{len(indices_disponibles)} dates disponibles dans l'intervalle")

# # Tirer 500 dates aléatoires parmi celles disponibles
# n_samples = min(500, len(indices_disponibles))  # au cas où il y en a moins de 500
# np.random.seed(42)
# indices_dates = np.random.choice(indices_disponibles, size=n_samples, replace=False)
# indices_dates.sort()

# print(f"{len(indices_dates)} dates sélectionnées")

# # 500 dates aléatoires
# # np.random.seed(42)
# # indices_dates = np.random.choice(42060, size=500, replace=False)
# # indices_dates.sort()

# variables = list(ds.attrs['variables'])
# vars_to_plot = ['2t', '10u', '10v']
# indices_vars = {v: variables.index(v) for v in vars_to_plot}
# print(indices_vars)

# # Lat/lon (même grille que tes fichiers d'inférence)
# lats = ds['latitudes'][:]
# lons = ds['longitudes'][:]

# triang = tri.Triangulation(lons, lats)

# for varname, var_idx in indices_vars.items():
#     print(f"Traitement de {varname}...")
#     data = ds['data'].oindex[indices_dates, var_idx, 0, :]  # (500, 665679)

#     q10 = np.nanpercentile(data, 10, axis=0) # que 90% des samples sont supérieur à la valeur Y(ex = 8m/s) au pixel A 
#     q50 = np.nanpercentile(data, 50, axis=0)
#     q90 = np.nanpercentile(data, 90, axis=0)

#     vmin = np.nanmin(q10)
#     vmax = np.nanmax(q90)
    
#     np.save(f'{DOSSIER}/q10_{varname}_test_arome.npy', q10)
#     np.save(f'{DOSSIER}/q50_{varname}_test_arome.npy', q50)
#     np.save(f'{DOSSIER}/q90_{varname}_test_arome.npy', q90)
#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#     for ax, values, title in zip(axes, [q10, q50, q90], ['Q10', 'Q50', 'Q90']):
#         im = ax.tripcolor(triang, values, cmap='RdBu_r', vmin=vmin, vmax=vmax)
#         fig.colorbar(im, ax=ax, label=varname, shrink=0.8)
#         ax.set_title(f'{title} — {varname}')
#         ax.set_xlabel('Longitude')
#         ax.set_ylabel('Latitude')
#         ax.set_aspect('equal')

#     plt.suptitle(f'Quantiles de {varname} — AROME test (500 dates)', fontsize=14)
#     plt.tight_layout()
#     plt.savefig(f'{DOSSIER}/quantiles_{varname}_test_arome.png', dpi=150, bbox_inches='tight')
#     plt.close()
#     print(f"Sauvegardé : quantiles_{varname}_test_arome.png")
    
######################################################### PLOT NETCDF ###################################################



NETCDF ='/project/home/p200177/DE_371/avritj/experiments_anemoi/inference/netcdf_full_training_angelique'
DOSSIER='/project/home/p200177/DE_371/angeliquebonamy/anemoi/inferences/final_training/scores_190K_steps'
# Trouver tous les .nc
fichiers = sorted(glob.glob(f'{NETCDF}/*.nc'))
print(f"{len(fichiers)} fichiers trouvés")

# Lat/lon depuis le premier fichier
ds_ref = nc.Dataset(fichiers[0])
lats = ds_ref['latitude'][:]
lons = ds_ref['longitude'][:]
ds_ref.close()

triang = tri.Triangulation(lons, lats)

vars_config = {
    '2t':  {'cmap': 'coolwarm', 'label': '2t (K)'},
    '10u': {'cmap': 'coolwarm',  'label': '10u (m/s)'},
    '10v': {'cmap': 'coolwarm',  'label': '10v (m/s)'},
}


for varname, cfg in vars_config.items():
    print(f"Chargement de {varname} sur {len(fichiers)} fichiers...")
    data = []
    fichiers_valides = 0
    for f in tqdm(fichiers, desc=f"  Lecture {varname}"):
        try:
            ds = nc.Dataset(f)
            print(np.array(ds[varname][1:,]).shape)
            # data.append(np.array(ds[varname][1, :]))
            print('TEST',varname, np.mean(np.array(ds[varname][1:,])))
            data.append(np.array(ds[varname][1:,]))

            ds.close()
            fichiers_valides += 1
        except Exception as e:
            print(f"\n  Fichier ignoré : {f} ({e})")

    print(f"  {fichiers_valides}/{len(fichiers)} fichiers lus avec succès")
    
    # data = []
    # for f in tqdm(fichiers, desc=f"  Lecture {varname}"):
    #     ds = nc.Dataset(f)
    #     data.append(ds[varname][0, :])
    #     ds.close()

    # data = np.stack(data, axis=0)  # (n_fichiers, 665679)
    data = np.concatenate(data, axis=0)
    print(data.shape,'je suis data shape')
    print(f"  Shape : {data.shape}")


#     print(f"  Calcul des quantiles... (peut prendre quelques minutes)")
#     q10 = np.nanpercentile(data, 10, axis=0)
#     print(f"  Q10 done")
#     q50 = np.nanpercentile(data, 50, axis=0)
#     print(f"  Q50 done")
#     q90 = np.nanpercentile(data, 90, axis=0)
#     print(f"  Q90 done")

#     # Sauvegarde npy
#     np.save(f'{DOSSIER}/q10_{varname}_inference_193K.npy', q10)
#     np.save(f'{DOSSIER}/q50_{varname}_inference_193K.npy', q50)
#     np.save(f'{DOSSIER}/q90_{varname}_inference_193K.npy', q90)
#     print(f"  .npy sauvegardés pour {varname}")

#     # Plot
#     vmin = np.nanmin(q10)
#     vmax = np.nanmax(q90)

#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#     for ax, values, title in zip(axes, [q10, q50, q90], ['Q10', 'Q50', 'Q90']):
#         im = ax.tripcolor(triang, values, cmap=cfg['cmap'], vmin=vmin, vmax=vmax)
#         fig.colorbar(im, ax=ax, label=cfg['label'], shrink=0.8)
#         ax.set_title(f'{title} — {varname}')
#         ax.set_xlabel('Longitude')
#         ax.set_ylabel('Latitude')
#         ax.set_aspect('equal')

#     plt.suptitle(f'Quantiles de {varname} — {len(fichiers)} samples', fontsize=14)
#     plt.tight_layout()
#     plt.savefig(f'{DOSSIER}/quantiles_{varname}_inference_193K.png', dpi=150, bbox_inches='tight')
#     plt.close()
#     print(f"  .png sauvegardé pour {varname}")
    
 
 
 


####################################################### PLOT DIFF #####################################################

# NETCDF ='/project/home/p200177/DE_371/avritj/experiments_anemoi/inference/netcdf_full_training_angelique'
# DOSSIER='/project/home/p200177/DE_371/angeliquebonamy/anemoi/inferences/final_training/scores_190K_steps'
# AROME_QUANTILES = '/project/home/p200177/DE_371/angeliquebonamy/anemoi/inferences/quantiles_arome'# Trouver tous les .nc
# fichiers = sorted(glob.glob(f'{NETCDF}/*.nc'))
# print(f"{len(fichiers)} fichiers trouvés")

# # Lat/lon depuis le premier fichier
# ds_ref = nc.Dataset(fichiers[0])
# lats = ds_ref['latitude'][:]
# lons = ds_ref['longitude'][:]
# ds_ref.close()

# triang = tri.Triangulation(lons, lats)
# vars_config = {
#     '2t':  {'cmap': 'coolwarm', 'label': '2t (K)'},
#     '10u': {'cmap': 'coolwarm',  'label': '10u (m/s)'},
#     '10v': {'cmap': 'coolwarm',  'label': '10v (m/s)'},
# }


# for varname, cfg in vars_config.items():
#     q10_inf   = np.load(f'{DOSSIER}/q10_{varname}_inference_193K.npy')
#     q50_inf   = np.load(f'{DOSSIER}/q50_{varname}_inference_193K.npy')
#     q90_inf   = np.load(f'{DOSSIER}/q90_{varname}_inference_193K.npy')

#     q10_arome = np.load(f'{AROME_QUANTILES}/q10_{varname}_train_arome.npy')
#     q50_arome = np.load(f'{AROME_QUANTILES}/q50_{varname}_train_arome.npy')
#     q90_arome = np.load(f'{AROME_QUANTILES}/q90_{varname}_train_arome.npy')

# #     # Différences
#     diff_q10 = q10_inf - q10_arome
#     diff_q50 = q50_inf - q50_arome
#     diff_q90 = q90_inf - q90_arome

# #     # Colorbar symétrique centrée sur 0
#     vmax = np.nanmax(np.abs([diff_q10, diff_q50, diff_q90]))
#     vmin = -vmax

#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#     for ax, diff, title in zip(axes, [diff_q10, diff_q50, diff_q90], ['Q10', 'Q50', 'Q90']):
#         im = ax.tripcolor(triang, diff, cmap=cfg['cmap'], vmin=vmin, vmax=vmax)
#         fig.colorbar(im, ax=ax, label=cfg['label'], shrink=0.8)
#         ax.set_title(f'Diff {title} — {varname} (inf - AROME)')
#         ax.set_xlabel('Longitude')
#         ax.set_ylabel('Latitude')
#         ax.set_aspect('equal')

#     plt.suptitle(f'Différence inférence - AROME : {varname}', fontsize=14)
#     plt.tight_layout()
#     plt.savefig(f'{DOSSIER}/diff_{varname}_inf_193K_train_arome.png', dpi=150, bbox_inches='tight')
#     plt.close()
#     print(f"Sauvegardé : {DOSSIER}/diff_{varname}_inf_193K_train_arome.png")
    
    
    # fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # for ax, values, title in zip(axes, [q10_arome, q50_arome, q90_arome], ['Q10', 'Q50', 'Q90']):
    #     vmin = np.nanmin(q10_arome)
    #     vmax = np.nanmax(q90_arome)
    #     im = ax.tripcolor(triang, values, cmap=cfg['cmap'], vmin=vmin, vmax=vmax)
    #     fig.colorbar(im, ax=ax, label=varname, shrink=0.8)
    #     ax.set_title(f'{title} — {varname}')
    #     ax.set_xlabel('Longitude')
    #     ax.set_ylabel('Latitude')
    #     ax.set_aspect('equal')

    # plt.suptitle(f'Quantiles de {varname} — AROME (500 dates)', fontsize=14)
    # plt.tight_layout()
    # plt.savefig(f'{DOSSIER}/quantiles_{varname}_arome.png', dpi=150, bbox_inches='tight')
    # plt.close()
    # print(f"Sauvegardé : quantiles_{varname}_arome.png")
    
    
    
    
    
################################################" Quantiles QQ ########################################################

# import numpy as np
# import matplotlib.pyplot as plt

# vars_config = {
#     '2t':  {'label': '2t (K)',    'color': '#D85A30'},
#     '10u': {'label': '10u (m/s)', 'color': '#378ADD'},
#     '10v': {'label': '10v (m/s)', 'color': '#1D9E75'},
# }
# DOSSIER='/project/home/p200177/DE_371/angeliquebonamy/anemoi/inferences/final_training/scores_190K_steps'
# AROME_QUANTILES = '/project/home/p200177/DE_371/angeliquebonamy/anemoi/inferences/quantiles_arome'

# # Quantiles à calculer (de 1% à 99%)
# percentiles = np.arange(1, 100, 1)

# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# for ax, (varname, cfg) in zip(axes, vars_config.items()):

#     q_inf   = np.load(f'{DOSSIER}/q10_{varname}_inference_193K.npy')
#     q_arome = np.load(f'{AROME_QUANTILES}/q10_{varname}_train_arome.npy')

#     # Calculer les quantiles sur tous les points de grille
#     q_inf_vals   = np.nanpercentile(q_inf,   percentiles)
#     q_arome_vals = np.nanpercentile(q_arome, percentiles)

#     # Droite y=x
#     vmin = min(q_inf_vals.min(), q_arome_vals.min())
#     vmax = max(q_inf_vals.max(), q_arome_vals.max())
#     ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1, label='y=x')

#     # QQ-plot
#     ax.scatter(q_arome_vals, q_inf_vals, color=cfg['color'], s=20, alpha=0.7)

#     ax.set_xlabel(f'AROME — {cfg["label"]}')
#     ax.set_ylabel(f'Inférence — {cfg["label"]}')
#     ax.set_title(f'QQ-plot {varname}')
#     ax.legend()
#     ax.set_aspect('equal')

# plt.suptitle('QQ-plot inférence vs AROME (Q10)', fontsize=14)
# plt.tight_layout()
# plt.savefig(f'{DOSSIER}/qq10plot_inf2_train_arome.png', dpi=150, bbox_inches='tight')
# plt.close()
# print("Sauvegardé : qqplot_inf2_train_arome.png")



import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import zarr
from tqdm import tqdm

NETCDF ='/project/home/p200177/DE_371/avritj/experiments_anemoi/inference/netcdf_full_training_angelique'
ZARR_PATH = '/project/home/p200177/DE_371/datasets/arome-an-oper-titan-0p025-2020-2024-1h-v4-eurw-precip6/arome-an-oper-titan-0p025-2020-2024-1h-v4-eurw-precip6/arome-an-oper-titan-0p025-2020-2024-1h-v4-eurw-precip6.zarr'
DOSSIER='/project/home/p200177/DE_371/angeliquebonamy/anemoi/inferences/final_training/scores_190K_steps'

vars_config = {
    '2t':  {'label': '2t (K)',    'color': '#D85A30'},
    '10u': {'label': '10u (m/s)', 'color': '#378ADD'},
    '10v': {'label': '10v (m/s)', 'color': '#1D9E75'},
}

percentiles = np.arange(1, 100, 1)

# --- Charger toutes les inférences : shape (n_samples * 665679,) ---
fichiers = sorted(glob.glob(f'{NETCDF}/*.nc'))

# --- Charger AROME ---output_file
ds_zarr = zarr.open(ZARR_PATH, mode='r')
variables = list(ds_zarr.attrs['variables'])
dates = ds_zarr['dates'][:]
date_start = np.datetime64('2020-01-01')
date_end   = np.datetime64('2023-01-04')
mask = (dates >= date_start) & (dates <= date_end)
indices_disponibles = np.where(mask)[0]
n_samples = min(500, len(indices_disponibles))
np.random.seed(42)
indices_dates = np.random.choice(indices_disponibles, size=n_samples, replace=False)
indices_dates.sort()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (varname, cfg) in zip(axes, vars_config.items()):
    data_inf = []

    for f in tqdm(fichiers, desc=f'Lecture inférence {varname}'):
        try:
            ds = nc.Dataset(f)
            d = np.array(ds[varname][1:, :])  # (100, points)
            data_inf.append(d.reshape(-1))    # flatten direct
            ds.close()
        except Exception as e:
            print(f"Fichier ignoré : {f} ({e})")

    data_inf = np.concatenate(data_inf)  # 1D directement
    # Inférence : flatten tous les samples et tous les points
    # data_inf = []
    # for f in tqdm(fichiers, desc=f'Lecture inférence {varname}'):
    #     try:
    #         ds = nc.Dataset(f)
    #         # data_inf.append(np.array(ds[varname][0, :]))
    #         data_inf.append(np.array(ds[varname][1:,: ])) # pour 100 samples

    #         ds.close()
    #     except Exception as e:
    #         print(f"Fichier ignoré : {f} ({e})")
    # data_inf = np.stack(data_inf, axis=0).flatten()  # (n_samples * 665679,)

    # AROME : flatten tous les samples et tous les points
    var_idx = variables.index(varname)
    data_arome = ds_zarr['data'].oindex[indices_dates, var_idx, 0, :]  # (500, 665679)
    data_arome = data_arome.flatten()  # (500 * 665679,)

    # Calcul des percentiles sur la distribution complète
    q_inf   = np.nanpercentile(data_inf,   percentiles)
    q_arome = np.nanpercentile(data_arome, percentiles)

    # Droite y=x
    vmin = min(q_inf.min(), q_arome.min())
    vmax = max(q_inf.max(), q_arome.max())
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1, label='y=x')

    # QQ-plot
    ax.scatter(q_arome, q_inf, color=cfg['color'], s=20, alpha=0.7)

    ax.set_xlabel(f'AROME — {cfg["label"]}')
    ax.set_ylabel(f'Inférence — {cfg["label"]}')
    ax.set_title(f'QQ-plot {varname}')
    ax.legend()
    ax.set_aspect('equal')

plt.suptitle('QQ-plot inférence vs AROME', fontsize=14)
plt.tight_layout()
plt.savefig(f'{DOSSIER}/qqplot_inf_arome_193K.png', dpi=150, bbox_inches='tight')
plt.close()
print("Sauvegardé : qqplot_inf_arome.png")

# """
# QQ-plot : Inférence vs AROME
# Deux modes :
#   - MODE 1 : QQ sur quantiles spatiaux (à partir de fichiers .npy déjà calculés)
#   - MODE 2 : QQ sur distribution complète flatten (relit les .nc et le .zarr)
# """

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.tri as tri
# import netCDF4 as nc
# import zarr
# import glob
# from tqdm import tqdm

# # ============================================================
# #  CONFIG — à adapter
# # ============================================================

# DOSSIER_QUANTILES = '/project/home/p200177/DE_371/angeliquebonamy/anemoi/inferences/quantiles_runner2'
# NETCDF_DIR = '/project/home/p200177/DE_371/avritj/experiments_anemoi/inference/netcdf_full_training_angelique/'
# ZARR_PATH  = '/project/home/p200177/DE_371/datasets/arome-an-oper-titan-0p025-2020-2024-1h-v4-eurw-precip6/arome-an-oper-titan-0p025-2020-2024-1h-v4-eurw-precip6/arome-an-oper-titan-0p025-2020-2024-1h-v4-eurw-precip6.zarr'


# # Période AROME pour le MODE 2
# DATE_START = np.datetime64('2020-01-01')
# DATE_END   = np.datetime64('2023-01-04')
# N_SAMPLES  = 500
# RANDOM_SEED = 42

# VARS_CONFIG = {
#     '2t':  {'label': '2t (K)',    'color': '#D85A30'},
#     '10u': {'label': '10u (m/s)', 'color': '#378ADD'},
#     '10v': {'label': '10v (m/s)', 'color': '#1D9E75'},
# }

# PERCENTILES = np.arange(1, 100, 1)  # P1 → P99

# # ============================================================
# #  MODE 1 — QQ à partir des fichiers .npy de quantiles spatiaux
# #  (utilise les q10/q50/q90 déjà sauvegardés par variable)
# # ============================================================

# def qq_from_npy(suffix_inf='inference21', suffix_arome='train_arome', output_file='qq_mode1.png'):
#     """
#     Pour chaque variable, charge les .npy de quantiles spatiaux (q10, q50, q90),
#     flatten les valeurs spatiales, calcule les percentiles globaux et trace le QQ.

#     suffix_inf   : suffixe des fichiers inférence  (ex: 'inference21')
#     suffix_arome : suffixe des fichiers AROME       (ex: 'train_arome')
#     """
#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))

#     for ax, (varname, cfg) in zip(axes, VARS_CONFIG.items()):
#         # Charger les 3 quantiles spatiaux et les concaténer
#         # → on obtient une distribution multi-quantile sur tous les pixels
#         q_inf_parts, q_arome_parts = [], []

#         for q_name in ['q10', 'q50', 'q90']:
#             path_inf   = f'{DOSSIER_QUANTILES}/{q_name}_{varname}_{suffix_inf}.npy'
#             path_arome = f'{DOSSIER_QUANTILES}/{q_name}_{varname}_{suffix_arome}.npy'
#             q_inf_parts.append(np.load(path_inf).flatten())
#             q_arome_parts.append(np.load(path_arome).flatten())

#         all_inf   = np.concatenate(q_inf_parts)
#         all_arome = np.concatenate(q_arome_parts)

#         # Percentiles globaux
#         q_inf_vals   = np.nanpercentile(all_inf,   PERCENTILES)
#         q_arome_vals = np.nanpercentile(all_arome, PERCENTILES)

#         _plot_qq(ax, q_arome_vals, q_inf_vals, cfg, varname)

#     plt.suptitle('QQ-plot inférence vs AROME (Mode 1 — quantiles spatiaux .npy)', fontsize=14)
#     plt.tight_layout()
#     out = f'{DOSSIER_QUANTILES}/{output_file}'
#     plt.savefig(out, dpi=150, bbox_inches='tight')
#     plt.close()
#     print(f"Sauvegardé : {out}")


# # ============================================================
# #  MODE 2 — QQ sur distribution complète (relit .nc + .zarr)
# # ============================================================

# def qq_from_raw(output_file='qq_mode2.png'):
#     """
#     Relit tous les fichiers .nc d'inférence et tire N_SAMPLES dates dans le .zarr AROME.
#     Flatten tout et calcule les percentiles sur la distribution complète.
#     """

#     # --- Sélection des dates AROME ---
#     ds_zarr   = zarr.open(ZARR_PATH, mode='r')
#     variables = list(ds_zarr.attrs['variables'])
#     dates     = ds_zarr['dates'][:]
#     mask      = (dates >= DATE_START) & (dates <= DATE_END)
#     indices_disponibles = np.where(mask)[0]
#     n = min(N_SAMPLES, len(indices_disponibles))
#     np.random.seed(RANDOM_SEED)
#     indices_dates = np.sort(np.random.choice(indices_disponibles, size=n, replace=False))
#     print(f"{n} dates AROME sélectionnées")

#     # --- Fichiers inférence ---
#     fichiers = sorted(glob.glob(f'{NETCDF_DIR}/*.nc'))
#     print(f"{len(fichiers)} fichiers .nc trouvés")

#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))

#     for ax, (varname, cfg) in zip(axes, VARS_CONFIG.items()):

#         # Inférence — flatten tous les samples × tous les points
#         data_inf = []
#         for f in tqdm(fichiers, desc=f'Lecture inférence {varname}'):
#             try:
#                 ds = nc.Dataset(f)
#                 data_inf.append(np.array(ds[varname][0, :]))
#                 ds.close()
#             except Exception as e:
#                 print(f"  Fichier ignoré : {f} ({e})")
#         data_inf = np.stack(data_inf, axis=0).flatten()

#         # AROME — flatten tous les samples × tous les points
#         var_idx    = variables.index(varname)
#         data_arome = ds_zarr['data'].oindex[indices_dates, var_idx, 0, :].flatten()

#         # Percentiles
#         q_inf   = np.nanpercentile(data_inf,   PERCENTILES)
#         q_arome = np.nanpercentile(data_arome, PERCENTILES)

#         _plot_qq(ax, q_arome, q_inf, cfg, varname)

#     plt.suptitle('QQ-plot inférence vs AROME (Mode 2 — distribution complète)', fontsize=14)
#     plt.tight_layout()
#     out = './output/qq_plot'  #f'{NETCDF_DIR}/qq_mode2.png'
#     plt.savefig(out, dpi=150, bbox_inches='tight')
#     plt.close()
#     print(f"Sauvegardé : {out}")


# # ============================================================
# #  HELPER — tracé d'un QQ-plot sur un axe matplotlib
# # ============================================================

# def _plot_qq(ax, q_ref, q_pred, cfg, varname):
#     """
#     q_ref  : quantiles de référence (AROME)  — axe X
#     q_pred : quantiles de l'inférence        — axe Y
#     """
#     vmin = min(q_ref.min(), q_pred.min())
#     vmax = max(q_ref.max(), q_pred.max())

#     # Diagonale y = x
#     ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1, label='y=x')

#     # Points QQ colorés par percentile (utile pour voir où le biais est fort)
#     sc = ax.scatter(q_ref, q_pred, c=PERCENTILES, cmap='plasma', s=25, alpha=0.85, zorder=3)
#     plt.colorbar(sc, ax=ax, label='Percentile', shrink=0.8)

#     # Biais moyen
#     bias = np.mean(q_pred - q_ref)
#     ax.set_xlabel(f'AROME — {cfg["label"]}')
#     ax.set_ylabel(f'Inférence — {cfg["label"]}')
#     ax.set_title(f'QQ-plot {varname}\nbiais moyen = {bias:+.3f}')
#     ax.legend(fontsize=8)
#     ax.set_aspect('equal')
#     ax.grid(True, alpha=0.3)


# # ============================================================
# #  MAIN
# # ============================================================

# if __name__ == '__main__':
#     # print("=== MODE 1 : QQ depuis .npy de quantiles spatiaux ===")
#     # qq_from_npy(
#     #     suffix_inf='inference21',
#     #     suffix_arome='train_arome',
#     #     output_file='qq_mode1.png'
#     # )

#     print("\n=== MODE 2 : QQ sur distribution complète ===")
#     qq_from_raw(output_file='qq_mode2.png')
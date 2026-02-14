#!/usr/bin/env python3
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------
# User settings
# ---------------------------
ncfile = "rrfs-3km-subdomain-grid.nc"   # <- change to your file
varname = "dummy"                      # your 2D mask/field

# CONUS-ish map extent (lon_min, lon_max, lat_min, lat_max)
conus_extent = (-130, -65, 22, 52)

# ---------------------------
# Read NetCDF
# ---------------------------
ds = xr.open_dataset(ncfile)

lat1d = ds["lat"].values
lon1d = ds["lon"].values

# Make 2D lon/lat grid (since you have 1D lat and 1D lon)
Lon, Lat = np.meshgrid(lon1d, lat1d)

# Use dummy as a mask/field to show the domain
field = ds[varname].values

# Mask NaNs (FillValue becomes NaN in many readers)
field = np.where(np.isfinite(field), field, np.nan)

# Compute bounding box for an outline
lat_min, lat_max = np.nanmin(lat1d), np.nanmax(lat1d)
lon_min, lon_max = np.nanmin(lon1d), np.nanmax(lon1d)

# ---------------------------
# Plot
# ---------------------------
proj = ccrs.PlateCarree()

fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-97, central_latitude=38))

ax.set_extent(conus_extent, crs=proj)

# Basemap features
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.6)
ax.add_feature(cfeature.STATES, linewidth=0.4)
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="white")

# Overlay your domain as shading
# If dummy is just a placeholder, this still shows the footprint where it's finite.
pm = ax.pcolormesh(
    Lon, Lat, field,
    transform=proj,
    shading="auto",
    alpha=0.5
)

# Add bounding box outline for clarity
box_lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
box_lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
ax.plot(box_lons, box_lats, transform=proj, linewidth=2)

plt.title("RRFS subdomain over CONUS")
plt.colorbar(pm, ax=ax, shrink=0.7, label=varname)
plt.tight_layout()
plt.savefig("hres_domain.png")
#plt.show()


import math
import xarray as xr
import numpy as np

core_grid = "rrfs-3km-subdomain-grid.nc"
out_grid = "rrfs-3km-boundary-20km-grid.nc"

n = math.ceil(20 / 3)  # 7 cells for 20 km at 3 km spacing

ds = xr.open_dataset(core_grid)
lat = ds["lat"].values  # adjust if your vars are named differently
lon = ds["lon"].values

# Assume 2D lat/lon arrays
dlat = lat[1, 0] - lat[0, 0]
dlon = lon[0, 1] - lon[0, 0]

ny, nx = lat.shape
new_lat = np.zeros((ny + 2*n, nx + 2*n))
new_lon = np.zeros((ny + 2*n, nx + 2*n))

# Fill center
new_lat[n:n+ny, n:n+nx] = lat
new_lon[n:n+ny, n:n+nx] = lon

# Extend edges by constant spacing
for i in range(n):
    new_lat[n-1-i, n:n+nx] = lat[0, :] - dlat * (i+1)
    new_lat[n+ny+i, n:n+nx] = lat[-1, :] + dlat * (i+1)
    new_lon[n:n+ny, n-1-i] = lon[:, 0] - dlon * (i+1)
    new_lon[n:n+ny, n+nx+i] = lon[:, -1] + dlon * (i+1)

# Fill corners (simple extension)
new_lat[:n, :n] = new_lat[n, n]
new_lat[:n, -n:] = new_lat[n, -n-1]
new_lat[-n:, :n] = new_lat[-n-1, n]
new_lat[-n:, -n:] = new_lat[-n-1, -n-1]

new_lon[:n, :n] = new_lon[n, n]
new_lon[:n, -n:] = new_lon[n, -n-1]
new_lon[-n:, :n] = new_lon[-n-1, n]
new_lon[-n:, -n:] = new_lon[-n-1, -n-1]

xr.Dataset(
    {"lat": (("y", "x"), new_lat), "lon": (("y", "x"), new_lon)},
).to_netcdf(out_grid)


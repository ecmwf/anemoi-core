#!/usr/bin/env python3
"""
Plot a single GRIB2 field using pygrib.

Examples:
  python plot_grib_field.py /path/to/file.grib2 --shortName t --typeOfLevel isobaricInhPa --level 850 --out t850.png
  python plot_grib_field.py /path/to/file.grib2 --shortName dswrf --typeOfLevel surface --step 0 --out dswrf.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("grib", type=Path, help="GRIB2 file path")
    p.add_argument("--shortName", required=True, help="GRIB shortName (e.g., t, q, u, v, dswrf, lsm, orog)")
    p.add_argument("--typeOfLevel", required=True, help="typeOfLevel (e.g., surface, isobaricInhPa, hybrid)")
    p.add_argument("--level", type=int, default=None, help="Level value (e.g., 850). Omit for surface.")
    p.add_argument("--step", type=int, default=0, help="Forecast step (default 0)")
    p.add_argument("--out", type=Path, default=Path("grib_field.png"), help="Output PNG path")
    p.add_argument("--lat-min", type=float, default=None, help="Min latitude for subdomain")
    p.add_argument("--lat-max", type=float, default=None, help="Max latitude for subdomain")
    p.add_argument("--lon-min", type=float, default=None, help="Min longitude for subdomain")
    p.add_argument("--lon-max", type=float, default=None, help="Max longitude for subdomain")
    args = p.parse_args()

    try:
        import pygrib  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"pygrib is required but not available: {exc}") from exc

    sel_kwargs = {
        "shortName": args.shortName,
        "typeOfLevel": args.typeOfLevel,
        "step": args.step,
    }
    if args.level is not None:
        sel_kwargs["level"] = args.level

    with pygrib.open(str(args.grib)) as grbs:
        msgs = grbs.select(**sel_kwargs)
        if not msgs:
            raise SystemExit(f"No GRIB messages found for {sel_kwargs}")
        grb = msgs[0]

    data = grb.values
    lats, lons = grb.latlons()

    if (
        args.lat_min is not None
        or args.lat_max is not None
        or args.lon_min is not None
        or args.lon_max is not None
    ):
        lat_min = -90.0 if args.lat_min is None else args.lat_min
        lat_max = 90.0 if args.lat_max is None else args.lat_max
        lon_min = -180.0 if args.lon_min is None else args.lon_min
        lon_max = 180.0 if args.lon_max is None else args.lon_max
        mask = (lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max)
        if not np.any(mask):
            raise SystemExit("Subdomain mask is empty. Check lat/lon bounds.")
        data = np.where(mask, data, np.nan)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), layout="tight")
    pcm = ax.pcolormesh(lons, lats, data, shading="auto")
    ax.set_title(f"{grb.name} ({grb.shortName}) {grb.typeOfLevel} level={grb.level} step={grb.step}")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    fig.colorbar(pcm, ax=ax, shrink=0.8)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

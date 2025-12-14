"""
Plot the target domain (and optional boundary buffer) from an anemoi-data YAML config.

Usage:
  python plot_domain.py --config training/docs/user-guide/examples/anemoi-data-lam-example.yaml --output domain.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, MutableMapping

import matplotlib.pyplot as plt
import yaml


def load_config(path: Path) -> MutableMapping[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def rectangle_coords(lat_min: float, lat_max: float, lon_min: float, lon_max: float):
    xs = [lon_min, lon_max, lon_max, lon_min, lon_min]
    ys = [lat_min, lat_min, lat_max, lat_max, lat_min]
    return xs, ys


def plot_domain(cfg: Mapping[str, object], output: Path | None = None):
    dataset = cfg["dataset"]
    domain = dataset["domain"]
    lat = domain["lat"]
    lon = domain["lon"]
    buf = domain.get("boundary_buffer_deg", {})

    lat_min, lat_max = float(lat["min"]), float(lat["max"])
    lon_min, lon_max = float(lon["min"]), float(lon["max"])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Core domain
    xs, ys = rectangle_coords(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)
    ax.plot(xs, ys, color="tab:blue", lw=2, label="Target domain")
    ax.fill(xs, ys, color="tab:blue", alpha=0.1)

    # Boundary buffer (if present)
    if buf:
        lat_buf = float(buf.get("lat", 0.0))
        lon_buf = float(buf.get("lon", 0.0))
        xs_buf, ys_buf = rectangle_coords(
            lat_min=lat_min - lat_buf,
            lat_max=lat_max + lat_buf,
            lon_min=lon_min - lon_buf,
            lon_max=lon_max + lon_buf,
        )
        ax.plot(xs_buf, ys_buf, color="tab:orange", lw=1.5, ls="--", label="Boundary buffer extent")

    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(dataset.get("name", "Domain"))
    ax.grid(True, ls=":", color="0.7")
    ax.legend(loc="best")
    ax.set_aspect("equal")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot domain from an anemoi-data YAML config.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional output image path (e.g., domain.png)."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    plot_domain(cfg, output=args.output)


if __name__ == "__main__":
    main()

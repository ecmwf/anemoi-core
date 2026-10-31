# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Visualise the learned semi-Lagrangian transport inside an ADR processor during a real forecast.

Runs an anemoi-inference ``SimpleRunner`` on ECMWF open data, with an :class:`NSLRecorder`
attached to every semi-Lagrangian layer in the model, then draws the diagnostics. The
reference wind the latent transport is compared against is the pressure-thickness-weighted
column mean of the u/v already present in the input state, mapped onto the processor's hidden
grid - so no second data download is needed.

Usage
-----
    python run_nsl_viz.py <run_id> [--lead-time 72] [--outdir nsl_viz] [--grid O96]
"""

import argparse
import datetime
import logging
import os
from collections import defaultdict

import nsl_plots as P
import numpy as np
from nsl_diagnostics import NSLRecorder
from nsl_diagnostics import column_mean_wind
from nsl_diagnostics import regrid_to_hidden
from persistent_runner import PersistingRunner

LOGGER = logging.getLogger(__name__)

CHECKPOINT = "~/paradis/checkpoint/{run_id}/inference-last.ckpt"
SOURCE = "ecmwf"  # Other options are: "azure", "aws", or "google"

PROBES = {
    "north_atlantic": (45.0, -40.0),  # jet-stream region: expect long, eastward-shifted clouds
    "tropical_pacific": (10.0, 150.0),  # weak mean flow: expect a cloud centred near the point
    "southern_ocean": (-55.0, 60.0),  # strong westerlies
}

PARAMS_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor"]
PARAMS_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]


def describe_forcings(handler):
    """Log how the checkpoint classifies its forcings, which decides what the runner must supply."""
    groups = {
        "constant coupled (carried by the runner)": {"include": ["constant+forcing"], "exclude": ["computed"]},
        "constant computed (lat/lon trig)": {"include": ["computed+constant"]},
        "dynamic coupled (persisted by this script)": {"include": ["forcing"], "exclude": ["computed", "constant"]},
        "dynamic computed (insolation, julian day, ...)": {"include": ["computed"], "exclude": ["constant"]},
    }
    LOGGER.info("Forcings expected by the checkpoint:")
    for title, selection in groups.items():
        names, _ = handler.metadata.select_variables_and_masks(**selection)
        LOGGER.info("  %-46s %s", title + ":", sorted(names) or "-")


def get_open_data(date, param, levelist=(), num_prev_steps=1, grid="O96", **kwargs):
    import earthkit.data as ekd
    import earthkit.regrid as ekr

    fields = defaultdict(list)
    dates = [date - datetime.timedelta(hours=6 * i) for i in range(num_prev_steps)]

    # Get the data for the current date and the previous dates, as the model is initialised
    # with t-6h data as well.
    for d in dates:
        data = ekd.from_source("ecmwf-open-data", date=d, param=param, levelist=list(levelist), source=SOURCE, **kwargs)
        for f in data:  # type: ignore
            # Open data is between -180 and 180, we need to shift it to 0-360
            assert f.to_numpy().shape == (721, 1440)
            values = np.roll(f.to_numpy(), -f.shape[1] // 2, axis=1)
            # Interpolate from 0.25 degrees to the model's input grid
            values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": grid})
            name = f"{f.metadata('param')}_{f.metadata('levelist')}" if levelist else f.metadata("param")
            fields[name].append(values)

    return {param: np.stack(values) for param, values in fields.items()}


def load_input_state(date, grid="O96", num_prev_steps=2, cache="input_state_nsl.npz"):
    """Load the input state for the given date, optionally using a cache."""
    if cache and os.path.exists(cache):
        LOGGER.info("Reusing cached input state %s", cache)
        return dict(date=date, fields=dict(np.load(cache)))

    LOGGER.info("Loading input state for date %s. Cache file (%s) not found", date, cache)
    fields = {}
    common = dict(date=date, grid=grid, num_prev_steps=num_prev_steps)

    fields.update(get_open_data(param=PARAMS_SFC, levtype="sfc", **common))
    missing = set(PARAMS_SFC) - set(fields)
    assert not missing, f"Missing surface parameters: {missing}"

    fields.update(get_open_data(param=PARAMS_PL, levelist=LEVELS, **common))
    expected = {f"{p}_{lev}" for p in PARAMS_PL for lev in LEVELS}
    missing = expected - set(fields) - {"q_10"}  # q at 10 hPa is not used by the model
    assert not missing, f"Missing pressure-level parameters: {missing}"

    fields.pop("q_10", None)

    # Transform geopotential height to geopotential
    for level in LEVELS:
        fields[f"z_{level}"] = fields.pop(f"gh_{level}") * 9.80665

    if cache:
        np.savez(cache, **fields)

    return dict(date=date, fields=fields)


def reference_column_wind(input_state, latitudes, longitudes, hidden_lat_deg, hidden_lon_deg):
    """Column-mean u/v from the input state, on the processor's hidden grid."""
    if latitudes is None or longitudes is None:
        raise ValueError(
            "The checkpoint metadata carries no input grid, so the reference wind cannot be placed "
            "on the hidden grid. Pass the grid explicitly, or run with era5_uv=None.",
        )

    fields = input_state["fields"]
    levels = sorted(
        int(name.split("_")[1]) for name in fields if name.startswith("u_") and f"v_{name.split('_')[1]}" in fields
    )
    if not levels:
        raise ValueError("The input state has no u_<level>/v_<level> pairs to build a reference wind from.")

    # The last time index is the analysis the forecast starts from.
    u = np.stack([np.asarray(fields[f"u_{lev}"])[-1] for lev in levels])
    v = np.stack([np.asarray(fields[f"v_{lev}"])[-1] for lev in levels])
    ubar, vbar = column_mean_wind(u, v, levels)

    return (
        regrid_to_hidden(ubar, latitudes, longitudes, hidden_lat_deg, hidden_lon_deg),
        regrid_to_hidden(vbar, latitudes, longitudes, hidden_lat_deg, hidden_lon_deg),
    )


def main(run_id, lead_time=12, outdir="nsl_viz", grid="O96", date=None, cache="input_state_nsl_{grid}.npz"):
    """Run the NSL visualization for a given run ID and lead time."""
    import earthkit.data as ekd
    from ecmwf.opendata import Client as OpendataClient

    from anemoi.inference.outputs.printer import print_state

    ekd.config.set({"cache-policy": "user"})

    date = date or OpendataClient(SOURCE).latest()
    LOGGER.info("Initial date is %s", date)

    os.makedirs(outdir, exist_ok=True)
    out = lambda name: os.path.join(outdir, name)  # noqa: E731

    runner = PersistingRunner(CHECKPOINT.format(run_id=run_id))

    input_state = load_input_state(date, grid=grid, cache=cache.format(grid=grid.lower()))

    handler = next(iter(runner.tensor_handlers.values()))
    step_hours = runner.checkpoint.timestep.total_seconds() / 3600
    describe_forcings(handler)

    # Fix: Snapshot the analysis values before `run` adds computed forcings to the input state, so the
    # dynamic coupled forcings can be held at those values for the whole rollout.
    runner.persisted_forcings = {
        name: np.asarray(values)[-1] if np.asarray(values).ndim > 1 else np.asarray(values)
        for name, values in input_state["fields"].items()
    }

    with NSLRecorder(runner.model, probes=PROBES) as rec:
        for state in runner.run(input_states=input_state, lead_time=lead_time):
            print_state(state)

    if not rec.diags:
        raise RuntimeError("The recorder captured nothing; the processor never called its diag_hook.")

    lat, lon = rec.lat_deg, rec.lon_deg
    num_layers = 1 + max(d.layer for d in rec.diags)
    LOGGER.info(
        "Recorded %d layers on a %dx%d hidden grid over %d steps",
        num_layers,
        rec.nlat,
        rec.nlon,
        1 + max(d.step for d in rec.diags),
    )

    LOGGER.info("Wrote diagnostics to %s", os.path.abspath(outdir))

    # Latent and physical winds
    layers = sorted(range(num_layers))
    ubar, vbar = reference_column_wind(input_state, handler.metadata.latitudes, handler.metadata.longitudes, lat, lon)
    P.plot_latent_winds(rec.diags, lat, lon, era5_uv=(ubar, vbar), layers=layers, path=out("01_latent_winds.png"))
    P.plot_physical_winds(lat, lon, era5_uv=(ubar, vbar), path=out("01_physical_winds.png"))

    # Departure clouds for each probe location
    for name, latlon in PROBES.items():
        i, j = rec.probe_index(name)
        P.plot_departure_cloud(rec.diags, name, latlon, ubar=(ubar[i, j], vbar[i, j]), path=out(f"02_cloud_{name}.png"))

    # Mode statistics
    P.plot_mode_statistics(rec.diags, path=out("03_mode_stats.png"))

    # Plot trajectories
    seeds = [(la, lo) for la in range(-60, 61, 15) for lo in range(0, 360, 20)]
    P.plot_trajectories(P.back_trajectories(rec.diags, lat, lon, seeds), path=out("04_trajectories.png"))

    # Plot fold maps
    for layer in (0, num_layers // 2, num_layers - 1):
        P.plot_fold_map(rec.diags, layer=layer, lat_deg=lat, lon_deg=lon, path=out(f"05_folds_layer{layer + 1}.png"))

    # Plot animation GIF of the latent winds over time
    if lead_time > step_hours:
        for layer in (0, num_layers // 2, num_layers - 1):
            P.animate_latent_speed(
                rec.diags,
                layer=layer,
                lat_deg=lat,
                lon_deg=lon,
                step_hours=step_hours,
                path=out(f"06_rollout_layer{layer + 1}.gif"),
            )

    LOGGER.info("Wrote diagnostics to %s", os.path.abspath(outdir))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="Training run id under the checkpoint directory")
    parser.add_argument("--lead-time", type=int, default=72, help="Forecast lead time in hours")
    parser.add_argument("--outdir", default="nsl_viz", help="Directory for the figures")
    parser.add_argument("--grid", default="O96", help="Model input grid, e.g. O96 or N320")
    parser.add_argument("--no-cache", action="store_true", help="Always re-download the input state")
    args = parser.parse_args()

    main(
        args.run_id,
        lead_time=args.lead_time,
        outdir=args.outdir,
        grid=args.grid,
        cache=None if args.no_cache else "input_state_nsl_{grid}.npz",
    )

# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Compare predictions against a gridded reference dataset (e.g. ERA5) on fixed validation dates.

Sparse observation targets cannot show what a model does between stations. This callback runs
the model on a few fixed validation samples, looks up the matching valid times in a reference
anemoi dataset on the same grid, and reports difference maps, power spectra and scalar scores.
The reference is used for evaluation only; it never enters the loss.
"""

from __future__ import annotations

import datetime
import logging
import re
import warnings
from typing import TYPE_CHECKING
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from pytorch_lightning.utilities import rank_zero_only
from scipy.spatial import Delaunay

from anemoi.training.diagnostics.callbacks.plot import BasePerEpochPlotCallback
from anemoi.training.diagnostics.evaluation.geospatial.projections import MapProjection
from anemoi.training.diagnostics.evaluation.plotting.sample import single_plot
from anemoi.training.diagnostics.evaluation.plotting.settings import LAYOUT
from anemoi.training.diagnostics.evaluation.plotting.spectrum import compute_spectra
from anemoi.training.diagnostics.evaluation.zonal import ZonalGrid
from anemoi.training.diagnostics.evaluation.zonal import hadley
from anemoi.training.diagnostics.evaluation.zonal import jet
from anemoi.training.diagnostics.evaluation.zonal import peak
from anemoi.training.diagnostics.evaluation.zonal import section_rmse
from anemoi.training.diagnostics.evaluation.zonal import streamfunction
from anemoi.training.diagnostics.evaluation.zonal import thermal_wind
from anemoi.training.diagnostics.evaluation.zonal import zonal_moments

if TYPE_CHECKING:
    import pytorch_lightning as pl

    from anemoi.training.diagnostics.callbacks.plot import PlottingSettings

LOGGER = logging.getLogger(__name__)

# Latitude bands for the optional band scores: [lat_min, lat_max) in degrees, the pole included.
BANDS = {"nh": (20.0, 90.0), "tropics": (-20.0, 20.0), "sh": (-90.0, -20.0)}
SCORES = ("rmse", "bias", "highk_ratio")
ZONAL_FLAGS = ("sections", "rmse_sections", "thermal_wind", "streamfunction", "eddy_fluxes")
ZONAL_DEFAULTS = {
    **dict.fromkeys(ZONAL_FLAGS, True),
    "mask_below_ground": True,
    "tw_layer": (850, 250),
    "vt_level": 850,
    "uv_level": 250,
}
LEVEL_VARIABLE = re.compile(r"^([uvt])_(\d+)$")


def band_masks(lat: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean mask per latitude band (see ``BANDS``)."""
    return {name: (lat >= lo) & ((lat < hi) | (hi >= 90.0)) for name, (lo, hi) in BANDS.items()}


class SpectralGrid:
    """Linear interpolation from scattered nodes onto one regular lat-lon grid, triangulated once.

    All fields share the same interpolation, so interpolation artefacts cancel when their spectra
    are compared. Nodes within 30 deg of the dateline are duplicated at +-360 deg so it is
    interpolated rather than filled. The Delaunay triangulation and the barycentric weights of
    every target point are computed once; interpolating a field is then a cheap gather.

    Parameters
    ----------
    lat, lon : np.ndarray
        Node coordinates in degrees, shape (grid,).
    """

    def __init__(self, lat: np.ndarray, lon: np.ndarray) -> None:
        lon = np.mod(np.asarray(lon, dtype=np.float64), 360.0)
        lat = np.asarray(lat, dtype=np.float64)
        # One output row per input latitude row (192 for O96), capped for scattered grids.
        n_lat = min(len(np.unique(np.round(lat, 6))), 2 * int(np.sqrt(len(lat) / 2)))
        n_lon = 2 * n_lat - 1  # Gauss-Legendre layout expected by compute_spectra
        mesh_lon, mesh_lat = np.meshgrid(
            np.linspace(0.0, 360.0, n_lon, endpoint=False),
            np.linspace(lat.max(), lat.min(), n_lat),
        )
        self.shape = mesh_lon.shape

        east, west = np.nonzero(lon > 330.0)[0], np.nonzero(lon < 30.0)[0]
        self._source = np.concatenate((np.arange(len(lon)), east, west))
        points = np.column_stack(
            (np.concatenate((lon, lon[east] - 360.0, lon[west] + 360.0)), lat[self._source]),
        )
        tri = Delaunay(points)
        targets = np.column_stack((mesh_lon.ravel(), mesh_lat.ravel()))
        simplex = tri.find_simplex(targets)
        self._inside = simplex >= 0
        simplex = np.where(self._inside, simplex, 0)
        transform = tri.transform[simplex]
        bary = np.einsum("ijk,ik->ij", transform[:, :2], targets - transform[:, 2])
        self._weights = np.column_stack((bary, 1.0 - bary.sum(axis=1)))
        self._vertices = tri.simplices[simplex]

    def interpolate(self, field: np.ndarray) -> np.ndarray:
        """Return ``field`` on the regular grid; points outside the hull get the field mean."""
        values = np.asarray(field, dtype=np.float64)[self._source]
        regular = np.einsum("ij,ij->i", values[self._vertices], self._weights)
        regular[~self._inside] = float(np.mean(field))
        return regular.reshape(self.shape)

    def spectra(self, fields: list[np.ndarray]) -> list[np.ndarray]:
        """Return the power per total wavenumber of each field."""
        return [np.asarray(compute_spectra(self.interpolate(field))) for field in fields]


def regular_grid_spectra(lat: np.ndarray, lon: np.ndarray, fields: list[np.ndarray]) -> list[np.ndarray]:
    """Spectra of ``fields`` after interpolation to a shared regular grid (see :class:`SpectralGrid`).

    Parameters
    ----------
    lat, lon : np.ndarray
        Node coordinates in degrees, shape (grid,).
    fields : list[np.ndarray]
        Fields of shape (grid,) to transform.

    Returns
    -------
    list[np.ndarray]
        Power per total wavenumber for each field.
    """
    return SpectralGrid(lat, lon).spectra(fields)


class ReferenceComparisonPlot(BasePerEpochPlotCallback):
    """Plot and score predictions against a reference dataset (e.g. ERA5) on fixed dates.

    For every configured ``date`` (the sample anchor, i.e. the offset-0 time of the
    validation sample) the model is run in validation mode on rank 0. For every selected
    task step and variable the callback then:

    - plots prediction, reference and prediction minus reference maps, and overlays the power
      spectra of prediction, reference and their difference;
    - logs the ``scores`` averaged over the dates: ``val_ref_rmse`` (default), and on request
      ``val_ref_bias`` and ``val_ref_highk_ratio`` (mean ratio of predicted to reference power
      above ``high_k``).

    ``score_dates`` adds dates that are scored but not plotted (and get no spectra), so the
    RMSE and bias can be averaged over many more cases than are worth a figure. With
    ``mean_maps: true`` one ``ref_mean_{step}`` figure shows, per variable, the bias and RMSE
    maps over all dates (plotted and score-only) and their zonal means against latitude, so
    the per-date ``ref_{date}`` figures can be cut to one date showing a typical forecast, or
    turned off with ``dates: []`` (then nothing gets spectra and ``highk_ratio`` is not logged). With
    ``bands: true`` the RMSE and bias are also logged per latitude band as
    ``val_ref_rmse/{nh,tropics,sh}/{var}/{step}`` (hard edges at +-20 deg), for
    ``band_variables`` only when given (default: every variable). All means are
    unweighted over grid points, which on an O96 grid is close to area weighting and equals the
    loss weighting when the graph ``area_weight`` is ``UniformWeights``.

    ``zonal`` adds zonal-mean circulation diagnostics (see
    :mod:`anemoi.training.diagnostics.evaluation.zonal`) from every ``u_*``, ``v_*`` and ``t_*``
    pressure level shared by the model outputs and the reference. Zonal moments are averaged
    over all dates (plotted and score-only), plotted in one ``zonal_{step}`` figure and scored
    as ``val_ref_zonal/{name}/{step}`` (errors are pred - ref, ratios pred / ref). Each flag
    adds a figure row and its scalars:

    - ``sections``: [u], [T] latitude-pressure sections; ``{u,t}_rmse`` (cos-lat weighted) and
      ``jet_{nh,sh}_{lat,speed}_err`` from [u] at ``uv_level``;
    - ``rmse_sections``: zonal mean of the point-wise RMSE of u, v and T over the dates per
      level, so where (latitude, height) the forecasts err, not only how wrong the mean
      circulation is; ``{u,v,t}_point_rmse`` (cos-lat weighted, pooled over levels);
    - ``thermal_wind``: shear across ``tw_layer`` against thermal-wind balance (|lat| >= 15);
      ``tw_residual_rmse`` of the prediction;
    - ``streamfunction``: mean meridional mass streamfunction; ``hadley_{nh,sh}_strength_ratio``
      and ``hadley_{nh,sh}_edge_err`` at 500 hPa (or the nearest level);
    - ``eddy_fluxes``: [v'T'] at ``vt_level`` and [u'v'] at ``uv_level``;
      ``vt{level}_{nh,sh}_ratio`` and ``uv{level}_{nh,sh}_ratio`` of the 25-75 deg peaks.

    ``mask_below_ground`` (default on) drops points where the level pressure exceeds the
    reference ``sp``, for both prediction and reference.

    The model forward runs synchronously; only figure rendering goes through the (possibly
    asynchronous) plot executor. The forward runs on rank 0 alone, so the callback is skipped
    when the model is sharded (``num_gpus_per_model > 1``).

    Example
    -------
    ```yaml
    - _target_: anemoi.training.diagnostics.callbacks.reference.ReferenceComparisonPlot
      reference_dataset: /path/to/era5-o96-2025.zarr   # or an open_dataset config dict
      variables: {z_500: z_500, t_850: t_850, q_700: q_700, u_250: u_250}
      dates: ["2025-01-15T00:00", "2025-03-01T12:00"]
      steps: [dacycle3, rstep0]                         # labels from task.get_metric_name
      score_dates: ["2025-01-02T06:00", ...]            # optional: scored, not plotted
      mean_maps: true                                   # optional: bias/RMSE maps over all dates
      bands: true                                       # optional: NH / Tropics / SH scores
      band_variables: [z_500, t_850]                    # optional: subset scored per band
      scores: [rmse]                                    # optional: add bias, highk_ratio
      zonal:                                            # optional: zonal-mean dynamics
        sections: true
        thermal_wind: true
        streamfunction: true
        eddy_fluxes: true
      every_n_epochs: 1
    ```
    """

    def __init__(
        self,
        reference_dataset: str | dict,
        variables: list[str] | dict[str, str],
        dates: list[str],
        steps: list[str] | None = None,
        every_n_epochs: int | None = None,
        dataset_name: str = "data",
        high_k: int = 60,
        grid_tolerance_deg: float = 1e-3,
        plotting_settings: PlottingSettings | None = None,
        score_dates: list[str] | None = None,
        bands: bool = False,
        band_variables: list[str] | None = None,
        scores: list[str] | None = None,
        zonal: dict | None = None,
        mean_maps: bool = False,
    ) -> None:
        super().__init__(
            every_n_epochs=every_n_epochs,
            dataset_names=[dataset_name],
            plotting_settings=plotting_settings,
        )
        self.reference_config = reference_dataset
        self.variables = dict(variables) if isinstance(variables, dict) else {v: v for v in variables}
        self.dates = [np.datetime64(d, "s") for d in dates]
        plotted = set(self.dates)
        self.score_dates = [d for d in (np.datetime64(d, "s") for d in score_dates or []) if d not in plotted]
        self.bands = bands
        self.mean_maps = mean_maps
        self.band_variables = list(self.variables) if band_variables is None else list(band_variables)
        unknown = sorted(set(self.band_variables) - set(self.variables))
        if unknown:
            msg = f"ReferenceComparisonPlot: band_variables {unknown} are not in variables {list(self.variables)}."
            raise ValueError(msg)
        if band_variables is not None and not bands:
            LOGGER.warning("ReferenceComparisonPlot: band_variables is ignored because bands is false.")
        self.score_names = ["rmse"] if scores is None else list(scores)
        unknown = sorted(set(self.score_names) - set(SCORES))
        if unknown:
            msg = f"ReferenceComparisonPlot: unknown scores {unknown}, expected a subset of {list(SCORES)}."
            raise ValueError(msg)
        if not self.dates and "highk_ratio" in self.score_names:
            LOGGER.warning("ReferenceComparisonPlot: highk_ratio needs plotted dates; none given, so it is not logged.")
        self.zonal = None
        if zonal is not None:
            unknown = sorted(set(zonal) - set(ZONAL_DEFAULTS))
            if unknown:
                msg = f"ReferenceComparisonPlot: unknown zonal options {unknown}, expected {list(ZONAL_DEFAULTS)}."
                raise ValueError(msg)
            self.zonal = {**ZONAL_DEFAULTS, **zonal}
            self.zonal["tw_layer"] = tuple(sorted(self.zonal["tw_layer"], reverse=True))
            if not any(self.zonal[flag] for flag in ZONAL_FLAGS):
                self.zonal = None
        self.steps = [s.lstrip("_") for s in steps] if steps is not None else None
        self.dataset_name = dataset_name
        self.high_k = high_k
        self.grid_tolerance_deg = grid_tolerance_deg
        self._reference = None
        self._grid_checked = False
        self._spectral_grid: SpectralGrid | None = None
        self._zonal_grid: ZonalGrid | None = None

    @property
    def reference(self) -> Any:
        """Open the reference dataset lazily (only on the rank that uses it)."""
        if self._reference is None:
            from anemoi.datasets import open_dataset

            self._reference = open_dataset(self.reference_config)
            missing = sorted(set(self.variables.values()) - set(self._reference.name_to_index))
            if missing:
                msg = f"ReferenceComparisonPlot: variables {missing} not in the reference dataset."
                raise ValueError(msg)
        return self._reference

    def _check_grid(self, lat: np.ndarray, lon: np.ndarray) -> None:
        if self._grid_checked:
            return
        ref_lat = np.asarray(self.reference.latitudes)
        ref_lon = np.asarray(self.reference.longitudes)
        if ref_lat.shape != lat.shape:
            msg = f"ReferenceComparisonPlot: reference grid has {ref_lat.size} points, model grid {lat.size}."
            raise ValueError(msg)
        dlon = np.abs((ref_lon - lon + 180.0) % 360.0 - 180.0)
        if np.abs(ref_lat - lat).max() > self.grid_tolerance_deg or dlon.max() > self.grid_tolerance_deg:
            msg = "ReferenceComparisonPlot: reference grid does not match the model grid (no regridding is done)."
            raise ValueError(msg)
        self._grid_checked = True

    def _reference_row(self, valid_time: np.datetime64, variables: list[str]) -> dict[str, np.ndarray]:
        """Read ``variables`` at ``valid_time`` with a single read of the date (one chunk per date)."""
        dates = np.asarray(self.reference.dates).astype("datetime64[s]")
        hits = np.nonzero(dates == valid_time)[0]
        if len(hits) == 0:
            msg = f"ReferenceComparisonPlot: {valid_time} not in the reference dataset."
            raise ValueError(msg)
        row = np.asarray(self.reference[int(hits[0])])
        name_to_index = self.reference.name_to_index
        return {var: np.asarray(row[name_to_index[var], 0, :], dtype=np.float64) for var in variables}

    def _zonal_levels(self, out_index: dict[str, int]) -> np.ndarray:
        """Pressure levels (hPa, top first) with u, v and t in both the model outputs and the reference."""
        found: dict[str, set[int]] = {"u": set(), "v": set(), "t": set()}
        for name in out_index:
            match = LEVEL_VARIABLE.match(name)
            if match and name in self.reference.name_to_index:
                found[match.group(1)].add(int(match.group(2)))
        levels = np.asarray(sorted(found["u"] & found["v"] & found["t"]), dtype=np.float64)
        needed = set(self.zonal["tw_layer"]) | {self.zonal["vt_level"], self.zonal["uv_level"]}
        missing = sorted(needed - set(levels.astype(int)))
        if len(levels) < 2 or missing:
            msg = (
                f"ReferenceComparisonPlot: zonal diagnostics need u, v, t on levels {sorted(needed)} in both the "
                f"model outputs and the reference; shared levels are {levels.astype(int).tolist()}."
            )
            raise ValueError(msg)
        if self.zonal["mask_below_ground"] and "sp" not in self.reference.name_to_index:
            msg = "ReferenceComparisonPlot: zonal mask_below_ground needs 'sp' in the reference dataset."
            raise ValueError(msg)
        return levels

    @staticmethod
    def _anchor_index(dataset: Any, reader: Any, date: np.datetime64) -> int:
        positions = np.asarray(dataset.anchors)[:, 1]
        anchor_dates = np.asarray(reader.dates)[positions].astype("datetime64[s]")
        hits = np.nonzero(anchor_dates == date)[0]
        if len(hits) == 0:
            msg = f"ReferenceComparisonPlot: {date} is not a valid validation sample anchor."
            raise ValueError(msg)
        return int(hits[0])

    def compute(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> dict:
        """Run the model on the fixed dates and collect prediction/reference pairs.

        Parameters
        ----------
        trainer : pl.Trainer
            Trainer whose datamodule provides the validation dataset.
        pl_module : pl.LightningModule
            Training module used to run the model.

        Returns
        -------
        dict
            ``{"lat", "lon", "cases": [{"date", "step", "valid_time", "fields": {var: (pred, ref)}}]}``,
            plus ``"zonal": {step: summary}`` when ``zonal`` is set (see :meth:`_zonal_summary`) and
            ``"mean_maps"`` when ``mean_maps`` is set (see :meth:`_mean_maps`).
        """
        name = self.dataset_name
        dataset = trainer.datamodule.ds_valid
        reader = dataset.data_readers[name]
        task = pl_module.task

        latlons = np.rad2deg(pl_module.model.model._graph_data[name].x.detach().cpu().numpy())
        lat, lon = latlons[:, 0], latlons[:, 1]
        self._check_grid(lat, lon)

        out_index = pl_module.data_indices[name].model.output.name_to_index
        missing = sorted(set(self.variables) - set(out_index))
        if missing:
            msg = f"ReferenceComparisonPlot: variables {missing} are not model outputs."
            raise ValueError(msg)

        task_steps = list(task.steps("validation"))
        labels = [task.get_metric_name(**step).lstrip("_") for step in task_steps]
        selected = self.steps if self.steps is not None else labels[-1:]
        unknown = sorted(set(selected) - set(labels))
        if unknown:
            msg = f"ReferenceComparisonPlot: steps {unknown} not in the validation task steps {labels}."
            raise ValueError(msg)

        levels, zonal_vars = self._prepare_zonal(out_index, lat)
        ref_vars = sorted(set(self.variables.values()) | set(zonal_vars))

        post_processors = pl_module.model.post_processors[name]
        cases = []
        plotted = set(self.dates)
        for date in self.dates + self.score_dates:
            index = self._anchor_index(dataset, reader, date)
            sample = dataset.get_sample(index)
            batch = {key: value.unsqueeze(0) for key, value in sample.items()}
            batch = pl_module.transfer_batch_to_device(batch, pl_module.device)
            batch = pl_module.on_after_batch_transfer(batch, 0)
            # Callback hooks run outside Lightning's autocast, so re-enter the trainer's precision
            # context explicitly (flash-attention rejects fp32 inputs under bf16/16-mixed).
            with torch.no_grad(), trainer.precision_plugin.forward_context():
                output = pl_module._step(batch, validation_mode=True)

            for step_kwargs, label, prediction in zip(task_steps, labels, output.predictions, strict=True):
                if label not in selected:
                    continue
                offset = task.get_output_offsets(**step_kwargs)[-1]
                valid_time = date + np.timedelta64(int(offset / datetime.timedelta(seconds=1)), "s")
                pred = post_processors(prediction[name].detach(), in_place=False)[0, -1, 0].float().cpu().numpy()
                ref = self._reference_row(valid_time, ref_vars)
                fields = {
                    var: (pred[:, out_index[var]].astype(np.float64), ref[ref_var])
                    for var, ref_var in self.variables.items()
                }
                case = {
                    "date": date,
                    "step": label,
                    "valid_time": valid_time,
                    "fields": fields,
                    "plot": date in plotted,
                }
                if levels is not None:
                    case["zonal"] = self._case_moments(levels, pred, out_index, ref)
                cases.append(case)

        # All numerics stay here, on the training thread, done once: the plot executor thread only
        # draws. SHTOOLS/FFTW planning and the shared BLAS pools are not safe to run concurrently on
        # a background thread while the training loop forks dataloader workers each epoch (that
        # combination hung debug jobs 33953861/33956258 after the first epoch's figures).
        self._attach_spectra(cases, lat, lon)
        result = {"lat": lat, "lon": lon, "cases": cases}
        if self.mean_maps:
            result["mean_maps"] = self._mean_maps(cases, lat, selected)
        return self._attach_zonal(result, levels, selected)

    def _attach_spectra(self, cases: list[dict], lat: np.ndarray, lon: np.ndarray) -> None:
        """Add pred, ref and pred - ref spectra of every variable to the plotted cases."""
        if self._spectral_grid is None:
            self._spectral_grid = SpectralGrid(lat, lon)
        for case in cases:
            if case["plot"]:
                case["spectra"] = {
                    var: self._spectral_grid.spectra([pred, ref, pred - ref])
                    for var, (pred, ref) in case["fields"].items()
                }

    def _mean_maps(self, cases: list[dict], lat: np.ndarray, steps: list[str]) -> dict:
        """Bias and RMSE maps over all dates per step and variable, plus their zonal means.

        Returns ``{step: {var: {"bias", "rmse", "lat_bias", "lat_rmse", "n_dates"}}}``.
        """
        if self._zonal_grid is None:
            self._zonal_grid = ZonalGrid(lat)
        maps = {}
        for step in steps:
            step_cases = [case for case in cases if case["step"] == step]
            maps[step] = {}
            for var in self.variables:
                diff = np.stack([pred - ref for pred, ref in (case["fields"][var] for case in step_cases)])
                bias, mse = diff.mean(axis=0), (diff**2).mean(axis=0)
                maps[step][var] = {
                    "bias": bias,
                    "rmse": np.sqrt(mse),
                    "lat_bias": self._zonal_grid.mean(bias)[0],
                    "lat_rmse": np.sqrt(self._zonal_grid.mean(mse)[0]),
                    "n_dates": len(step_cases),
                }
        return maps

    def _attach_zonal(self, result: dict, levels: np.ndarray | None, steps: list[str]) -> dict:
        """Replace the per-case zonal moments by one date-averaged summary per step."""
        if levels is not None:
            result["zonal"] = {
                step: self._zonal_summary(
                    levels,
                    [case.pop("zonal") for case in result["cases"] if case["step"] == step],
                )
                for step in steps
            }
        return result

    def _prepare_zonal(self, out_index: dict[str, int], lat: np.ndarray) -> tuple[np.ndarray | None, list[str]]:
        """Zonal levels (None when off) and the extra reference variables they need."""
        if self.zonal is None:
            return None, []
        levels = self._zonal_levels(out_index)
        if self._zonal_grid is None:
            self._zonal_grid = ZonalGrid(lat)
        return levels, self._zonal_variables(levels) + (["sp"] if self.zonal["mask_below_ground"] else [])

    @staticmethod
    def _zonal_variables(levels: np.ndarray) -> list[str]:
        return [f"{var}_{int(level)}" for var in "uvt" for level in levels]

    def _case_moments(self, levels: np.ndarray, pred: np.ndarray, out_index: dict[str, int], ref: dict) -> dict:
        """Zonal moments of one case for pred and ref; the full level fields are not kept."""
        mask = 100.0 * levels[:, None] < ref["sp"][None] if self.zonal["mask_below_ground"] else None
        sources = {
            "pred": {var: pred[:, out_index[var]].astype(np.float64) for var in self._zonal_variables(levels)},
            "ref": ref,
        }
        stacked = {
            side: {var: np.stack([source[f"{var}_{int(level)}"] for level in levels]) for var in "uvt"}
            for side, source in sources.items()
        }
        moments = {side: zonal_moments(self._zonal_grid, *fields.values(), mask) for side, fields in stacked.items()}
        if self.zonal["rmse_sections"]:
            moments["err"] = {
                var: self._zonal_grid.mean((stacked["pred"][var] - stacked["ref"][var]) ** 2, mask) for var in "uvt"
            }
        return moments

    def _zonal_summary(self, levels: np.ndarray, moments: list[dict]) -> dict:
        """Average the per-date zonal moments and derive balance, streamfunction and scalars.

        Returns ``{"lat", "levels", "pred": {...}, "ref": {...}}`` where each side holds the mean
        moments (``u, v, t, vt, uv, frac``), ``tw`` (thermal-wind ``actual``/``balanced``), ``psi``
        and the scalars ``jet``, ``hadley``, ``vt_peak``, ``uv_peak``. With ``rmse_sections``,
        ``"err": {"u", "v", "t"}`` holds the zonal-mean point-wise RMSE over the dates, (level, n_lat).
        """
        lat = self._zonal_grid.lat
        index = {int(level): i for i, level in enumerate(levels)}
        psi_level = index[min(index, key=lambda p: abs(p - 500))]
        summary = {"lat": lat, "levels": levels}
        for side in ("pred", "ref"):
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows below ground
                mean = {key: np.nanmean([m[side][key] for m in moments], axis=0) for key in moments[0][side]}
            mean["tw"] = thermal_wind(lat, levels, mean["t"], mean["u"], self.zonal["tw_layer"])
            mean["psi"] = streamfunction(lat, levels, mean["v"], mean["frac"])
            mean["jet"] = jet(lat, mean["u"][index[self.zonal["uv_level"]]])
            mean["hadley"] = hadley(lat, mean["psi"][psi_level])
            mean["vt_peak"] = peak(lat, mean["vt"][index[self.zonal["vt_level"]]])
            mean["uv_peak"] = peak(lat, mean["uv"][index[self.zonal["uv_level"]]])
            summary[side] = mean
        if self.zonal["rmse_sections"]:
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                summary["err"] = {var: np.sqrt(np.nanmean([m["err"][var] for m in moments], axis=0)) for var in "uvt"}
        return summary

    def scores(self, result: dict) -> dict[str, float]:
        """Return the enabled ``scores`` per variable and step, plus the zonal scalars.

        The high-k ratio uses the plotted dates only; RMSE and bias use all dates, and per band
        for ``band_variables`` when ``bands`` is set.
        """
        masks = band_masks(np.asarray(result["lat"])) if self.bands else {}
        sums: dict[str, list[float]] = {}

        def add(key: str, value: float) -> None:
            sums.setdefault(key, []).append(value)

        moments = {
            "rmse": lambda diff: float(np.sqrt(np.mean(diff**2))),
            "bias": lambda diff: float(np.mean(diff)),
        }
        moments = {name: fn for name, fn in moments.items() if name in self.score_names}
        for case in result["cases"]:
            step = case["step"]
            for var, (pred, ref) in case["fields"].items():
                diff = pred - ref
                for metric, fn in moments.items():
                    add(f"val_ref_{metric}/{var}/{step}", fn(diff))
                    for band, mask in masks.items() if var in self.band_variables else ():
                        add(f"val_ref_{metric}/{band}/{var}/{step}", fn(diff[mask]))
                if "highk_ratio" in self.score_names and "spectra" in case:
                    p_pred, p_ref, _ = case["spectra"][var]
                    k = slice(min(self.high_k, len(p_ref) - 1), None)
                    ratio = float(np.mean(p_pred[k] / np.maximum(p_ref[k], np.finfo(float).tiny)))
                    add(f"val_ref_highk_ratio/{var}/{step}", ratio)
        scores = {key: float(np.mean(values)) for key, values in sums.items()}
        for step, summary in result.get("zonal", {}).items():
            scores.update({f"val_ref_zonal/{key}/{step}": value for key, value in self.zonal_scores(summary).items()})
        return scores

    def zonal_scores(self, summary: dict) -> dict[str, float]:
        """Scalars of one step's zonal summary for the enabled ``zonal`` flags (pred - ref or pred / ref)."""
        lat, pred, ref = summary["lat"], summary["pred"], summary["ref"]
        flags = self.zonal
        out = _hemisphere_scores(pred, ref, flags)
        if flags["sections"]:
            out["u_rmse"] = section_rmse(lat, pred["u"] - ref["u"])
            out["t_rmse"] = section_rmse(lat, pred["t"] - ref["t"])
        if flags["rmse_sections"]:
            out.update({f"{var}_point_rmse": section_rmse(lat, rmse) for var, rmse in summary["err"].items()})
        if flags["thermal_wind"]:
            residual = pred["tw"]["actual"] - pred["tw"]["balanced"]
            out["tw_residual_rmse"] = section_rmse(lat, residual[None])
        return {key: float(value) for key, value in out.items()}

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, **kwargs) -> None:
        del kwargs
        if trainer.sanity_checking or trainer.current_epoch % self.every_n_epochs != 0:
            return
        if getattr(pl_module, "model_comm_group_size", 1) > 1:
            LOGGER.warning("ReferenceComparisonPlot: skipped, the model is sharded across GPUs.")
            return

        result = self.compute(trainer, pl_module)
        if trainer.logger is not None:
            trainer.logger.log_metrics(self.scores(result), step=trainer.global_step)
        self.plot(trainer, pl_module, self.dataset_names, epoch=trainer.current_epoch, result=result)

    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        dataset_names: list[str],
        epoch: int,
        result: dict,
    ) -> None:
        """Render one figure per date: a row per (variable, step), columns pred | ref | diff | spectra."""
        del pl_module, dataset_names
        lat, lon = result["lat"], result["lon"]
        # Project like the other plot callbacks, so the coastline overlay lines up with the data.
        x, y = MapProjection.equirectangular().project(np.stack([lat, lon], axis=1))
        datashader = self.plotting_settings.datashader

        for date in self.dates:
            date_cases = [case for case in result["cases"] if case["date"] == date]
            rows = [(var, case) for var in self.variables for case in date_cases]
            fig, axes = plt.subplots(len(rows), 4, figsize=(20, 3.2 * len(rows)), layout=LAYOUT, squeeze=False)
            for row, (var, case) in enumerate(rows):
                pred, ref = case["fields"][var]
                diff = pred - ref
                vmin, vmax = np.nanmin([pred.min(), ref.min()]), np.nanmax([pred.max(), ref.max()])
                dmax = float(np.nanmax(np.abs(diff))) or 1.0
                norm = Normalize(vmin, vmax)
                label = f"{var} {case['step']} ({case['valid_time']})"
                single_plot(fig, axes[row, 0], x, y, pred, norm=norm, title=f"pred {label}", datashader=datashader)
                single_plot(fig, axes[row, 1], x, y, ref, norm=norm, title=f"reference {var}", datashader=datashader)
                single_plot(
                    fig,
                    axes[row, 2],
                    x,
                    y,
                    diff,
                    cmap="bwr",
                    norm=Normalize(-dmax, dmax),
                    title=f"pred - ref (rmse {np.sqrt(np.mean(diff**2)):.3g})",
                    datashader=datashader,
                )
                spectra = case["spectra"][var]
                ax = axes[row, 3]
                for spectrum, name in zip(spectra, ("pred", "reference", "pred - ref"), strict=True):
                    ax.loglog(np.arange(1, len(spectrum)), spectrum[1:], label=name)
                ax.axvline(self.high_k, color="grey", lw=0.5, ls="--")
                ax.set_xlabel("$k$")
                ax.set_ylabel("$P(k)$")
                ax.legend()
                ax.set_title(f"spectra {var} {case['step']}")
            tag = f"ref_{np.datetime_as_string(date, unit='h')}".replace(":", "")
            self._output_figure(trainer.logger, fig, epoch=epoch, tag=tag, exp_log_tag="val_ref")

        for step, maps in result.get("mean_maps", {}).items():
            fig = self._plot_mean_maps(step, maps, x, y, lat_rows=self._zonal_grid.lat)
            self._output_figure(trainer.logger, fig, epoch=epoch, tag=f"ref_mean_{step}", exp_log_tag="val_ref")

        for step, summary in result.get("zonal", {}).items():
            fig = self._plot_zonal(step, summary, n_dates=len(self.dates) + len(self.score_dates))
            self._output_figure(trainer.logger, fig, epoch=epoch, tag=f"zonal_{step}", exp_log_tag="val_ref")

    def _plot_mean_maps(self, step: str, maps: dict, x: np.ndarray, y: np.ndarray, lat_rows: np.ndarray) -> plt.Figure:
        """One row per variable: bias map | RMSE map | zonal-mean bias and RMSE against latitude."""
        datashader = self.plotting_settings.datashader
        fig, axes = plt.subplots(len(maps), 3, figsize=(16, 3.2 * len(maps)), layout=LAYOUT, squeeze=False)
        n_dates = 0
        for row, (var, stats) in enumerate(maps.items()):
            n_dates = stats["n_dates"]
            bmax = float(np.nanmax(np.abs(stats["bias"]))) or 1.0
            single_plot(
                fig,
                axes[row, 0],
                x,
                y,
                stats["bias"],
                cmap="bwr",
                norm=Normalize(-bmax, bmax),
                title=f"bias {var} {step} (mean {np.mean(stats['bias']):.3g})",
                datashader=datashader,
            )
            single_plot(
                fig,
                axes[row, 1],
                x,
                y,
                stats["rmse"],
                cmap="viridis",
                norm=Normalize(0.0, float(np.nanmax(stats["rmse"])) or 1.0),
                title=f"rmse {var} {step} (pooled {np.sqrt(np.mean(stats['rmse'] ** 2)):.3g})",
                datashader=datashader,
            )
            ax = axes[row, 2]
            ax.plot(lat_rows, stats["lat_rmse"], color="C0", label="rmse")
            ax.plot(lat_rows, stats["lat_bias"], color="C3", label="bias")
            for edge in (-20.0, 20.0):
                ax.axvline(edge, color="grey", lw=0.5, ls="--")
            ax.axhline(0.0, color="grey", lw=0.5)
            ax.set_xlabel("latitude")
            ax.set_title(f"zonal mean {var} {step}")
            ax.legend(fontsize="small")
        fig.suptitle(f"pred - reference, {step}, over {n_dates} dates")
        return fig

    def _plot_zonal(self, step: str, summary: dict, n_dates: int) -> plt.Figure:
        """Latitude-pressure sections (pred | ref | pred - ref) and latitude profiles of one step."""
        flags = self.zonal
        sections = []
        if flags["sections"]:
            sections += [("[u] (m/s)", "u", 1.0), ("[T] (K)", "t", 1.0)]
        if flags["streamfunction"]:
            sections.append(("psi (1e10 kg/s)", "psi", 1e-10))
        profiles = (["tw"] if flags["thermal_wind"] else []) + (["vt", "uv"] if flags["eddy_fluxes"] else [])
        profiles += ["jet"] if flags["sections"] else []

        n_rows = len(sections) + flags["rmse_sections"] + bool(profiles)
        fig, axes = plt.subplots(n_rows, 4, figsize=(22, 3.6 * n_rows), layout=LAYOUT, squeeze=False)
        for row, (title, key, scale) in enumerate(sections):
            _section_row(fig, axes[row], summary, key, title, scale)
        if flags["rmse_sections"]:
            _rmse_row(fig, axes[len(sections)], summary)
        if profiles:
            for ax, kind in zip(axes[-1], profiles, strict=False):
                _profile_panel(ax, kind, summary, flags)
            for ax in axes[-1][len(profiles) :]:
                ax.axis("off")
        fig.suptitle(f"zonal means {step}, mean over {n_dates} dates")
        return fig


def _hemisphere_scores(pred: dict, ref: dict, flags: dict) -> dict[str, float]:
    """Per-hemisphere jet, Hadley-cell and eddy-flux peak scalars for the enabled ``flags``."""

    def ratio(a: float, b: float) -> float:
        return a / b if b != 0.0 else float("nan")

    out = {}
    for hemi in ("nh", "sh"):
        if flags["sections"]:
            out[f"jet_{hemi}_lat_err"] = pred["jet"][hemi][0] - ref["jet"][hemi][0]
            out[f"jet_{hemi}_speed_err"] = pred["jet"][hemi][1] - ref["jet"][hemi][1]
        if flags["streamfunction"]:
            out[f"hadley_{hemi}_strength_ratio"] = ratio(pred["hadley"][hemi][0], ref["hadley"][hemi][0])
            out[f"hadley_{hemi}_edge_err"] = pred["hadley"][hemi][1] - ref["hadley"][hemi][1]
        if flags["eddy_fluxes"]:
            out[f"vt{flags['vt_level']}_{hemi}_ratio"] = ratio(pred["vt_peak"][hemi], ref["vt_peak"][hemi])
            out[f"uv{flags['uv_level']}_{hemi}_ratio"] = ratio(pred["uv_peak"][hemi], ref["uv_peak"][hemi])
    return out


def _section_row(fig: plt.Figure, axes: np.ndarray, summary: dict, key: str, title: str, scale: float) -> None:
    """Draw pred | ref | pred - ref latitude-pressure sections of ``key`` on the first three axes."""
    lat, levels = summary["lat"], summary["levels"]
    pred, ref = scale * summary["pred"][key], scale * summary["ref"][key]
    if key == "psi":
        vmax = float(np.nanmax(np.abs([pred, ref]))) or 1.0
        vmin, cmap = -vmax, "RdBu_r"
    else:
        vmin, vmax = float(np.nanmin([pred, ref])), float(np.nanmax([pred, ref]))
        vmax, cmap = max(vmax, vmin + 1e-6), "viridis"
    diff = pred - ref
    dmax = float(np.nanmax(np.abs(diff))) or 1.0
    nice = MaxNLocator(nbins=20, symmetric=key == "psi").tick_values(vmin, vmax)
    panels = (
        (pred, f"pred {title}", nice, cmap),
        (ref, f"reference {title}", nice, cmap),
        (
            diff,
            f"pred - ref (rmse {section_rmse(lat, diff):.3g})",
            MaxNLocator(nbins=20, symmetric=True).tick_values(-dmax, dmax),
            "RdBu_r",
        ),
    )
    for ax, (field, label, contour_levels, colormap) in zip(axes, panels, strict=False):
        image = ax.contourf(lat, levels, np.ma.masked_invalid(field), levels=contour_levels, cmap=colormap)
        fig.colorbar(image, ax=ax)
        ax.set_yscale("log")
        ax.set_ylim(levels.max(), levels.min())
        ax.set_yticks(levels, [str(int(level)) for level in levels])
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlabel("latitude")
        ax.set_ylabel("hPa")
        ax.set_title(label)
    axes[3].axis("off")


def _rmse_row(fig: plt.Figure, axes: np.ndarray, summary: dict) -> None:
    """Draw the zonal-mean point-wise RMSE sections of u, v and T on the first three axes."""
    lat, levels = summary["lat"], summary["levels"]
    for ax, (var, unit) in zip(axes, (("u", "m/s"), ("v", "m/s"), ("t", "K")), strict=False):
        rmse = summary["err"][var]
        vmax = float(np.nanmax(rmse)) or 1.0
        contour_levels = MaxNLocator(nbins=20).tick_values(0.0, vmax)
        image = ax.contourf(lat, levels, np.ma.masked_invalid(rmse), levels=contour_levels, cmap="magma_r")
        fig.colorbar(image, ax=ax)
        ax.set_yscale("log")
        ax.set_ylim(levels.max(), levels.min())
        ax.set_yticks(levels, [str(int(level)) for level in levels])
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlabel("latitude")
        ax.set_ylabel("hPa")
        ax.set_title(f"rmse {var.upper() if var == 't' else var} ({unit}), pooled {section_rmse(lat, rmse):.3g}")
    axes[3].axis("off")


def _profile_panel(ax: plt.Axes, kind: str, summary: dict, flags: dict) -> None:
    """Draw one latitude profile (pred and ref): thermal-wind balance, an eddy flux or the jet."""
    lat, levels = summary["lat"], summary["levels"]
    sides = (("pred", summary["pred"], "C0"), ("ref", summary["ref"], "C1"))
    if kind == "tw":
        bottom, top = flags["tw_layer"]
        for side, data, colour in sides:
            residual = section_rmse(lat, (data["tw"]["actual"] - data["tw"]["balanced"])[None])
            ax.plot(lat, data["tw"]["actual"], color=colour, label=f"{side} [u]{top}-[u]{bottom}")
            ax.plot(lat, data["tw"]["balanced"], color=colour, ls="--", label=f"{side} balanced ({residual:.2g})")
        ax.set_title(f"shear {bottom}-{top} hPa vs thermal wind (m/s)")
    else:
        key, level, title = {
            "vt": ("vt", flags["vt_level"], "[v'T'] (K m/s)"),
            "uv": ("uv", flags["uv_level"], "[u'v'] (m2/s2)"),
            "jet": ("u", flags["uv_level"], "[u] (m/s)"),
        }[kind]
        i = int(np.nonzero(levels == level)[0][0])
        for side, data, colour in sides:
            ax.plot(lat, data[key][i], color=colour, label=side)
        ax.set_title(f"{title} at {level} hPa")
    ax.axhline(0.0, color="grey", lw=0.5)
    ax.set_xlabel("latitude")
    ax.legend(fontsize="small")

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
from typing import TYPE_CHECKING
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from pytorch_lightning.utilities import rank_zero_only
from scipy.spatial import Delaunay

from anemoi.training.diagnostics.callbacks.plot import BasePerEpochPlotCallback
from anemoi.training.diagnostics.evaluation.geospatial.projections import MapProjection
from anemoi.training.diagnostics.evaluation.plotting.sample import single_plot
from anemoi.training.diagnostics.evaluation.plotting.settings import LAYOUT
from anemoi.training.diagnostics.evaluation.plotting.spectrum import compute_spectra

if TYPE_CHECKING:
    import pytorch_lightning as pl

    from anemoi.training.diagnostics.callbacks.plot import PlottingSettings

LOGGER = logging.getLogger(__name__)

# Latitude bands for the optional band scores: [lat_min, lat_max) in degrees, the pole included.
BANDS = {"nh": (20.0, 90.0), "tropics": (-20.0, 20.0), "sh": (-90.0, -20.0)}


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
    - logs ``val_ref_rmse``, ``val_ref_bias`` and ``val_ref_highk_ratio`` (mean ratio of
      predicted to reference power above ``high_k``) averaged over the dates.

    ``score_dates`` adds dates that are scored but not plotted (and get no spectra), so the
    RMSE and bias can be averaged over many more cases than are worth a figure. With
    ``bands: true`` the RMSE and bias are also logged per latitude band as
    ``val_ref_rmse/{nh,tropics,sh}/{var}/{step}`` (hard edges at +-20 deg). All means are
    unweighted over grid points, which on an O96 grid is close to area weighting and equals the
    loss weighting when the graph ``area_weight`` is ``UniformWeights``.

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
      bands: true                                       # optional: NH / Tropics / SH scores
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
        self.steps = [s.lstrip("_") for s in steps] if steps is not None else None
        self.dataset_name = dataset_name
        self.high_k = high_k
        self.grid_tolerance_deg = grid_tolerance_deg
        self._reference = None
        self._grid_checked = False
        self._spectral_grid: SpectralGrid | None = None

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

    def _reference_field(self, valid_time: np.datetime64, variable: str) -> np.ndarray:
        dates = np.asarray(self.reference.dates).astype("datetime64[s]")
        hits = np.nonzero(dates == valid_time)[0]
        if len(hits) == 0:
            msg = f"ReferenceComparisonPlot: {valid_time} not in the reference dataset."
            raise ValueError(msg)
        return np.asarray(self.reference[int(hits[0])][self.reference.name_to_index[variable], 0, :], dtype=np.float64)

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
            ``{"lat", "lon", "cases": [{"date", "step", "valid_time", "fields": {var: (pred, ref)}}]}``.
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
                fields = {
                    var: (pred[:, out_index[var]].astype(np.float64), self._reference_field(valid_time, ref_var))
                    for var, ref_var in self.variables.items()
                }
                cases.append(
                    {
                        "date": date,
                        "step": label,
                        "valid_time": valid_time,
                        "fields": fields,
                        "plot": date in plotted,
                    },
                )

        # All numerics stay here, on the training thread, done once: the plot executor thread only
        # draws. SHTOOLS/FFTW planning and the shared BLAS pools are not safe to run concurrently on
        # a background thread while the training loop forks dataloader workers each epoch (that
        # combination hung debug jobs 33953861/33956258 after the first epoch's figures).
        if self._spectral_grid is None:
            self._spectral_grid = SpectralGrid(lat, lon)
        for case in cases:
            if not case["plot"]:
                continue
            case["spectra"] = {
                var: self._spectral_grid.spectra([pred, ref, pred - ref]) for var, (pred, ref) in case["fields"].items()
            }
        return {"lat": lat, "lon": lon, "cases": cases}

    def scores(self, result: dict) -> dict[str, float]:
        """Return date-averaged RMSE, bias and high-k spectral ratio per variable and step.

        The high-k ratio uses the plotted dates only; RMSE and bias use all dates, and per band
        when ``bands`` is set.
        """
        masks = band_masks(np.asarray(result["lat"])) if self.bands else {}
        sums: dict[str, list[float]] = {}

        def add(key: str, value: float) -> None:
            sums.setdefault(key, []).append(value)

        for case in result["cases"]:
            step = case["step"]
            for var, (pred, ref) in case["fields"].items():
                diff = pred - ref
                add(f"val_ref_rmse/{var}/{step}", float(np.sqrt(np.mean(diff**2))))
                add(f"val_ref_bias/{var}/{step}", float(np.mean(diff)))
                for band, mask in masks.items():
                    add(f"val_ref_rmse/{band}/{var}/{step}", float(np.sqrt(np.mean(diff[mask] ** 2))))
                    add(f"val_ref_bias/{band}/{var}/{step}", float(np.mean(diff[mask])))
                if "spectra" in case:
                    p_pred, p_ref, _ = case["spectra"][var]
                    k = slice(min(self.high_k, len(p_ref) - 1), None)
                    ratio = float(np.mean(p_pred[k] / np.maximum(p_ref[k], np.finfo(float).tiny)))
                    add(f"val_ref_highk_ratio/{var}/{step}", ratio)
        return {key: float(np.mean(values)) for key, values in sums.items()}

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

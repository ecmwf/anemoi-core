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
from scipy.interpolate import griddata

from anemoi.training.diagnostics.callbacks.plot import BasePerEpochPlotCallback
from anemoi.training.diagnostics.evaluation.plotting.sample import single_plot
from anemoi.training.diagnostics.evaluation.plotting.settings import LAYOUT
from anemoi.training.diagnostics.evaluation.plotting.spectrum import compute_spectra

if TYPE_CHECKING:
    import pytorch_lightning as pl

    from anemoi.training.diagnostics.callbacks.plot import PlottingSettings

LOGGER = logging.getLogger(__name__)


def regular_grid_spectra(lat: np.ndarray, lon: np.ndarray, fields: list[np.ndarray]) -> list[np.ndarray]:
    """Interpolate fields from scattered nodes to one regular lat-lon grid and return their spectra.

    All fields share the same interpolation, so interpolation artefacts cancel when the spectra
    are compared with each other. Nodes are padded by +-360 deg in longitude so the dateline is
    interpolated rather than filled.

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
    lon = np.mod(lon, 360.0)
    # One output row per input latitude row (192 for O96), capped for scattered grids.
    n_lat = min(len(np.unique(np.round(lat, 6))), 2 * int(np.sqrt(len(lat) / 2)))
    n_lon = 2 * n_lat - 1  # Gauss-Legendre layout expected by compute_spectra
    grid_lat = np.linspace(lat.max(), lat.min(), n_lat)
    grid_lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)
    mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)

    pad_lon = np.concatenate((lon - 360.0, lon, lon + 360.0))
    pad_lat = np.concatenate((lat, lat, lat))
    spectra = []
    for field in fields:
        values = np.concatenate((field, field, field))
        regular = griddata((pad_lon, pad_lat), values, (mesh_lon, mesh_lat), method="linear")
        regular = np.nan_to_num(regular, nan=float(np.nanmean(field)))
        spectra.append(np.asarray(compute_spectra(regular)))
    return spectra


class ReferenceComparisonPlot(BasePerEpochPlotCallback):
    """Plot and score predictions against a reference dataset (e.g. ERA5) on fixed dates.

    For every configured ``date`` (the sample anchor, i.e. the offset-0 time of the
    validation sample) the model is run in validation mode on rank 0. For every selected
    task step and variable the callback then:

    - plots prediction, reference and prediction minus reference maps, and overlays the power
      spectra of prediction, reference and their difference;
    - logs ``val_ref_rmse``, ``val_ref_bias`` and ``val_ref_highk_ratio`` (mean ratio of
      predicted to reference power above ``high_k``) averaged over the dates.

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
    ) -> None:
        super().__init__(
            every_n_epochs=every_n_epochs,
            dataset_names=[dataset_name],
            plotting_settings=plotting_settings,
        )
        self.reference_config = reference_dataset
        self.variables = dict(variables) if isinstance(variables, dict) else {v: v for v in variables}
        self.dates = [np.datetime64(d, "s") for d in dates]
        self.steps = [s.lstrip("_") for s in steps] if steps is not None else None
        self.dataset_name = dataset_name
        self.high_k = high_k
        self.grid_tolerance_deg = grid_tolerance_deg
        self._reference = None
        self._grid_checked = False

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
        for date in self.dates:
            index = self._anchor_index(dataset, reader, date)
            sample = dataset.get_sample(index)
            batch = {key: value.unsqueeze(0) for key, value in sample.items()}
            batch = pl_module.transfer_batch_to_device(batch, pl_module.device)
            batch = pl_module.on_after_batch_transfer(batch, 0)
            with torch.no_grad():
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
                cases.append({"date": date, "step": label, "valid_time": valid_time, "fields": fields})
        return {"lat": lat, "lon": lon, "cases": cases}

    def scores(self, result: dict) -> dict[str, float]:
        """Return date-averaged RMSE, bias and high-k spectral ratio per variable and step."""
        sums: dict[str, list[float]] = {}
        for case in result["cases"]:
            for var, (pred, ref) in case["fields"].items():
                diff = pred - ref
                p_pred, p_ref = regular_grid_spectra(result["lat"], result["lon"], [pred, ref])
                k = slice(min(self.high_k, len(p_ref) - 1), None)
                ratio = float(np.mean(p_pred[k] / np.maximum(p_ref[k], np.finfo(float).tiny)))
                for metric, value in (
                    ("rmse", float(np.sqrt(np.mean(diff**2)))),
                    ("bias", float(np.mean(diff))),
                    ("highk_ratio", ratio),
                ):
                    sums.setdefault(f"val_ref_{metric}/{var}/{case['step']}", []).append(value)
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
        del pl_module, dataset_names
        lat, lon = result["lat"], result["lon"]
        plot_lon = np.where(lon > 180.0, lon - 360.0, lon)
        datashader = self.plotting_settings.datashader

        for date in self.dates:
            date_cases = [case for case in result["cases"] if case["date"] == date]
            n_rows = len(date_cases)
            for var in self.variables:
                fig, axes = plt.subplots(n_rows, 4, figsize=(20, 3.6 * n_rows), layout=LAYOUT, squeeze=False)
                for row, case in enumerate(date_cases):
                    pred, ref = case["fields"][var]
                    diff = pred - ref
                    vmin, vmax = np.nanmin([pred.min(), ref.min()]), np.nanmax([pred.max(), ref.max()])
                    dmax = float(np.nanmax(np.abs(diff))) or 1.0
                    title = f"{var} {case['step']} valid {case['valid_time']}"
                    norm = Normalize(vmin, vmax)
                    single_plot(
                        fig,
                        axes[row, 0],
                        plot_lon,
                        lat,
                        pred,
                        norm=norm,
                        title=f"pred {title}",
                        datashader=datashader,
                    )
                    single_plot(
                        fig,
                        axes[row, 1],
                        plot_lon,
                        lat,
                        ref,
                        norm=norm,
                        title="reference",
                        datashader=datashader,
                    )
                    single_plot(
                        fig,
                        axes[row, 2],
                        plot_lon,
                        lat,
                        diff,
                        cmap="bwr",
                        norm=Normalize(-dmax, dmax),
                        title=f"pred - ref (rmse {np.sqrt(np.mean(diff**2)):.3g})",
                        datashader=datashader,
                    )
                    spectra = regular_grid_spectra(lat, lon, [pred, ref, diff])
                    ax = axes[row, 3]
                    for spectrum, label in zip(spectra, ("pred", "reference", "pred - ref"), strict=True):
                        ax.loglog(np.arange(1, len(spectrum)), spectrum[1:], label=label)
                    ax.axvline(self.high_k, color="grey", lw=0.5, ls="--")
                    ax.set_xlabel("$k$")
                    ax.set_ylabel("$P(k)$")
                    ax.legend()
                    ax.set_title(f"spectra {var} {case['step']}")
                tag = f"ref_{var}_{np.datetime_as_string(date, unit='h')}".replace(":", "")
                self._output_figure(trainer.logger, fig, epoch=epoch, tag=tag, exp_log_tag=f"val_ref_{var}")

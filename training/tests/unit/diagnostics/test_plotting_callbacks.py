# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# ruff: noqa: ANN001, ANN201

from collections.abc import Callable
from functools import partial
from typing import Any
from typing import ClassVar
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from anemoi.training.diagnostics.callbacks.plot import LossCurvePlot
from anemoi.training.diagnostics.callbacks.plot import SpatialMapPlot
from anemoi.training.diagnostics.callbacks.plot_adapter import EnsemblePlotAdapterWrapper
from anemoi.training.diagnostics.callbacks.plot_adapter import ForecasterPlotAdapter
from anemoi.training.diagnostics.evaluation.plotting.graph import get_edge_trainable_modules
from anemoi.training.diagnostics.evaluation.plotting.spatial_map import ensemble_plot_fn
from anemoi.training.diagnostics.evaluation.plotting.spatial_map import histogram_plot_fn
from anemoi.training.diagnostics.evaluation.plotting.spatial_map import sample_plot_fn
from anemoi.training.diagnostics.evaluation.plotting.spatial_map import spectrum_plot_fn
from anemoi.training.tasks import Forecaster
from anemoi.training.tasks import TemporalDownscaler
from anemoi.training.train.step_output import TrainingStepOutput
from anemoi.training.utils.masks import NoOutputMask


# --- Legacy-name shims used only by this test module ------------------------
#
# The historic PlotSample/PlotEnsSample/PlotHistogram/PlotSpectrum callback
# classes were consolidated into a single SpatialMapPlot callback + pluggable
# ``plot_fn``. These shim factories rebuild the old-style constructors so
# the existing test bodies keep working without touching every call site.
def _wrap_plot_callback(
    *,
    plot_fn,
    tag_infix,
    with_auxiliary=False,
    plot_fn_kwargs=None,
) -> Callable[..., SpatialMapPlot]:
    """Return a callable that builds a SpatialMapPlot wired to ``plot_fn``.

    ``plot_fn_kwargs`` is a mapping of legacy kwargs → default sentinel that
    the constructor pops from ``**kwargs``, binds into ``plot_fn`` via
    ``functools.partial`` (dropping ``None`` values), and mirrors as
    attributes on the callback so legacy assertions such as ``callback.log_scale``
    keep working.
    """
    plot_fn_kwargs = plot_fn_kwargs or {}

    def _factory(**kwargs) -> SpatialMapPlot:
        bound = {}
        for name, default in plot_fn_kwargs.items():
            if name in kwargs:
                bound[name] = kwargs.pop(name)
            else:
                bound.setdefault(name, default)
        cb = SpatialMapPlot(
            plot_fn=partial(plot_fn, **{k: v for k, v in bound.items() if v is not None}) or plot_fn,
            tag_infix=tag_infix,
            with_auxiliary=with_auxiliary,
            **kwargs,
        )
        for name, value in bound.items():
            setattr(cb, name, value)
        return cb

    return _factory


PlotSample = _wrap_plot_callback(
    plot_fn=sample_plot_fn,
    tag_infix="sample",
    with_auxiliary=True,
    # legacy tests occasionally pass ensemble-only kwargs to PlotSample; accept-and-ignore
    plot_fn_kwargs={"accumulation_levels_plot": None, "plot_members": None},
)
PlotEnsSample = _wrap_plot_callback(
    plot_fn=ensemble_plot_fn,
    tag_infix="ens_sample",
    plot_fn_kwargs={
        "accumulation_levels_plot": None,
        "n_plots_per_sample": 4,
        "plot_members": None,
    },
)
PlotHistogram = _wrap_plot_callback(
    plot_fn=histogram_plot_fn,
    tag_infix="histo",
    plot_fn_kwargs={"log_scale": False, "precip_and_related_fields": None},
)
PlotSpectrum = _wrap_plot_callback(
    plot_fn=spectrum_plot_fn,
    tag_infix="spec",
    plot_fn_kwargs={"min_delta": None},
)

# Suite of Unit Tests for Plotting Callbacks
# ------------------------------------------
# Tests to check PlotHistogram, PlotSpectrum, LossCurvePlot, PlotSample instantiation
# Tests to check PlotHistogram, PlotSpectrum, LossCurvePlot, PlotSample plot methods
# Tests to check plot_loss, plot_histogram, plot_spectrum, plot_predicted_multilevel_flat_sample return a figure


def test_plot_histogram_instantiation():
    """PlotHistogram can be instantiated with parameters."""
    callback = PlotHistogram(
        sample_idx=0,
        parameters=["t2m", "tp", "u10"],
        dataset_names=["data"],
    )
    assert callback.sample_idx == 0
    assert callback.parameters == ["t2m", "tp", "u10"]
    assert callback.log_scale is False


def test_plot_spectrum_instantiation():
    """PlotSpectrum can be instantiated with parameters."""
    callback = PlotSpectrum(
        sample_idx=0,
        parameters=["t2m", "tp"],
        dataset_names=["data"],
    )
    assert callback.sample_idx == 0
    assert callback.parameters == ["t2m", "tp"]
    assert callback.min_delta is None


def test_plot_loss_instantiation():
    """LossCurvePlot can be instantiated with optional parameter_groups."""
    callback = LossCurvePlot(parameter_groups={})
    assert callback.parameter_groups == {}
    assert callback.dataset_names == ["data"]

    callback2 = LossCurvePlot(
        parameter_groups={"group_a": ["t2m", "tp"], "group_b": ["u10", "v10"]},
        dataset_names=["data"],
    )
    assert len(callback2.parameter_groups) == 2
    assert callback2.parameter_groups["group_a"] == ["t2m", "tp"]


def test_graph_trainable_features_plot_handles_noop_processor_graph_provider():
    class DummyModel:
        pass

    class NoOpGraphProvider:
        trainable = None

    model = DummyModel()
    model._graph_name_data = "data"
    model._graph_name_hidden = "hidden"
    model.encoder_graph_provider = None
    model.decoder_graph_provider = None
    model.processor_graph_provider = NoOpGraphProvider()

    edge_modules = get_edge_trainable_modules(model, dataset_name="data")

    assert edge_modules == {}


def test_graph_trainable_features_plot_handles_noop_mapper_graph_providers():
    class NoOpGraphProvider:
        trainable = None

    class DummyModel:
        pass

    model = DummyModel()
    model._graph_name_data = "data"
    model._graph_name_hidden = "hidden"
    model.encoder_graph_provider = NoOpGraphProvider()
    model.decoder_graph_provider = NoOpGraphProvider()
    model.processor_graph_provider = NoOpGraphProvider()

    edge_modules = get_edge_trainable_modules(model, dataset_name="data")

    assert edge_modules == {}


def test_graph_trainable_features_plot_handles_missing_dataset_key_in_provider_map():
    class TrainableTensor:
        trainable = object()

    class TrainableProvider:
        trainable = TrainableTensor()

    class DummyModel:
        pass

    model = DummyModel()
    model._graph_name_data = "data"
    model._graph_name_hidden = "hidden"
    model.encoder_graph_provider = {"other": TrainableProvider()}
    model.decoder_graph_provider = {"other": TrainableProvider()}
    model.processor_graph_provider = None

    edge_modules = get_edge_trainable_modules(model, dataset_name="data")

    assert edge_modules == {}


# ---- Config and mocks for BasePlotAdditionalMetrics.process and task-type tests ----

_PLOT_PROCESS_CONFIG = {
    "system": {"output": {"plots": None}},
    "diagnostics": {
        "plot": {
            "datashader": False,
            "asynchronous": False,
            "frequency": {"batch": 1, "epoch": 1},
        },
    },
    "data": {
        "datasets": {
            "data": {"diagnostic": None},
        },
    },
}


def _make_pl_module_forecaster(
    *,
    n_step_input: int = 1,
    n_step_output: int = 1,
    validation_rollout: int = 2,
    nlatlon: int = 50,
) -> MagicMock:
    """Mock pl_module for forecaster task: output_times as given."""
    pl_module = MagicMock()
    pl_module.local_rank = 0
    pl_module.grid_dim = 3  # latlon dim=2

    # Use Forecaster task
    pl_module.task = Forecaster(
        multistep_input=n_step_input,
        multistep_output=n_step_output,
        timestep="6H",
        validation_rollout=validation_rollout,
        rollout={"start": 1, "epoch_increment": 1, "maximum": validation_rollout},
    )
    pl_module.n_step_input = pl_module.task.num_input_timesteps
    pl_module.n_step_output = pl_module.task.num_output_timesteps
    pl_module.plot_adapter = pl_module.task._plot_adapter

    # Mock data_indices
    # data_indices[dataset_name].data.output.full, model.output.name_to_index for plot_parameters_dict
    data_indices = MagicMock()
    data_indices.data.output.full = slice(None)
    data_indices.model.output.name_to_index = {"a": 0, "b": 1}
    pl_module.data_indices = {"data": data_indices}

    # Mock graph latlons (radians), converted to deg in process
    pl_module.model.model._graph_data = {"data": MagicMock()}
    pl_module.model.model._graph_data["data"].__getitem__ = lambda _self, _k: MagicMock()
    graph_data = pl_module.model.model._graph_data["data"]
    graph_data.__getitem__ = lambda k: torch.zeros(nlatlon, 2) if k == pl_module.model.model._graph_name_data else None

    # Use no-op output_mask
    pl_module.output_mask = {"data": NoOutputMask()}

    return pl_module


def _make_pl_module_temporal_downscaler(*, nlatlon=50) -> MagicMock:
    """Mock pl_module for temporal downscaler."""
    pl_module = MagicMock()
    pl_module.local_rank = 0
    pl_module.grid_dim = 3

    # Use TemporalDownscaler task
    pl_module.task = TemporalDownscaler(input_timestep="6H", output_timestep="3H", output_left_boundary=True)
    pl_module.n_step_input = pl_module.task.num_input_timesteps
    pl_module.n_step_output = pl_module.task.num_output_timesteps
    pl_module.plot_adapter = pl_module.task._plot_adapter

    # Mock data_indices
    data_indices = MagicMock()
    data_indices.data.output.full = slice(None)
    data_indices.model.output.name_to_index = {"a": 0, "b": 1}
    pl_module.data_indices = {"data": data_indices}

    # Mock graph data
    pl_module.model.model._graph_data = {"data": MagicMock()}
    pl_module.model.model._graph_data["data"].__getitem__ = lambda _k: torch.zeros(nlatlon, 2)

    # Use no-op output_mask
    pl_module.output_mask = {"data": NoOutputMask()}

    return pl_module


def _identity_post_processor() -> Callable[[torch.Tensor | Any], torch.Tensor | Any]:
    """Return a callable that returns the input tensor (for shape-preserving mock)."""

    def _call(x, in_place=False) -> torch.Tensor | Any:
        del in_place
        return x.clone() if isinstance(x, torch.Tensor) else x

    return _call


class _IdentityProcessor:
    """Shape-preserving processor with the small API used by plotting callbacks."""

    processors: ClassVar[dict] = {}

    def __call__(self, x, in_place=False) -> torch.Tensor | Any:
        del in_place
        return x.clone() if isinstance(x, torch.Tensor) else x

    def cpu(self) -> "_IdentityProcessor":
        return self


def _step_output(
    predictions: list[dict[str, torch.Tensor]],
    plot_kwargs: dict[str, Any] | None = None,
) -> TrainingStepOutput:
    return TrainingStepOutput(
        loss=torch.tensor(0.0),
        metrics={},
        predictions=predictions,
        plot_kwargs={} if plot_kwargs is None else plot_kwargs,
    )


# ---- BasePlotAdditionalMetrics.process: input/output shapes ----


def test_process_forecaster_output_shapes():
    """BasePlotAdditionalMetrics.process: forecaster task yields expected data and output_tensor shapes."""
    callback = PlotSample(
        sample_idx=0,
        parameters=["a", "b", "c"],
        accumulation_levels_plot=[0.5],
        dataset_names=["data"],
    )
    batch_size, n_ens, nlatlon, nvar = 2, 1, 50, 3
    n_step_input = 1
    n_step_output = 1
    output_times = 2
    total_targets = output_times * n_step_output  # 2
    n_time = 1 + total_targets + 1  # 4 time steps in the batch
    pl_module = _make_pl_module_forecaster(
        n_step_input=n_step_input,
        n_step_output=n_step_output,
        validation_rollout=output_times,
        nlatlon=nlatlon,
    )
    batch = {"data": torch.randn(batch_size, n_time, n_ens, nlatlon, nvar)}
    # each pred[dataset] shape is (bs, n_step_output, ens, latlon, nvar)
    outputs = _step_output(
        [
            {"data": torch.randn(batch_size, n_step_output, n_ens, nlatlon, nvar)},
            {"data": torch.randn(batch_size, n_step_output, n_ens, nlatlon, nvar)},
        ],
    )
    callback.post_processors = {"data": _identity_post_processor()}
    callback.latlons = {"data": np.zeros((nlatlon, 2))}

    data, output_tensor = callback.process(pl_module, "data", outputs, batch)

    # data: one sample from input_tensor (4 time steps); shape (time_steps, n_ens, nlatlon, nvar)
    assert data.shape == (1 + total_targets + 1, n_ens, nlatlon, nvar), data.shape
    # output_tensor: (output_times, n_step_output, n_ens, nlatlon, nvar) after mask
    assert output_tensor.shape == (output_times, n_step_output, n_ens, nlatlon, nvar), output_tensor.shape


def test_plot_sample_uses_auxiliary_output_from_validation_output():
    """PlotSample forwards auxiliary output from validation metadata."""
    callback = PlotSample(
        sample_idx=0,
        parameters=["a", "b"],
        accumulation_levels_plot=[0.5],
        dataset_names=["data"],
    )

    batch_size, n_ens, nlatlon, nvar = 2, 1, 20, 2
    pl_module = _make_pl_module_forecaster(validation_rollout=1, nlatlon=nlatlon)
    pl_module.allgather_batch = lambda tensor, _dataset_name: tensor
    pl_module.model.post_processors = {"data": _IdentityProcessor()}
    conditioned_target = {"data": torch.full((batch_size, 1, n_ens, nlatlon, nvar), 3.0)}

    batch = {"data": torch.randn(batch_size, 3, n_ens, nlatlon, nvar)}
    output = _step_output(
        [{"data": torch.zeros(batch_size, 1, n_ens, nlatlon, nvar)}],
        plot_kwargs={"auxiliary_output": conditioned_target},
    )
    trainer = MagicMock()
    trainer.current_epoch = 0

    callback.plot = MagicMock()
    callback.on_validation_batch_end(trainer, pl_module, output, batch, batch_idx=0)

    plotted_output = callback.plot.call_args.args[3]
    plotted_auxiliary = callback.plot.call_args.kwargs["auxiliary_output"]
    torch.testing.assert_close(plotted_output.predictions[0]["data"], output.predictions[0]["data"])
    torch.testing.assert_close(plotted_auxiliary["data"], conditioned_target["data"])
    assert plotted_output.plot_kwargs == {}


def test_process_time_interpolator_output_shapes():
    """BasePlotAdditionalMetrics.process: time-interpolator task yields expected shapes."""
    callback = PlotSample(
        sample_idx=0,
        parameters=["a", "b"],
        accumulation_levels_plot=[0.5],
        dataset_names=["data"],
    )
    batch_size, n_ens, nlatlon, nvar = 2, 1, 50, 3

    pl_module = _make_pl_module_temporal_downscaler(nlatlon=nlatlon)

    total_targets = pl_module.task.num_output_timesteps  # no n_step_output factor for temporal downscaler
    n_time = 1 + total_targets + 1  # 4 time steps in the batch

    batch = {"data": torch.randn(batch_size, n_time, n_ens, nlatlon, nvar)}
    outputs = _step_output(
        [
            {"data": torch.randn(batch_size, 1, n_ens, nlatlon, nvar)},
            {"data": torch.randn(batch_size, 1, n_ens, nlatlon, nvar)},
        ],
    )
    callback.post_processors = {"data": _identity_post_processor()}
    callback.latlons = {"data": np.zeros((nlatlon, 2))}

    data, output_tensor = callback.process(pl_module, "data", outputs, batch)

    assert data.shape == (1 + total_targets + 1, n_ens, nlatlon, nvar), data.shape
    assert output_tensor.shape == (pl_module.task.num_output_timesteps, 1, n_ens, nlatlon, nvar), output_tensor.shape


def test_process_temporal_downscaler_multi_out_squeeze():
    """BasePlotAdditionalMetrics.process: temporal downscaler multi-out (ndim=5, shape[0]=1) squeezes to 4D."""
    callback = PlotSample(
        sample_idx=0,
        parameters=["a"],
        accumulation_levels_plot=[0.5],
        dataset_names=["data"],
    )
    batch_size, nlatlon, nvar = 2, 50, 3

    pl_module = _make_pl_module_temporal_downscaler(nlatlon=nlatlon)

    sample_idx = 10
    batch = {"data": torch.randn(batch_size, sample_idx, 1, nlatlon, nvar)}
    # Simulate multi-out: each output (1, 1, 1, nlatlon, nvar) so cat gives (2, 1, 1, nlatlon, nvar);
    # after squeeze(0) we get (2, 1, nlatlon, nvar)
    outputs = _step_output(
        [
            {"data": torch.randn(batch_size, 1, 1, nlatlon, nvar)},
            {"data": torch.randn(batch_size, 1, 1, nlatlon, nvar)},
        ],
    )
    callback.post_processors = {"data": _identity_post_processor()}
    callback.latlons = {"data": np.zeros((nlatlon, 2))}

    _, output_tensor = callback.process(pl_module, "data", outputs, batch)

    # output_tensor: (num_output_timesteps, 1, n_ens, nlatlon, nvar) - 5D
    assert output_tensor.ndim == 5, output_tensor.shape
    assert output_tensor.shape == (pl_module.task.num_output_timesteps, 1, 1, nlatlon, nvar), output_tensor.shape


# ---- process() cache ----


def test_process_cache_shared_across_callbacks():
    """A shared processed_cache avoids redundant post-processing across PlotSample, PlotSpectrum, PlotHistogram.

    Verifies:
    - post-processor called once per (dataset, members) pair despite N callbacks
    - cache hit returns the identical tuple object (not a copy)
    - different members values get separate cache entries
    - no cache (None) falls back to recomputing on every call
    """
    batch_size, n_ens, nlatlon, nvar = 2, 1, 50, 3
    pl_module = _make_pl_module_forecaster(nlatlon=nlatlon)

    batch = {"data": torch.randn(batch_size, 4, n_ens, nlatlon, nvar)}
    outputs = _step_output(
        [
            {"data": torch.randn(batch_size, 1, n_ens, nlatlon, nvar)},
            {"data": torch.randn(batch_size, 1, n_ens, nlatlon, nvar)},
        ],
    )

    call_count = 0
    real_processor = _identity_post_processor()

    def counting_processor(x, **kwargs) -> torch.Tensor | Any:
        nonlocal call_count
        call_count += 1
        return real_processor(x, **kwargs)

    shared_post_processors = {"data": counting_processor}
    shared_latlons = {"data": np.zeros((nlatlon, 2))}

    plot_sample = PlotSample(
        sample_idx=0,
        parameters=["a", "b"],
        accumulation_levels_plot=[0.5],
        dataset_names=["data"],
    )
    plot_spectrum = PlotSpectrum(sample_idx=0, parameters=["a", "b"], min_delta=0.0, dataset_names=["data"])
    plot_histogram = PlotHistogram(
        sample_idx=0,
        parameters=["a", "b"],
        precip_and_related_fields=[],
        dataset_names=["data"],
    )

    for cb in (plot_sample, plot_spectrum, plot_histogram):
        cb.post_processors = shared_post_processors
        cb.latlons = shared_latlons

    cache: dict = {}

    # --- shared cache: post-processor must fire only once across all three callbacks ---
    result_sample = plot_sample.process(pl_module, "data", outputs, batch, processed_cache=cache)
    calls_after_first = call_count

    result_spectrum = plot_spectrum.process(pl_module, "data", outputs, batch, processed_cache=cache)
    result_histogram = plot_histogram.process(pl_module, "data", outputs, batch, processed_cache=cache)

    assert (
        call_count == calls_after_first
    ), f"post-processor called {call_count - calls_after_first} extra time(s) on cache hits"
    assert result_sample is result_spectrum is result_histogram, "cache hits must return the identical tuple object"
    assert len(cache) == 1, f"expected 1 cache entry for (dataset, members=0), got {len(cache)}"

    # --- different members value gets a separate entry, not a cache hit ---
    result_all_members = plot_sample.process(pl_module, "data", outputs, batch, members=None, processed_cache=cache)
    assert result_all_members is not result_sample, "different members must not share a cache entry"
    assert len(cache) == 2, f"expected 2 cache entries after adding members=None, got {len(cache)}"

    # --- no cache: recomputes on every call ---
    call_count = 0
    plot_sample.process(pl_module, "data", outputs, batch)
    plot_sample.process(pl_module, "data", outputs, batch)
    assert call_count >= 2, "expected post-processor to be called on each process() call without a cache"


def test_process_cache_ensemble_list_members():
    """process() with members as a list (PlotEnsSample) hashes correctly and hits cache on repeat."""
    batch_size, n_ens, nlatlon, nvar = 2, 1, 50, 3
    pl_module = _make_pl_module_forecaster(nlatlon=nlatlon)

    batch = {"data": torch.randn(batch_size, 4, n_ens, nlatlon, nvar)}
    outputs = _step_output(
        [
            {"data": torch.randn(batch_size, 1, n_ens, nlatlon, nvar)},
            {"data": torch.randn(batch_size, 1, n_ens, nlatlon, nvar)},
        ],
    )

    call_count = 0
    real_processor = _identity_post_processor()

    def counting_processor(x, **kwargs) -> torch.Tensor | Any:
        nonlocal call_count
        call_count += 1
        return real_processor(x, **kwargs)

    plot_ens = PlotEnsSample(
        sample_idx=0,
        parameters=["a", "b"],
        accumulation_levels_plot=[0.5],
        members=[0, 1],
        dataset_names=["data"],
    )
    plot_ens.post_processors = {"data": counting_processor}
    plot_ens.latlons = {"data": np.zeros((nlatlon, 2))}

    cache: dict = {}

    # first call populates the cache
    result_first = plot_ens.process(pl_module, "data", outputs, batch, members=[0, 1], processed_cache=cache)
    assert len(cache) == 1, f"expected 1 cache entry for members=[0, 1], got {len(cache)}"
    calls_after_first = call_count

    # second call with the same list must hit the cache
    result_second = plot_ens.process(pl_module, "data", outputs, batch, members=[0, 1], processed_cache=cache)
    assert call_count == calls_after_first, "post-processor called again despite list-members cache hit"
    assert result_first is result_second, "list-members cache hit must return the identical tuple"

    # a different list gets a separate entry
    result_other = plot_ens.process(pl_module, "data", outputs, batch, members=[0], processed_cache=cache)
    assert result_other is not result_first, "different member lists must not share a cache entry"
    assert len(cache) == 2, f"expected 2 cache entries after adding members=[0], got {len(cache)}"


# ---- LossCurvePlot ----

_PLOT_LOSS_CONFIG = {
    "system": {"output": {"plots": None}},
    "diagnostics": {
        "plot": {
            "datashader": False,
            "asynchronous": False,
            "frequency": {"batch": 1, "epoch": 1},
        },
    },
    "data": {"datasets": {"data": {"diagnostic": None}}},
}


def test_plot_loss_sort_and_color_by_parameter_group_small_list():
    """sort_and_color_by_parameter_group: <=15 params returns identity sort and correct output shapes."""
    from anemoi.training.diagnostics.evaluation.plotting.loss import sort_and_color_by_parameter_group

    parameter_names = ["t2m", "tp", "u10", "v10"]
    sort_idx, colors, xticks, legend_patches = sort_and_color_by_parameter_group(parameter_names, {})

    assert sort_idx.shape == (len(parameter_names),)
    assert np.array_equal(sort_idx, np.arange(len(parameter_names)))
    assert len(colors) == len(parameter_names)
    assert isinstance(xticks, dict)
    assert len(legend_patches) >= 1
    # One patch per unique "group" (here each param is its own group for <=15)
    assert len(legend_patches) == len(parameter_names)


def test_plot_loss_sort_and_color_by_parameter_group_with_groups():
    """sort_and_color_by_parameter_group: with parameter_groups and >15 params returns grouped sort."""
    from anemoi.training.diagnostics.evaluation.plotting.loss import sort_and_color_by_parameter_group

    parameter_groups = {
        "pressure": ["tp", "sp"] + [f"p{i}" for i in range(6)],
        "wind": ["u10", "v10"] + [f"w{i}" for i in range(6)],
    }
    # >15 parameters to trigger the grouping branch (<=15 keeps each param as its own group)
    parameter_names = ["tp", "sp", "p0", "p1", "p2", "p3", "p4", "p5", "u10", "v10", "w0", "w1", "w2", "w3", "w4", "w5"]
    sort_idx, colors, xticks, legend_patches = sort_and_color_by_parameter_group(parameter_names, parameter_groups)

    assert sort_idx.shape == (len(parameter_names),)
    assert len(colors) == len(parameter_names)
    assert isinstance(xticks, dict)
    assert len(legend_patches) == 2  # pressure and wind


def test_plot_loss_temporal_downscaler():
    """LossCurvePlot._plot uses output_times=1 only one figure is produced."""
    from unittest.mock import patch

    from anemoi.training.losses.mse import MSELoss

    callback = LossCurvePlot(parameter_groups={}, dataset_names=["data"])
    callback.latlons = {}

    nvar = 3
    trainer = MagicMock()
    trainer.logger = MagicMock()
    pl_module = MagicMock()
    pl_module.task = TemporalDownscaler(input_timestep="6H", output_timestep="3H", output_left_boundary=True)
    pl_module.n_step_input = pl_module.task.num_input_timesteps
    pl_module.n_step_output = pl_module.task.num_output_timesteps
    pl_module.plot_adapter = pl_module.task._plot_adapter
    pl_module.local_rank = 0
    pl_module.data_indices = {"data": MagicMock()}
    pl_module.data_indices["data"].model.output.name_to_index = {"a": 0, "b": 1, "c": 2}
    pl_module.data_indices["data"].data.output.full = torch.arange(nvar)
    pl_module.model.metadata = {"dataset": {"variables_metadata": None}}
    batch_size, nlatlon = 2, 10
    n_time = 4
    batch = {"data": torch.randn(batch_size, n_time, 1, nlatlon, nvar)}
    outputs = _step_output(
        [{"data": torch.randn(batch_size, 1, 1, nlatlon, nvar)}],
    )
    callback.loss = {"data": MSELoss()}

    with (
        patch.object(callback, "_output_figure") as mock_output_figure,
        patch(
            "anemoi.training.diagnostics.evaluation.plotting.loss.argsort_variablename_variablelevel",
            return_value=np.arange(nvar),
        ),
        patch("anemoi.training.diagnostics.evaluation.plotting.loss.plot_loss", return_value=MagicMock()),
    ):
        callback._plot(
            trainer,
            pl_module,
            ["data"],
            outputs,
            batch,
            batch_idx=0,
            epoch=0,
        )
        # Non-forecaster forces output_times=1, so only one rollout step -> one figure
        assert mock_output_figure.call_count == 1


def test_plot_loss_single_step_transport():
    """LossCurvePlot._plot with a one-step transport model produces one figure."""
    from unittest.mock import patch

    from anemoi.training.losses.mse import MSELoss

    callback = LossCurvePlot(parameter_groups={}, dataset_names=["data"])
    callback.latlons = {}

    nvar = 3
    n_step_input = 1
    n_step_output = 1
    trainer = MagicMock()
    trainer.logger = MagicMock()
    pl_module = MagicMock()
    pl_module.n_step_input = n_step_input
    pl_module.n_step_output = n_step_output
    pl_module.local_rank = 0
    pl_module.plot_adapter = MagicMock()
    pl_module.plot_adapter.loss_plot_times = 1
    pl_module.plot_adapter.get_loss_plot_batch_start = lambda r: n_step_input + r * n_step_output
    pl_module.data_indices = {"data": MagicMock()}
    pl_module.data_indices["data"].model.output.name_to_index = {"a": 0, "b": 1, "c": 2}
    pl_module.data_indices["data"].data.output.full = torch.arange(nvar)
    pl_module.model.metadata = {"dataset": {"variables_metadata": None}}
    batch_size, nlatlon = 2, 10
    n_time = n_step_input + n_step_output + 1
    batch = {"data": torch.randn(batch_size, n_time, 1, nlatlon, nvar)}
    # Single output (no rollout)
    outputs = _step_output(
        [{"data": torch.randn(batch_size, n_step_output, 1, nlatlon, nvar)}],
    )
    callback.loss = {"data": MSELoss()}
    pl_module.task.steps.return_value = [{}]
    pl_module.task.get_targets.return_value = {"data": torch.randn(batch_size, n_step_output, 1, nlatlon, nvar)}
    pl_module.task.get_metric_name.return_value = ""

    with (
        patch.object(callback, "_output_figure") as mock_output_figure,
        patch(
            "anemoi.training.diagnostics.evaluation.plotting.loss.argsort_variablename_variablelevel",
            return_value=np.arange(nvar),
        ),
        patch("anemoi.training.diagnostics.evaluation.plotting.loss.plot_loss", return_value=MagicMock()),
    ):
        callback._plot(
            trainer,
            pl_module,
            ["data"],
            outputs,
            batch,
            batch_idx=0,
            epoch=0,
        )
        # One-step transport models have output_times=1, so one figure.
        assert mock_output_figure.call_count == 1


def test_plot_loss_forecaster():
    """LossCurvePlot._plot uses one figure per rollout step."""
    from unittest.mock import patch

    from anemoi.training.losses.mse import MSELoss

    callback = LossCurvePlot(parameter_groups={}, dataset_names=["data"])
    callback.latlons = {}

    nvar = 3
    output_times = 3
    n_step_input = 1
    n_step_output = 1
    trainer = MagicMock()
    trainer.logger = MagicMock()
    pl_module = MagicMock()
    pl_module.n_step_input = n_step_input
    pl_module.n_step_output = n_step_output
    pl_module.local_rank = 0
    pl_module.plot_adapter = MagicMock()
    pl_module.plot_adapter.loss_plot_times = output_times
    pl_module.plot_adapter.get_loss_plot_batch_start = lambda r: n_step_input + r * n_step_output
    pl_module.data_indices = {"data": MagicMock()}
    pl_module.data_indices["data"].model.output.name_to_index = {"a": 0, "b": 1, "c": 2}
    pl_module.data_indices["data"].data.output.full = torch.arange(nvar)
    pl_module.model.metadata = {"dataset": {"variables_metadata": None}}
    batch_size, nlatlon = 2, 10
    # Batch needs at least n_step_input + output_times * n_step_output time steps
    n_time = n_step_input + output_times * n_step_output + 1
    batch = {"data": torch.randn(batch_size, n_time, 1, nlatlon, nvar)}
    # One prediction per rollout step
    outputs = _step_output(
        [{"data": torch.randn(batch_size, n_step_output, 1, nlatlon, nvar)} for _ in range(output_times)],
    )
    callback.loss = {"data": MSELoss()}
    pl_module.task.steps.return_value = [{"rollout_step": i} for i in range(output_times)]
    pl_module.task.get_targets.return_value = {"data": torch.randn(batch_size, n_step_output, 1, nlatlon, nvar)}
    pl_module.task.get_metric_name.return_value = ""

    with (
        patch.object(callback, "_output_figure") as mock_output_figure,
        patch(
            "anemoi.training.diagnostics.evaluation.plotting.loss.argsort_variablename_variablelevel",
            return_value=np.arange(nvar),
        ),
        patch("anemoi.training.diagnostics.evaluation.plotting.loss.plot_loss", return_value=MagicMock()),
    ):
        callback._plot(
            trainer,
            pl_module,
            ["data"],
            outputs,
            batch,
            batch_idx=0,
            epoch=0,
        )
        # Forecaster keeps output_times, so one figure per rollout step
        assert mock_output_figure.call_count == output_times


def test_plot_loss_accepts_processed_cache_kwarg():
    """LossCurvePlot._plot accepts and ignores processed_cache without error and still produces figures."""
    from unittest.mock import patch

    from anemoi.training.losses.mse import MSELoss

    callback = LossCurvePlot(parameter_groups={}, dataset_names=["data"])
    callback.latlons = {}

    nvar = 3
    output_times = 2
    n_step_input, n_step_output = 1, 1
    trainer = MagicMock()
    trainer.logger = MagicMock()
    pl_module = MagicMock()
    pl_module.n_step_input = n_step_input
    pl_module.n_step_output = n_step_output
    pl_module.local_rank = 0
    pl_module.data_indices = {"data": MagicMock()}
    pl_module.data_indices["data"].model.output.name_to_index = {"a": 0, "b": 1, "c": 2}
    pl_module.data_indices["data"].data.output.full = torch.arange(nvar)
    pl_module.model.metadata = {"dataset": {"variables_metadata": None}}
    batch_size, nlatlon = 2, 10
    batch = {"data": torch.randn(batch_size, n_step_input + output_times * n_step_output + 1, 1, nlatlon, nvar)}
    outputs = _step_output(
        [{"data": torch.randn(batch_size, n_step_output, 1, nlatlon, nvar)} for _ in range(output_times)],
    )
    callback.loss = {"data": MSELoss()}
    pl_module.task.steps.return_value = [{"rollout_step": i} for i in range(output_times)]
    pl_module.task.get_targets.return_value = {"data": torch.randn(batch_size, n_step_output, 1, nlatlon, nvar)}
    pl_module.task.get_metric_name.return_value = ""

    with (
        patch.object(callback, "_output_figure") as mock_output_figure,
        patch(
            "anemoi.training.diagnostics.evaluation.plotting.loss.argsort_variablename_variablelevel",
            return_value=np.arange(nvar),
        ),
        patch("anemoi.training.diagnostics.evaluation.plotting.loss.plot_loss", return_value=MagicMock()),
    ):
        callback._plot(
            trainer,
            pl_module,
            ["data"],
            outputs,
            batch,
            batch_idx=0,
            epoch=0,
            processed_cache={},
        )
        assert mock_output_figure.call_count == output_times


# ---- PlotSpectrum ----


def test_plot_spectrum_temporal_downscaler():
    """PlotSpectrum._plot produces one figure per output_times for temporal downscaler."""
    from unittest.mock import patch

    callback = PlotSpectrum(
        sample_idx=0,
        parameters=["a", "b"],
        dataset_names=["data"],
    )
    nvar = 2
    nlatlon = 20
    pl_module = _make_pl_module_temporal_downscaler(nlatlon=nlatlon)

    callback.post_processors = {"data": _identity_post_processor()}
    callback.latlons = {"data": np.zeros((nlatlon, 2))}
    batch = {"data": torch.randn(2, 10, 1, nlatlon, nvar)}
    outputs = _step_output(
        [
            {"data": torch.randn(2, 1, 1, nlatlon, nvar)},
            {"data": torch.randn(2, 1, 1, nlatlon, nvar)},
        ],
    )
    trainer = MagicMock()
    trainer.logger = MagicMock()

    with (
        patch.object(callback, "_output_figure") as mock_output_figure,
        patch("anemoi.training.diagnostics.evaluation.plotting.spectrum.plot_power_spectrum", return_value=MagicMock()),
    ):
        callback._plot(
            trainer,
            pl_module,
            ["data"],
            outputs,
            batch,
            batch_idx=0,
            epoch=0,
        )
        assert mock_output_figure.call_count == pl_module.task.num_output_timesteps


def test_plot_spectrum_forecaster():
    """PlotSpectrum._plot produces one figure per (rollout_step, out_step) for forecaster."""
    from unittest.mock import patch

    callback = PlotSpectrum(
        sample_idx=0,
        parameters=["a", "b"],
        dataset_names=["data"],
    )
    rollout_steps = 2
    n_step_output = 1
    nvar = 2
    nlatlon = 20
    pl_module = _make_pl_module_forecaster(
        n_step_output=n_step_output,
        validation_rollout=rollout_steps,
        nlatlon=nlatlon,
    )
    callback.post_processors = {"data": _identity_post_processor()}
    callback.latlons = {"data": np.zeros((nlatlon, 2))}
    sample_idx = 10
    batch = {"data": torch.randn(2, sample_idx, 1, nlatlon, nvar)}
    outputs = _step_output(
        [{"data": torch.randn(2, n_step_output, 1, nlatlon, nvar)} for _ in range(rollout_steps)],
    )
    trainer = MagicMock()
    trainer.logger = MagicMock()

    with (
        patch.object(callback, "_output_figure") as mock_output_figure,
        patch("anemoi.training.diagnostics.evaluation.plotting.spectrum.plot_power_spectrum", return_value=MagicMock()),
    ):
        callback._plot(
            trainer,
            pl_module,
            ["data"],
            outputs,
            batch,
            batch_idx=0,
            epoch=0,
        )
        # Forecaster branch: rollout_steps * n_step_output figures
        assert mock_output_figure.call_count == rollout_steps * n_step_output


# ---- PlotHistogram ----


def test_plot_histogram_temporal_downscaler():
    """PlotHistogram._plot produces one figure per output_times for temporal downscaler."""
    from unittest.mock import patch

    callback = PlotHistogram(
        sample_idx=0,
        parameters=["a", "b"],
        dataset_names=["data"],
    )
    nvar = 2
    nlatlon = 20
    pl_module = _make_pl_module_temporal_downscaler(nlatlon=nlatlon)

    callback.post_processors = {"data": _identity_post_processor()}
    callback.latlons = {"data": np.zeros((nlatlon, 2))}
    batch = {"data": torch.randn(2, 10, 1, nlatlon, nvar)}
    outputs = _step_output(
        [
            {"data": torch.randn(2, 1, 1, nlatlon, nvar)},
            {"data": torch.randn(2, 1, 1, nlatlon, nvar)},
        ],
    )
    trainer = MagicMock()
    trainer.logger = MagicMock()

    with (
        patch.object(callback, "_output_figure") as mock_output_figure,
        patch("anemoi.training.diagnostics.evaluation.plotting.histogram.plot_histogram", return_value=MagicMock()),
    ):
        callback._plot(
            trainer,
            pl_module,
            ["data"],
            outputs,
            batch,
            batch_idx=0,
            epoch=0,
        )
        assert mock_output_figure.call_count == pl_module.task.num_output_timesteps


def test_plot_histogram_forecaster():
    """PlotHistogram._plot produces one figure per (rollout_step, out_step) for forecaster."""
    from unittest.mock import patch

    callback = PlotHistogram(
        sample_idx=0,
        parameters=["a", "b"],
        dataset_names=["data"],
    )
    validation_rollout = 2
    n_step_output = 1
    nvar = 2
    nlatlon = 20
    pl_module = _make_pl_module_forecaster(
        validation_rollout=validation_rollout,
        n_step_output=n_step_output,
        nlatlon=nlatlon,
    )
    callback.post_processors = {"data": _identity_post_processor()}
    callback.latlons = {"data": np.zeros((nlatlon, 2))}
    sample_idx = 10
    batch = {"data": torch.randn(2, sample_idx, 1, nlatlon, nvar)}
    outputs = _step_output(
        [{"data": torch.randn(2, n_step_output, 1, nlatlon, nvar)} for _ in range(validation_rollout)],
    )
    trainer = MagicMock()
    trainer.logger = MagicMock()

    with (
        patch.object(callback, "_output_figure") as mock_output_figure,
        patch("anemoi.training.diagnostics.evaluation.plotting.histogram.plot_histogram", return_value=MagicMock()),
    ):
        callback._plot(
            trainer,
            pl_module,
            ["data"],
            outputs,
            batch,
            batch_idx=0,
            epoch=0,
        )
        assert mock_output_figure.call_count == validation_rollout * n_step_output


# ---- Plot functions (diagnostics.plots) return a figure ----


def skip_missing_pyshtools():
    """Skip tests if pyshtools is not installed (required for power spectrum plots)."""
    try:
        import pyshtools  # noqa: F401
    except ImportError:
        return pytest.mark.skip(reason="pyshtools not installed")
    else:
        return lambda f: f


def test_plots_plot_loss_returns_figure():
    """plot_loss returns a Figure and runs without error."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    from anemoi.training.diagnostics.evaluation.plotting.loss import plot_loss

    x = np.array([0.1, 0.2, 0.15, 0.25])
    colors = np.array(["C0", "C1", "C2", "C3"])
    xticks = {"a": 0, "b": 1, "c": 2, "d": 3}
    legend_patches = [mpatches.Patch(color="C0", label="a"), mpatches.Patch(color="C1", label="b")]

    fig = plot_loss(x, colors, xticks=xticks, legend_patches=legend_patches)

    assert fig is not None
    assert hasattr(fig, "savefig")
    fig.clear()
    plt.close(fig)


def test_plots_plot_histogram_returns_figure():
    """plot_histogram returns a Figure and runs without error."""
    import matplotlib.pyplot as plt

    from anemoi.training.diagnostics.evaluation.plotting.histogram import plot_histogram

    # parameters: variable_idx -> (variable_name, diagnostic_only)
    parameters = {0: ("t2m", False), 1: ("tp", True)}
    nlatlon, nvar = 12, 2
    rng = np.random.default_rng()
    x = rng.standard_normal((nlatlon, nvar)).astype(np.float64)
    y_true = rng.standard_normal((nlatlon, nvar)).astype(np.float64)
    y_pred = rng.standard_normal((nlatlon, nvar)).astype(np.float64)

    fig = plot_histogram(
        parameters,
        x,
        y_true,
        y_pred,
        precip_and_related_fields=["tp"],
        log_scale=False,
    )

    assert fig is not None
    assert hasattr(fig, "savefig")
    fig.clear()
    plt.close(fig)


@skip_missing_pyshtools()
def test_plots_plot_power_spectrum_returns_figure():
    """plot_power_spectrum returns a Figure and runs without error."""
    import matplotlib.pyplot as plt

    from anemoi.training.diagnostics.evaluation.plotting.spectrum import plot_power_spectrum

    # parameters: variable_idx -> (variable_name, diagnostic_only)
    parameters = {0: ("t2m", False), 1: ("tp", True)}
    nvar = 2
    # Use a 2D grid of points
    lat = np.linspace(50, 55, 4)
    lon = np.linspace(0, 5, 4)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    latlons = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=1)
    nlatlon = latlons.shape[0]
    rng = np.random.default_rng()
    x = rng.standard_normal((nlatlon, nvar)).astype(np.float64)
    y_true = rng.standard_normal((nlatlon, nvar)).astype(np.float64)
    y_pred = rng.standard_normal((nlatlon, nvar)).astype(np.float64)

    fig = plot_power_spectrum(parameters, latlons, x, y_true, y_pred, min_delta=0.01)

    assert fig is not None
    assert hasattr(fig, "savefig")
    fig.clear()
    plt.close(fig)


def test_plots_plot_predicted_multilevel_flat_sample_returns_figure():
    """plot_predicted_multilevel_flat_sample returns a Figure and runs without error."""
    import matplotlib.pyplot as plt

    from anemoi.training.diagnostics.evaluation.plotting.sample import plot_predicted_multilevel_flat_sample

    parameters = {0: ("t2m", True), 1: ("tp", False)}
    n_plots_per_sample = 6
    nlatlon, nvar = 12, 2
    latlons = np.stack(
        [np.linspace(50, 55, nlatlon), np.linspace(0, 5, nlatlon)],
        axis=1,
    )
    rng = np.random.default_rng()
    x = rng.standard_normal((nlatlon, nvar)).astype(np.float64)
    y_true = rng.standard_normal((nlatlon, nvar)).astype(np.float64)
    y_pred = rng.standard_normal((nlatlon, nvar)).astype(np.float64)

    fig = plot_predicted_multilevel_flat_sample(
        parameters,
        n_plots_per_sample,
        latlons,
        0.5,
        x,
        y_true,
        y_pred,
        datashader=False,
    )

    assert fig is not None
    assert hasattr(fig, "savefig")
    fig.clear()
    plt.close(fig)


def test_plots_plot_predicted_multilevel_flat_sample_accepts_auxiliary_panel():
    """plot_predicted_multilevel_flat_sample can add the corrupted-target panel."""
    import matplotlib.pyplot as plt

    from anemoi.training.diagnostics.evaluation.plotting.sample import plot_predicted_multilevel_flat_sample

    parameters = {0: ("t2m", False), 1: ("tp", True)}
    nlatlon, nvar = 12, 2
    latlons = np.stack(
        [np.linspace(50, 55, nlatlon), np.linspace(0, 5, nlatlon)],
        axis=1,
    )
    rng = np.random.default_rng()
    x = rng.standard_normal((nlatlon, nvar)).astype(np.float64)
    y_true = rng.standard_normal((nlatlon, nvar)).astype(np.float64)
    y_pred = rng.standard_normal((nlatlon, nvar)).astype(np.float64)
    auxiliary = rng.standard_normal((nlatlon, nvar)).astype(np.float64)

    fig = plot_predicted_multilevel_flat_sample(
        parameters,
        7,
        latlons,
        0.5,
        x,
        y_true,
        y_pred,
        auxiliary=auxiliary,
        auxiliary_label="corrupted targets",
        datashader=False,
    )

    assert fig is not None
    plot_titles = [ax.get_title() for ax in fig.axes]
    assert any(title == "t2m corrupted targets" for title in plot_titles)
    assert any(title == "tp corrupted targets" for title in plot_titles)
    assert "tp increment [pred - input]" not in plot_titles
    assert "tp persist err" not in plot_titles
    fig.clear()
    plt.close(fig)


# Ensemble plot tests


NUM_FIXED_CALLBACKS = 3  # ParentUUIDCallback, CheckVariableOrder, RegisterMigrations

default_config = """
diagnostics:
  callbacks: []

  plot:
    enabled: False
    callbacks: []

  debug:
    # this will detect and trace back NaNs / Infs etc. but will slow down training
    anomaly_detection: False

  enable_checkpointing: False
  checkpoint:

  log: {}
"""


# Ensemble adapter tests
def test_ensemble_plot_adapter_is_ensemble():
    """Test EnsemblePlotAdapterWrapper.is_ensemble property."""
    task = MagicMock()
    inner = ForecasterPlotAdapter(task)
    adapter = EnsemblePlotAdapterWrapper(inner)
    assert adapter.is_ensemble is True
    assert inner.is_ensemble is False


def test_ensemble_plot_adapter_select_members():
    """Test EnsemblePlotAdapterWrapper.select_members method."""
    task = MagicMock()
    inner = ForecasterPlotAdapter(task)
    adapter = EnsemblePlotAdapterWrapper(inner)

    tensor = torch.randn(2, 3, 4, 100, 5)  # (batch, steps, members, grid, vars)

    # Select single member
    result = adapter.select_members(tensor, members=0)
    assert result.shape == (2, 3, 1, 100, 5)

    # Select multiple members
    result = adapter.select_members(tensor, members=[0, 2])
    assert result.shape == (2, 3, 2, 100, 5)

    # Select all members (None)
    result = adapter.select_members(tensor, members=None)
    assert result.shape == (2, 3, 4, 100, 5)
    assert torch.equal(result, tensor)


def test_ensemble_plot_adapter_prepare_loss_batch():
    """Test EnsemblePlotAdapterWrapper.prepare_loss_batch keeps ensemble shape."""
    task = MagicMock()
    inner = ForecasterPlotAdapter(task)
    adapter = EnsemblePlotAdapterWrapper(inner)

    batch = {"data": torch.randn(2, 5, 3, 100, 5)}  # (batch, time, members, grid, vars)
    result = adapter.prepare_loss_batch(batch)

    assert result["data"].shape == (2, 5, 3, 100, 5)
    assert torch.equal(result["data"], batch["data"])


def test_ensemble_plot_adapter_delegates_to_inner():
    """Test that EnsemblePlotAdapterWrapper delegates iter_plot_samples and other methods to inner."""
    task = MagicMock()
    inner = MagicMock()
    inner._task = task
    inner.get_loss_plot_batch_start.return_value = 42
    inner.prepare_plot_output_tensor.side_effect = lambda x: x

    adapter = EnsemblePlotAdapterWrapper(inner)

    assert adapter.get_loss_plot_batch_start(rollout_step=1) == 42
    inner.get_loss_plot_batch_start.assert_called_once_with(rollout_step=1)

    tensor = torch.randn(3, 4)
    adapter.prepare_plot_output_tensor(tensor)
    inner.prepare_plot_output_tensor.assert_called_once_with(tensor)

    data = np.zeros((5, 100, 10))
    output = np.zeros((3, 100, 10))
    list(adapter.iter_plot_samples(data, output))
    inner.iter_plot_samples.assert_called_once_with(data, output)


def test_base_adapter_select_members_is_noop():
    """Test that BasePlotAdapter.select_members is a no-op."""
    task = MagicMock()
    inner = ForecasterPlotAdapter(task)
    tensor = torch.randn(2, 3, 100, 5)

    result = inner.select_members(tensor, members=0)
    assert torch.equal(result, tensor)


def test_base_adapter_prepare_loss_batch_is_noop():
    """Test that BasePlotAdapter.prepare_loss_batch is a no-op."""
    task = MagicMock()
    inner = ForecasterPlotAdapter(task)
    batch = {"data": torch.randn(2, 5, 100, 5)}

    result = inner.prepare_loss_batch(batch)
    assert torch.equal(result["data"], batch["data"])


def test_ensemble_plot_ens_sample_instantiation():
    """Test that PlotEnsSample can be instantiated."""
    plot_ens_sample = PlotEnsSample(
        sample_idx=0,
        parameters=["temperature", "pressure"],
        accumulation_levels_plot=[0.1, 0.5, 0.9],
        members=None,
    )
    assert plot_ens_sample is not None
    assert plot_ens_sample.plot_members is None


@pytest.mark.parametrize("projection_kind", ["robinson", "mollweide"])
def test_sample_plot_fn_global_non_equirectangular_projection_does_not_crash(projection_kind):
    """Global data with non-equirectangular Cartopy projections must not raise.

    ValueError from set_extent (regression test for Robinson/Mollweide extent clamp).
    """
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt

    parameters = {0: ("t2m", False)}
    nlatlon, nvar = 100, 1
    latlons = np.stack(
        [np.linspace(-90, 90, nlatlon), np.linspace(-180, 180, nlatlon)],
        axis=1,
    )
    rng = np.random.default_rng(0)
    x = rng.standard_normal((nlatlon, nvar)).astype(np.float64)
    y_true = rng.standard_normal((nlatlon, nvar)).astype(np.float64)
    y_pred = rng.standard_normal((nlatlon, nvar)).astype(np.float64)

    settings = MagicMock()
    settings.datashader = False
    settings.projection_kind = projection_kind
    settings.precip_and_related_fields = None
    settings.colormaps = None

    fig = sample_plot_fn(
        parameters,
        x=x,
        y_true=y_true,
        y_pred=y_pred,
        latlons=latlons,
        settings=settings,
        per_sample=6,
        accumulation_levels_plot=[0.5],
    )

    assert fig is not None
    fig.clear()
    plt.close(fig)

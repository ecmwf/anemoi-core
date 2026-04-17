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
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import omegaconf
import pytest
import torch

from anemoi.training.diagnostics.callbacks.plot import GraphTrainableFeaturesPlot
from anemoi.training.diagnostics.callbacks.plot import PlotHistogram
from anemoi.training.diagnostics.callbacks.plot import PlotLoss
from anemoi.training.diagnostics.callbacks.plot import PlotSample
from anemoi.training.diagnostics.callbacks.plot import PlotSpectrum
from anemoi.training.tasks import Forecaster
from anemoi.training.tasks import TemporalDownscaler
from anemoi.training.utils.masks import NoOutputMask

# Suite of Unit Tests for Plotting Callbacks
# ------------------------------------------
# Tests to check PlotHistogram, PlotSpectrum, PlotLoss, PlotSample instantiation
# Tests to check PlotHistogram, PlotSpectrum, PlotLoss, PlotSample plot methods
# Tests to check plot_loss, plot_histogram, plot_spectrum, plot_predicted_multilevel_flat_sample return a figure


def test_plot_histogram_instantiation():
    """PlotHistogram can be instantiated with config and parameters."""
    config = omegaconf.OmegaConf.create(_PLOT_PROCESS_CONFIG)
    callback = PlotHistogram(
        config=config,
        sample_idx=0,
        parameters=["t2m", "tp", "u10"],
        dataset_names=["data"],
    )
    assert callback.sample_idx == 0
    assert callback.parameters == ["t2m", "tp", "u10"]
    assert callback.log_scale is False


def test_plot_spectrum_instantiation():
    """PlotSpectrum can be instantiated with config and parameters."""
    config = omegaconf.OmegaConf.create(_PLOT_PROCESS_CONFIG)
    callback = PlotSpectrum(
        config=config,
        sample_idx=0,
        parameters=["t2m", "tp"],
        dataset_names=["data"],
    )
    assert callback.sample_idx == 0
    assert callback.parameters == ["t2m", "tp"]
    assert callback.min_delta is None


def test_plot_loss_instantiation():
    """PlotLoss can be instantiated with config and optional parameter_groups."""
    config = omegaconf.OmegaConf.create(_PLOT_LOSS_CONFIG)
    callback = PlotLoss(config=config, parameter_groups={})
    assert callback.parameter_groups == {}
    assert callback.dataset_names == ["data"]

    callback2 = PlotLoss(
        config=config,
        parameter_groups={"group_a": ["t2m", "tp"], "group_b": ["u10", "v10"]},
        dataset_names=["data"],
    )
    assert len(callback2.parameter_groups) == 2
    assert callback2.parameter_groups["group_a"] == ["t2m", "tp"]


def test_graph_trainable_features_plot_handles_noop_processor_graph_provider():
    config = omegaconf.OmegaConf.create(
        {
            "system": {"output": {"plots": None}},
            "diagnostics": {
                "plot": {
                    "datashader": False,
                    "asynchronous": False,
                    "frequency": {"epoch": 1},
                },
            },
        },
    )
    callback = GraphTrainableFeaturesPlot(config=config)

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

    edge_modules = callback.get_edge_trainable_modules(model, dataset_name="data")

    assert edge_modules == {}


def test_graph_trainable_features_plot_handles_noop_mapper_graph_providers():
    config = omegaconf.OmegaConf.create(
        {
            "system": {"output": {"plots": None}},
            "diagnostics": {
                "plot": {
                    "datashader": False,
                    "asynchronous": False,
                    "frequency": {"epoch": 1},
                },
            },
        },
    )
    callback = GraphTrainableFeaturesPlot(config=config)

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

    edge_modules = callback.get_edge_trainable_modules(model, dataset_name="data")

    assert edge_modules == {}


def test_graph_trainable_features_plot_handles_missing_dataset_key_in_provider_map():
    config = omegaconf.OmegaConf.create(
        {
            "system": {"output": {"plots": None}},
            "diagnostics": {
                "plot": {
                    "datashader": False,
                    "asynchronous": False,
                    "frequency": {"epoch": 1},
                },
            },
        },
    )
    callback = GraphTrainableFeaturesPlot(config=config)

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

    edge_modules = callback.get_edge_trainable_modules(model, dataset_name="data")

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


# ---- BasePlotAdditionalMetrics.process: input/output shapes ----


def test_process_forecaster_output_shapes():
    """BasePlotAdditionalMetrics.process: forecaster task yields expected data and output_tensor shapes."""
    config = omegaconf.OmegaConf.create(_PLOT_PROCESS_CONFIG)
    callback = PlotSample(
        config=config,
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
    # outputs: (loss, [pred_0, pred_1, ...]); each pred[dataset] (bs, n_step_output, ens, latlon, nvar)
    outputs = (
        torch.tensor(0.0),
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


def test_process_temporal_downscaler_output_shapes():
    """BasePlotAdditionalMetrics.process: temporal downscaler task yields expected shapes."""
    config = omegaconf.OmegaConf.create(_PLOT_PROCESS_CONFIG)
    callback = PlotSample(
        config=config,
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
    outputs = (
        torch.tensor(0.0),
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
    config = omegaconf.OmegaConf.create(_PLOT_PROCESS_CONFIG)
    callback = PlotSample(
        config=config,
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
    outputs = (
        torch.tensor(0.0),
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


# ---- PlotLoss ----

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
    """PlotLoss.sort_and_color_by_parameter_group: <=15 params returns identity sort and correct output shapes."""
    config = omegaconf.OmegaConf.create(_PLOT_LOSS_CONFIG)
    callback = PlotLoss(config=config, parameter_groups={})
    parameter_names = ["t2m", "tp", "u10", "v10"]
    sort_idx, colors, xticks, legend_patches = callback.sort_and_color_by_parameter_group(parameter_names)

    assert sort_idx.shape == (len(parameter_names),)
    assert np.array_equal(sort_idx, np.arange(len(parameter_names)))
    assert len(colors) == len(parameter_names)
    assert isinstance(xticks, dict)
    assert len(legend_patches) >= 1
    # One patch per unique "group" (here each param is its own group for <=15)
    assert len(legend_patches) == len(parameter_names)


def test_plot_loss_sort_and_color_by_parameter_group_with_groups():
    """PlotLoss.sort_and_color_by_parameter_group: with parameter_groups and >15 params returns grouped sort."""
    config = omegaconf.OmegaConf.create(_PLOT_LOSS_CONFIG)
    callback = PlotLoss(
        config=config,
        parameter_groups={
            "pressure": ["tp", "sp"] + [f"p{i}" for i in range(6)],
            "wind": ["u10", "v10"] + [f"w{i}" for i in range(6)],
        },
    )
    # >15 parameters to trigger the grouping branch (<=15 keeps each param as its own group)
    parameter_names = ["tp", "sp", "p0", "p1", "p2", "p3", "p4", "p5", "u10", "v10", "w0", "w1", "w2", "w3", "w4", "w5"]
    sort_idx, colors, xticks, legend_patches = callback.sort_and_color_by_parameter_group(parameter_names)

    assert sort_idx.shape == (len(parameter_names),)
    assert len(colors) == len(parameter_names)
    assert isinstance(xticks, dict)
    assert len(legend_patches) == 2  # pressure and wind


def test_plot_loss_temporal_downscaler():
    """PlotLoss._plot uses output_times=1 only one figure is produced."""
    from unittest.mock import patch

    from anemoi.training.losses.mse import MSELoss

    config = omegaconf.OmegaConf.create(_PLOT_LOSS_CONFIG)
    callback = PlotLoss(config=config, parameter_groups={}, dataset_names=["data"])
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
    outputs = (
        torch.tensor(0.0),
        [{"data": torch.randn(batch_size, 1, 1, nlatlon, nvar)}],
    )
    callback.loss = {"data": MSELoss()}

    with (
        patch.object(callback, "_output_figure") as mock_output_figure,
        patch(
            "anemoi.training.diagnostics.callbacks.plot.argsort_variablename_variablelevel",
            return_value=np.arange(nvar),
        ),
        patch("anemoi.training.diagnostics.callbacks.plot.plot_loss", return_value=MagicMock()),
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


def test_plot_loss_diffusion():
    """PlotLoss._plot with diffusion (forecaster, output_times=1) produces one figure."""
    from unittest.mock import patch

    from anemoi.training.losses.mse import MSELoss

    config = omegaconf.OmegaConf.create(_PLOT_LOSS_CONFIG)
    callback = PlotLoss(config=config, parameter_groups={}, dataset_names=["data"])
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
    outputs = (
        torch.tensor(0.0),
        [{"data": torch.randn(batch_size, n_step_output, 1, nlatlon, nvar)}],
    )
    callback.loss = {"data": MSELoss()}
    pl_module.task.steps.return_value = [{}]
    pl_module.task.get_targets.return_value = {"data": torch.randn(batch_size, n_step_output, 1, nlatlon, nvar)}
    pl_module.task.get_metric_name.return_value = ""

    with (
        patch.object(callback, "_output_figure") as mock_output_figure,
        patch(
            "anemoi.training.diagnostics.callbacks.plot.argsort_variablename_variablelevel",
            return_value=np.arange(nvar),
        ),
        patch("anemoi.training.diagnostics.callbacks.plot.plot_loss", return_value=MagicMock()),
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
        # Diffusion has output_times=1, so one figure
        assert mock_output_figure.call_count == 1


def test_plot_loss_forecaster():
    """PlotLoss._plot uses one figure per rollout step."""
    from unittest.mock import patch

    from anemoi.training.losses.mse import MSELoss

    config = omegaconf.OmegaConf.create(_PLOT_LOSS_CONFIG)
    callback = PlotLoss(config=config, parameter_groups={}, dataset_names=["data"])
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
    outputs = (
        torch.tensor(0.0),
        [{"data": torch.randn(batch_size, n_step_output, 1, nlatlon, nvar)} for _ in range(output_times)],
    )
    callback.loss = {"data": MSELoss()}
    pl_module.task.steps.return_value = [{"rollout_step": i} for i in range(output_times)]
    pl_module.task.get_targets.return_value = {"data": torch.randn(batch_size, n_step_output, 1, nlatlon, nvar)}
    pl_module.task.get_metric_name.return_value = ""

    with (
        patch.object(callback, "_output_figure") as mock_output_figure,
        patch(
            "anemoi.training.diagnostics.callbacks.plot.argsort_variablename_variablelevel",
            return_value=np.arange(nvar),
        ),
        patch("anemoi.training.diagnostics.callbacks.plot.plot_loss", return_value=MagicMock()),
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


# ---- PlotSpectrum ----


def test_plot_spectrum_temporal_downscaler():
    """PlotSpectrum._plot produces one figure per output_times for temporal downscaler."""
    from unittest.mock import patch

    config = omegaconf.OmegaConf.create(_PLOT_PROCESS_CONFIG)
    callback = PlotSpectrum(
        config=config,
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
    outputs = (
        torch.tensor(0.0),
        [
            {"data": torch.randn(2, 1, 1, nlatlon, nvar)},
            {"data": torch.randn(2, 1, 1, nlatlon, nvar)},
        ],
    )
    trainer = MagicMock()
    trainer.logger = MagicMock()

    with (
        patch.object(callback, "_output_figure") as mock_output_figure,
        patch("anemoi.training.diagnostics.callbacks.plot.plot_power_spectrum", return_value=MagicMock()),
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

    config = omegaconf.OmegaConf.create(_PLOT_PROCESS_CONFIG)
    callback = PlotSpectrum(
        config=config,
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
    outputs = (
        torch.tensor(0.0),
        [{"data": torch.randn(2, n_step_output, 1, nlatlon, nvar)} for _ in range(rollout_steps)],
    )
    trainer = MagicMock()
    trainer.logger = MagicMock()

    with (
        patch.object(callback, "_output_figure") as mock_output_figure,
        patch("anemoi.training.diagnostics.callbacks.plot.plot_power_spectrum", return_value=MagicMock()),
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

    config = omegaconf.OmegaConf.create(_PLOT_PROCESS_CONFIG)
    callback = PlotHistogram(
        config=config,
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
    outputs = (
        torch.tensor(0.0),
        [
            {"data": torch.randn(2, 1, 1, nlatlon, nvar)},
            {"data": torch.randn(2, 1, 1, nlatlon, nvar)},
        ],
    )
    trainer = MagicMock()
    trainer.logger = MagicMock()

    with (
        patch.object(callback, "_output_figure") as mock_output_figure,
        patch("anemoi.training.diagnostics.callbacks.plot.plot_histogram", return_value=MagicMock()),
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

    config = omegaconf.OmegaConf.create(_PLOT_PROCESS_CONFIG)
    callback = PlotHistogram(
        config=config,
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
    outputs = (
        torch.tensor(0.0),
        [{"data": torch.randn(2, n_step_output, 1, nlatlon, nvar)} for _ in range(validation_rollout)],
    )
    trainer = MagicMock()
    trainer.logger = MagicMock()

    with (
        patch.object(callback, "_output_figure") as mock_output_figure,
        patch("anemoi.training.diagnostics.callbacks.plot.plot_histogram", return_value=MagicMock()),
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

    from anemoi.training.diagnostics.plots import plot_loss

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

    from anemoi.training.diagnostics.plots import plot_histogram

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

    from anemoi.training.diagnostics.plots import plot_power_spectrum

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

    from anemoi.training.diagnostics.plots import plot_predicted_multilevel_flat_sample

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

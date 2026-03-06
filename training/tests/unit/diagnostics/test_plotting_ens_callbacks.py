# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# ruff: noqa: ANN201

from unittest.mock import MagicMock

import omegaconf
import torch

from anemoi.training.diagnostics.callbacks.plot_ens import EnsemblePlotMixin
from anemoi.training.diagnostics.callbacks.plot_ens import PlotEnsSample
from anemoi.training.diagnostics.callbacks.plot_ens import PlotHistogram
from anemoi.training.diagnostics.callbacks.plot_ens import PlotSample
from anemoi.training.diagnostics.callbacks.plot_ens import PlotSpectrum

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


# Ensemble callback tests
def test_ensemble_plot_mixin_handle_batch_and_output():
    """Test EnsemblePlotMixin._handle_ensemble_batch_and_output method."""
    mixin = EnsemblePlotMixin()
    dataset_name = "data"

    # Mock lightning module: allgather_batch(tensor, grid_indices, grid_dim) -> tensor
    pl_module = MagicMock()
    pl_module.allgather_batch.side_effect = lambda x, *_args: x
    pl_module.grid_indices = {dataset_name: MagicMock()}
    pl_module.grid_dim = -2

    # Mock ensemble output: (loss, list of dict[dataset_name -> tensor])
    loss = torch.tensor(0.5)
    y_preds = [
        {dataset_name: torch.randn(2, 3, 4, 5)},
        {dataset_name: torch.randn(2, 3, 4, 5)},
    ]
    output = [loss, y_preds]

    # Mock batch: dict[dataset_name -> tensor]
    batch = {dataset_name: torch.randn(2, 10, 4, 5)}

    processed_batch, processed_output = mixin._handle_ensemble_batch_and_output(pl_module, output, batch)

    # Check that processed_batch is the allgathered batch dict
    assert dataset_name in processed_batch
    assert torch.equal(processed_batch[dataset_name], batch[dataset_name])
    # Check that output is restructured as [loss, y_preds]
    assert len(processed_output) == 2
    assert torch.equal(processed_output[0], loss)
    assert len(processed_output[1]) == 2


def test_ensemble_plot_mixin_process():
    """Test EnsemblePlotMixin.process method."""
    mixin = EnsemblePlotMixin()
    mixin.sample_idx = 0
    mixin.latlons = None
    dataset_name = "data"

    # Mock lightning module
    pl_module = MagicMock()
    pl_module.task_type = "forecaster"
    pl_module.n_step_input = 2
    pl_module.n_step_output = 1
    pl_module.plot_adapter = MagicMock()
    pl_module.plot_adapter.output_times = 3
    pl_module.plot_adapter.get_total_plot_targets.return_value = 3
    pl_module.plot_adapter.prepare_plot_output_tensor.side_effect = lambda x: x
    pl_module.model.model._graph_name_data = "x"
    pl_module.model.model._graph_data = {dataset_name: MagicMock()}
    graph_node = MagicMock()
    graph_node.x = torch.randn(100, 2)
    pl_module.model.model._graph_data[dataset_name].__getitem__ = lambda _self, k: graph_node if k == "x" else None

    # data_indices: dict[dataset_name -> IndexCollection]
    data_indices = MagicMock()
    data_indices.data.output.full = slice(None)
    pl_module.data_indices = {dataset_name: data_indices}

    # batch: dict[dataset_name -> tensor], shape (bs, input_steps + forecast_steps, latlon, nvar)
    # n_step_input=1, total_targets=3, need 5 steps: 1 + 3 + 1
    batch = {dataset_name: torch.randn(2, 6, 100, 5)}

    # outputs: (loss, list of dict[dataset_name -> tensor])
    # Each pred tensor: (bs, latlon, n_ensemble, nvar) for indexing [:, :, members, ...]
    # Batch slice 0:5 gives (2, 5, 100, 5) for input_tensor
    data_tensor = torch.randn(2, 5, 100, 5)
    output_preds = [
        {dataset_name: torch.randn(2, 100, 1, 5)},
        {dataset_name: torch.randn(2, 100, 1, 5)},
        {dataset_name: torch.randn(2, 100, 1, 5)},
    ]
    outputs = [torch.tensor(0.5), output_preds]

    # Mock post_processors: returns tensor; for outputs, shape (bs, latlon, n_ensemble, nvar)
    mock_post_processor = MagicMock()
    mock_post_processor.side_effect = [
        data_tensor,
        torch.randn(2, 100, 1, 5),
        torch.randn(2, 100, 1, 5),
        torch.randn(2, 100, 1, 5),
    ]
    mock_post_processor.cpu.return_value = mock_post_processor
    pl_module.model.post_processors = {dataset_name: mock_post_processor}

    # post_processors on mixin: dict[dataset_name -> processor]
    mixin.post_processors = {dataset_name: mock_post_processor}

    # output_mask.apply as identity
    pl_module.output_mask = {dataset_name: MagicMock()}
    pl_module.output_mask[dataset_name].apply.side_effect = lambda x, **_kwargs: x

    data, result_output_tensor = mixin.process(
        pl_module,
        dataset_name,
        outputs,
        batch,
        members=0,
    )

    # Check instantiation
    assert data is not None
    assert result_output_tensor is not None

    # data: (n_steps, latlon, nvar) with n_steps = 1 + total_targets = 5
    # result_output_tensor equas to(output_times, latlon, nvar) = (3, 100, 5)
    assert data.shape[1:] == (100, 5), f"Expected data shape (..., 100, 5), got {data.shape}"
    assert result_output_tensor.shape == (
        3,
        100,
        5,
    ), f"Expected output_tensor shape (3, 100, 5), got {result_output_tensor.shape}"


def test_ensemble_plot_callbacks_instantiation():
    """Test that ensemble plot callbacks can be instantiated."""
    config = omegaconf.OmegaConf.create(
        {
            "system": {"output": {"plots": "path_to_plots"}},
            "diagnostics": {
                "plot": {
                    "parameters": ["temperature", "pressure"],
                    "datashader": False,
                    "asynchronous": False,
                    "frequency": {"batch": 1},
                },
            },
            "data": {
                "diagnostic": None,
                "datasets": {"data": {"diagnostic": None}},
            },
            "dataloader": {"read_group_size": 1},
        },
    )

    # Test plotting class instantiation
    plot_ens_sample = PlotEnsSample(
        config=config,
        sample_idx=0,
        parameters=["temperature", "pressure"],
        accumulation_levels_plot=[0.1, 0.5, 0.9],
        output_steps=1,
    )
    assert plot_ens_sample is not None

    plot_sample = PlotSample(
        config=config,
        sample_idx=0,
        parameters=["temperature"],
        accumulation_levels_plot=[0.5],
        output_steps=1,
    )
    assert plot_sample is not None

    plot_spectrum = PlotSpectrum(
        config=config,
        sample_idx=0,
        parameters=["temperature"],
        output_steps=1,
    )
    assert plot_spectrum is not None

    plot_histogram = PlotHistogram(
        config=config,
        sample_idx=0,
        parameters=["temperature"],
        output_steps=1,
    )
    assert plot_histogram is not None

# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for PlotPayload and PlotSetup in BasePlotAdapter."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch

from anemoi.training.diagnostics.callbacks.plot_adapter import PlotPayload
from anemoi.training.diagnostics.callbacks.plot_adapter import PlotSetup
from anemoi.training.tasks import Forecaster
from anemoi.training.train.step_output import TrainingStepOutput
from anemoi.training.utils.masks import NoOutputMask


def _identity_post_processor() -> MagicMock:
    """Return a mock post-processor that acts as identity."""
    mock = MagicMock()

    def _call(x: torch.Tensor | np.ndarray, in_place: bool = True) -> torch.Tensor | np.ndarray:  # noqa: ARG001
        return x.clone() if isinstance(x, torch.Tensor) else x

    mock.side_effect = _call
    mock.cpu = MagicMock(return_value=mock)
    mock.processors = MagicMock()
    mock.processors.values = MagicMock(return_value=[])
    return mock


def _make_pl_module(*, nlatlon: int = 50) -> MagicMock:
    """Create a mock pl_module with all attributes needed by prepare_batch."""
    pl_module = MagicMock()
    pl_module.local_rank = 0
    pl_module.grid_dim = 3

    # allgather_batch is identity in non-distributed setting
    pl_module.allgather_batch = MagicMock(side_effect=lambda x, _name: x)

    # post_processors
    post_proc = _identity_post_processor()
    pl_module.model.post_processors = {"data": post_proc}

    # data_indices
    data_indices = MagicMock()
    data_indices.data.output.full = slice(None)
    pl_module.data_indices = {"data": data_indices}

    # output_mask
    pl_module.output_mask = {"data": NoOutputMask()}

    # graph data for latlons
    graph_data_mock = MagicMock()
    graph_data_mock.x = torch.zeros(nlatlon, 2)
    pl_module.model.model._graph_data = {"data": graph_data_mock}

    return pl_module


class TestPlotPayload:
    """Tests for the PlotPayload dataclass."""

    def test_payload_creation(self) -> None:
        """PlotPayload can be created with expected fields."""
        payload = PlotPayload(
            batch_idx=0,
            batch={"data": torch.randn(2, 4, 1, 50, 3)},
            predictions=[{"data": torch.randn(2, 1, 1, 50, 3)}],
            post_processors={"data": _identity_post_processor()},
            latlons={"data": np.zeros((50, 2))},
        )
        assert payload.batch_idx == 0
        assert "data" in payload.batch
        assert "data" in payload.post_processors
        assert "data" in payload.latlons

    def test_payload_is_immutable_batch_idx(self) -> None:
        """PlotPayload batch_idx is set correctly."""
        payload = PlotPayload(
            batch_idx=5,
            batch={},
            predictions=[],
            post_processors={},
            latlons={},
        )
        assert payload.batch_idx == 5


class TestPlotSetup:
    """Tests for the PlotSetup dataclass."""

    def test_plot_setup_creation(self) -> None:
        """PlotSetup can be created with expected fields."""
        setup = PlotSetup(
            post_processors={"data": _identity_post_processor()},
            latlons={"data": np.zeros((50, 2))},
            feature_indices={"data": slice(None)},
        )
        assert "data" in setup.post_processors
        assert "data" in setup.latlons
        assert "data" in setup.feature_indices


class TestPreparePayload:
    """Tests for BasePlotAdapter.prepare_payload."""

    def test_prepare_payload_returns_plot_payload(self) -> None:
        """prepare_payload returns a PlotPayload instance."""
        task = Forecaster(
            multistep_input=2,
            multistep_output=1,
            timestep="6H",
            rollout={"start": 1, "epoch_increment": 1, "maximum": 2},
        )
        adapter = task._plot_adapter

        pl_module = _make_pl_module()
        batch = {"data": torch.randn(2, 4, 1, 50, 3)}
        output = TrainingStepOutput(
            loss=torch.tensor(0.0),
            metrics={},
            predictions=[{"data": torch.randn(2, 1, 1, 50, 3)}, {"data": torch.randn(2, 1, 1, 50, 3)}],
        )

        payload = adapter.prepare_payload(pl_module, batch, output, batch_idx=0)

        assert isinstance(payload, PlotPayload)
        assert payload.batch_idx == 0
        assert "data" in payload.batch
        assert "data" in payload.latlons

    def test_prepare_payload_caches_setup_not_payload(self) -> None:
        """Calling prepare_payload twice caches only the setup, not the payload itself.

        The PlotSetup (post_processors, latlons, feature_indices) is reused
        across calls so that the expensive post-processor deep-copy runs only
        once per epoch.  The large per-batch tensors (batch, predictions) are
        gathered fresh each call so they are freed between batches.
        """
        task = Forecaster(
            multistep_input=2,
            multistep_output=1,
            timestep="6H",
            rollout={"start": 1, "epoch_increment": 1, "maximum": 2},
        )
        adapter = task._plot_adapter

        pl_module = _make_pl_module()
        batch = {"data": torch.randn(2, 4, 1, 50, 3)}
        output = TrainingStepOutput(
            loss=torch.tensor(0.0),
            metrics={},
            predictions=[{"data": torch.randn(2, 1, 1, 50, 3)}, {"data": torch.randn(2, 1, 1, 50, 3)}],
        )

        payload1 = adapter.prepare_payload(pl_module, batch, output, batch_idx=7)
        setup_after_first = adapter._cached_setup
        payload2 = adapter.prepare_payload(pl_module, batch, output, batch_idx=7)

        # Setup is cached and reused
        assert adapter._cached_setup is setup_after_first
        assert payload1.post_processors is payload2.post_processors
        assert payload1.latlons is payload2.latlons
        # But payloads themselves are distinct objects
        assert payload1 is not payload2

    def test_prepare_payload_invalidates_on_new_batch_idx(self) -> None:
        """Calling prepare_payload with a different batch_idx produces a new payload."""
        task = Forecaster(
            multistep_input=2,
            multistep_output=1,
            timestep="6H",
            rollout={"start": 1, "epoch_increment": 1, "maximum": 2},
        )
        adapter = task._plot_adapter

        pl_module = _make_pl_module()
        batch = {"data": torch.randn(2, 4, 1, 50, 3)}
        output = TrainingStepOutput(
            loss=torch.tensor(0.0),
            metrics={},
            predictions=[{"data": torch.randn(2, 1, 1, 50, 3)}, {"data": torch.randn(2, 1, 1, 50, 3)}],
        )

        payload1 = adapter.prepare_payload(pl_module, batch, output, batch_idx=0)
        payload2 = adapter.prepare_payload(pl_module, batch, output, batch_idx=1)

        assert payload1 is not payload2
        assert payload2.batch_idx == 1

    def test_prepare_payload_gathers_batches(self) -> None:
        """prepare_payload calls allgather_batch for each dataset in batch."""
        task = Forecaster(
            multistep_input=2,
            multistep_output=1,
            timestep="6H",
            rollout={"start": 1, "epoch_increment": 1, "maximum": 1},
        )
        adapter = task._plot_adapter

        pl_module = _make_pl_module()
        batch = {"data": torch.randn(2, 4, 1, 50, 3)}
        output = TrainingStepOutput(
            loss=torch.tensor(0.0),
            metrics={},
            predictions=[{"data": torch.randn(2, 1, 1, 50, 3)}],
        )

        adapter.prepare_payload(pl_module, batch, output, batch_idx=0)

        # allgather_batch called for batch["data"] + each pred dict entry
        assert pl_module.allgather_batch.call_count >= 2

    def test_prepare_payload_post_processors_on_cpu(self) -> None:
        """prepare_payload moves post_processors to CPU."""
        task = Forecaster(
            multistep_input=2,
            multistep_output=1,
            timestep="6H",
            rollout={"start": 1, "epoch_increment": 1, "maximum": 1},
        )
        adapter = task._plot_adapter

        pl_module = _make_pl_module()
        batch = {"data": torch.randn(2, 4, 1, 50, 3)}
        output = TrainingStepOutput(
            loss=torch.tensor(0.0),
            metrics={},
            predictions=[{"data": torch.randn(2, 1, 1, 50, 3)}],
        )

        payload = adapter.prepare_payload(pl_module, batch, output, batch_idx=0)

        # The post_processor's .cpu() should have been called
        assert payload.post_processors["data"].cpu.called

    def test_prepare_payload_computes_latlons(self) -> None:
        """prepare_payload extracts latlons from graph_data and converts to degrees."""
        task = Forecaster(
            multistep_input=2,
            multistep_output=1,
            timestep="6H",
            rollout={"start": 1, "epoch_increment": 1, "maximum": 1},
        )
        adapter = task._plot_adapter

        # Set known radian values
        pl_module = _make_pl_module(nlatlon=10)
        pl_module.model.model._graph_data["data"].x = torch.ones(10, 2) * (np.pi / 2)

        batch = {"data": torch.randn(2, 4, 1, 10, 3)}
        output = TrainingStepOutput(
            loss=torch.tensor(0.0),
            metrics={},
            predictions=[{"data": torch.randn(2, 1, 1, 10, 3)}],
        )

        payload = adapter.prepare_payload(pl_module, batch, output, batch_idx=0)

        # pi/2 radians = 90 degrees
        np.testing.assert_allclose(payload.latlons["data"], 90.0, atol=1e-5)

    def test_post_processor_deepcopy_called_once_across_batches(self) -> None:
        """post-processor deep-copy is done only once even across multiple batch calls."""
        import copy

        task = Forecaster(
            multistep_input=2,
            multistep_output=1,
            timestep="6H",
            rollout={"start": 1, "epoch_increment": 1, "maximum": 1},
        )
        adapter = task._plot_adapter

        pl_module = _make_pl_module()
        batch = {"data": torch.randn(2, 4, 1, 50, 3)}
        output = TrainingStepOutput(
            loss=torch.tensor(0.0),
            metrics={},
            predictions=[{"data": torch.randn(2, 1, 1, 50, 3)}],
        )

        with MagicMock(wraps=copy.deepcopy) as mock_deepcopy:
            import anemoi.training.diagnostics.callbacks.plot_adapter as _mod

            original = _mod.copy.deepcopy
            _mod.copy.deepcopy = mock_deepcopy
            try:
                adapter.prepare_payload(pl_module, batch, output, batch_idx=0)
                count_after_first = mock_deepcopy.call_count
                adapter.prepare_payload(pl_module, batch, output, batch_idx=1)
                count_after_second = mock_deepcopy.call_count
            finally:
                _mod.copy.deepcopy = original

        # deepcopy called once (for setup), not again for subsequent batches
        assert count_after_first == count_after_second


class TestGetDenormalized:
    """Tests for PlotPayload.get_denormalized lazy denormalization."""

    def test_get_denormalized_returns_tensors(self) -> None:
        """get_denormalized returns (denormed_input, denormed_output) tensors."""
        task = Forecaster(
            multistep_input=2,
            multistep_output=1,
            timestep="6H",
            rollout={"start": 1, "epoch_increment": 1, "maximum": 2},
        )
        adapter = task._plot_adapter

        pl_module = _make_pl_module(nlatlon=50)
        batch = {"data": torch.randn(2, 4, 1, 50, 3)}
        output = TrainingStepOutput(
            loss=torch.tensor(0.0),
            metrics={},
            predictions=[{"data": torch.randn(2, 1, 1, 50, 3)}, {"data": torch.randn(2, 1, 1, 50, 3)}],
        )

        payload = adapter.prepare_payload(pl_module, batch, output, batch_idx=0)
        denormed_input, denormed_output = payload.get_denormalized("data")

        assert isinstance(denormed_input, torch.Tensor)
        assert isinstance(denormed_output, torch.Tensor)

    def test_denormed_input_shape(self) -> None:
        """denormed_input has shape matching batch[dataset][..., feature_indices]."""
        batch_size, nlatlon, nvar = 2, 50, 3
        task = Forecaster(
            multistep_input=2,
            multistep_output=1,
            timestep="6H",
            rollout={"start": 1, "epoch_increment": 1, "maximum": 1},
        )
        adapter = task._plot_adapter

        pl_module = _make_pl_module(nlatlon=nlatlon)
        batch = {"data": torch.randn(batch_size, 4, 1, nlatlon, nvar)}
        output = TrainingStepOutput(
            loss=torch.tensor(0.0),
            metrics={},
            predictions=[{"data": torch.randn(batch_size, 1, 1, nlatlon, nvar)}],
        )

        payload = adapter.prepare_payload(pl_module, batch, output, batch_idx=0)
        denormed_input, _ = payload.get_denormalized("data")

        assert denormed_input.shape == (batch_size, 4, 1, nlatlon, nvar)

    def test_denormed_output_shape(self) -> None:
        """denormed_output has shape (n_steps, batch, n_step_out, ens, grid, vars)."""
        batch_size, nlatlon, nvar, n_steps = 2, 50, 3, 2
        task = Forecaster(
            multistep_input=2,
            multistep_output=1,
            timestep="6H",
            rollout={"start": 1, "epoch_increment": 1, "maximum": n_steps},
        )
        adapter = task._plot_adapter

        pl_module = _make_pl_module(nlatlon=nlatlon)
        batch = {"data": torch.randn(batch_size, 4, 1, nlatlon, nvar)}
        output = TrainingStepOutput(
            loss=torch.tensor(0.0),
            metrics={},
            predictions=[{"data": torch.randn(batch_size, 1, 1, nlatlon, nvar)} for _ in range(n_steps)],
        )

        payload = adapter.prepare_payload(pl_module, batch, output, batch_idx=0)
        _, denormed_output = payload.get_denormalized("data")

        assert denormed_output.shape[0] == n_steps
        assert denormed_output.shape[1] == batch_size

    def test_get_denormalized_not_cached(self) -> None:
        """get_denormalized computes fresh each call (no caching) to avoid holding tensors."""
        task = Forecaster(
            multistep_input=2,
            multistep_output=1,
            timestep="6H",
            rollout={"start": 1, "epoch_increment": 1, "maximum": 1},
        )
        adapter = task._plot_adapter

        pl_module = _make_pl_module()
        batch = {"data": torch.randn(2, 4, 1, 50, 3)}
        output = TrainingStepOutput(
            loss=torch.tensor(0.0),
            metrics={},
            predictions=[{"data": torch.randn(2, 1, 1, 50, 3)}],
        )

        payload = adapter.prepare_payload(pl_module, batch, output, batch_idx=5)

        result1 = payload.get_denormalized("data")
        result2 = payload.get_denormalized("data")

        # Each call returns new tensor objects (not cached)
        assert result1[0] is not result2[0]
        assert result1[1] is not result2[1]

    def test_multiple_datasets_independent(self) -> None:
        """Each dataset gets its own independent denormalized tensors."""
        task = Forecaster(
            multistep_input=2,
            multistep_output=1,
            timestep="6H",
            rollout={"start": 1, "epoch_increment": 1, "maximum": 1},
        )
        adapter = task._plot_adapter

        pl_module = _make_pl_module(nlatlon=50)
        # Add a second dataset
        post_proc_b = _identity_post_processor()
        pl_module.model.post_processors["data_b"] = post_proc_b
        pl_module.data_indices["data_b"] = pl_module.data_indices["data"]
        pl_module.output_mask["data_b"] = NoOutputMask()
        graph_data_mock_b = MagicMock()
        graph_data_mock_b.x = torch.zeros(50, 2)
        pl_module.model.model._graph_data["data_b"] = graph_data_mock_b

        batch = {"data": torch.randn(2, 4, 1, 50, 3), "data_b": torch.randn(2, 4, 1, 50, 3)}
        output = TrainingStepOutput(
            loss=torch.tensor(0.0),
            metrics={},
            predictions=[{"data": torch.randn(2, 1, 1, 50, 3), "data_b": torch.randn(2, 1, 1, 50, 3)}],
        )

        payload = adapter.prepare_payload(pl_module, batch, output, batch_idx=0)

        result_a = payload.get_denormalized("data")
        result_b = payload.get_denormalized("data_b")

        assert result_a[0] is not result_b[0]


class TestClearCache:
    """Tests for BasePlotAdapter.clear_cache."""

    def test_clear_cache_releases_setup(self) -> None:
        """clear_cache sets _cached_setup to None."""
        task = Forecaster(
            multistep_input=2,
            multistep_output=1,
            timestep="6H",
            rollout={"start": 1, "epoch_increment": 1, "maximum": 1},
        )
        adapter = task._plot_adapter

        pl_module = _make_pl_module()
        batch = {"data": torch.randn(2, 4, 1, 50, 3)}
        output = TrainingStepOutput(
            loss=torch.tensor(0.0),
            metrics={},
            predictions=[{"data": torch.randn(2, 1, 1, 50, 3)}],
        )

        adapter.prepare_payload(pl_module, batch, output, batch_idx=0)
        assert adapter._cached_setup is not None

        adapter.clear_cache()
        assert adapter._cached_setup is None

    def test_prepare_payload_works_after_clear(self) -> None:
        """prepare_payload produces a fresh payload after clear_cache."""
        task = Forecaster(
            multistep_input=2,
            multistep_output=1,
            timestep="6H",
            rollout={"start": 1, "epoch_increment": 1, "maximum": 1},
        )
        adapter = task._plot_adapter

        pl_module = _make_pl_module()
        batch = {"data": torch.randn(2, 4, 1, 50, 3)}
        output = TrainingStepOutput(
            loss=torch.tensor(0.0),
            metrics={},
            predictions=[{"data": torch.randn(2, 1, 1, 50, 3)}],
        )

        payload1 = adapter.prepare_payload(pl_module, batch, output, batch_idx=0)
        adapter.clear_cache()
        payload2 = adapter.prepare_payload(pl_module, batch, output, batch_idx=0)

        assert payload1 is not payload2

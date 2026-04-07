# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for GraphDiffusionDownscaler and diffusion downscaler model."""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.train.tasks.diffusiondownscaler import GraphDiffusionDownscaler

# ============================================================
# Helpers
# ============================================================


# Question for Joffrey - shall we have a test with diagnostics? how are those handled for DS?
def _make_index_collection(
    name_to_index: dict[str, int],
    forcing: list[str] | None = None,
    diagnostic: list[str] | None = None,
) -> IndexCollection:
    cfg = DictConfig({"forcing": forcing or [], "diagnostic": diagnostic or [], "target": []})
    return IndexCollection(cfg, name_to_index)


class _IdentityProcessor(torch.nn.Module):
    def forward(self, x, in_place=False, **_kwargs):
        return x if in_place else x.clone()


class _ScaleBy2Processor(torch.nn.Module):
    def forward(self, x, in_place=False, **_kwargs):
        return x * 2.0


def _make_mixed_model():
    """Create a minimal AnemoiD2ModelEncProcDec mock with mixed prognostic/diagnostic."""
    from anemoi.models.models.diffusiondownscaler_encoder_processor_decoder import AnemoiD2ModelEncProcDec

    model = object.__new__(AnemoiD2ModelEncProcDec)
    model.data_indices = {
        "in_lres": _make_index_collection({"10u": 0, "10v": 1, "2t": 2}),
        "out_hres": _make_index_collection({"10u": 0, "tp": 1}, diagnostic=["tp"]),
    }
    model._residual_pairs = {"out_hres": "in_lres"}
    model._matching_channel_indices_out_hres = torch.tensor([0])
    model._matching_indices_keys = [("out_hres", "in_lres", "_matching_channel_indices_out_hres")]
    return model


# ============================================================
# Tests
# ============================================================


def test_get_noise_level_dict():
    """_get_noise_level returns dict[str, Tensor] for sigma and weight."""
    ds = GraphDiffusionDownscaler.__new__(GraphDiffusionDownscaler)
    ds.rho = 7.0
    ds.lognormal_mean = -1.2
    ds.lognormal_std = 1.2
    ds.training_approach = "probabilistic_low_noise"
    ds._residual_pairs = {"out_hres": "in_lres"}

    shape = {"out_hres": (4, 1, 1, 1, 1)}
    sigma, weight = ds._get_noise_level(
        shape=shape,
        sigma_max=100000.0,
        sigma_min=0.02,
        sigma_data=1.0,
        rho=7.0,
        device=torch.device("cpu"),
    )
    assert isinstance(sigma, dict) and "out_hres" in sigma
    assert sigma["out_hres"].shape == (4, 1, 1, 1, 1)
    assert (sigma["out_hres"] > 0).all()
    assert (weight["out_hres"] > 0).all()


def test_noise_scales_with_sigma():
    """_noise_target produces larger noise for larger sigma (dict interface)."""
    ds = GraphDiffusionDownscaler.__new__(GraphDiffusionDownscaler)
    x = {"out_hres": torch.zeros(2, 1, 1, 10, 3)}

    torch.manual_seed(42)
    result_small = ds._noise_target(x, {"out_hres": torch.full((2, 1, 1, 1, 1), 0.01)})
    torch.manual_seed(42)
    result_large = ds._noise_target(x, {"out_hres": torch.full((2, 1, 1, 1, 1), 100.0)})

    assert result_large["out_hres"].abs().mean() > result_small["out_hres"].abs().mean()


def test_channel_reindex_correctness():
    """_match_tensor_channels reindexes a tensor to match output channel order."""
    from anemoi.models.models.diffusiondownscaler_encoder_processor_decoder import AnemoiD2ModelEncProcDec

    input_idx = {"t2m": 0, "u10": 1, "v10": 2, "sp": 3}
    output_idx = {"sp": 0, "t2m": 1, "u10": 2}
    indices = AnemoiD2ModelEncProcDec._match_tensor_channels(None, input_idx, output_idx)

    x = torch.tensor([[10.0, 20.0, 30.0, 40.0]])  # t2m=10, u10=20, v10=30, sp=40
    assert torch.equal(x[..., indices], torch.tensor([[40.0, 10.0, 20.0]]))


def test_mixed_residual_target():
    """Prognostic channels get y-x_interp, diagnostic channels get y directly."""
    out_indices = _make_index_collection({"10u": 0, "tp": 1}, diagnostic=["tp"])
    prog_out = out_indices.model.output.prognostic
    diag_out = out_indices.model.output.diagnostic

    y = torch.tensor([[[[[150.0, 5.0]]]]])  # 10u=150, tp=5
    x_interp = torch.tensor([[[[[100.0]]]]])  # matched 10u

    target = y.clone()
    target[..., prog_out] = y[..., prog_out] - x_interp  # 150 - 100 = 50

    assert torch.allclose(target[..., prog_out], torch.tensor([[[[[50.0]]]]]))
    assert torch.allclose(target[..., diag_out], torch.tensor([[[[[5.0]]]]]))


def test_compute_residuals_with_tendency():
    """compute_residuals applies tendency processor to prognostic, state processor to diagnostic."""
    model = _make_mixed_model()

    target = model.compute_residuals(
        y=torch.tensor([[[[[150.0, 5.0]]]]]),
        x_interp=torch.tensor([[[[[100.0]]]]]),
        pre_processors_state=_IdentityProcessor(),
        pre_processors_tendencies=_ScaleBy2Processor(),
    )

    prog_out = model.data_indices["out_hres"].model.output.prognostic
    diag_out = model.data_indices["out_hres"].model.output.diagnostic

    # Prognostic: (150-100) * 2 = 100, Diagnostic: 5.0 (identity)
    assert torch.allclose(target[..., prog_out], torch.tensor([[[[[100.0]]]]]))
    assert torch.allclose(target[..., diag_out], torch.tensor([[[[[5.0]]]]]))


def test_add_interp_reconstruction():
    """add_interp_to_state adds x_interp to prognostic channels, leaves diagnostic alone."""
    model = _make_mixed_model()
    identity = _IdentityProcessor()

    result = model.add_interp_to_state(
        state_inp=torch.tensor([[[[[100.0, 200.0, 300.0]]]]]),  # in_lres: 10u, 10v, 2t
        model_output=torch.tensor([[[[[50.0, 5.0]]]]]),  # out_hres: 10u_residual, tp_direct
        post_processors_state={"in_lres": identity, "out_hres": identity},
        post_processors_tendencies=None,
    )

    prog_out = model.data_indices["out_hres"].model.output.prognostic
    diag_out = model.data_indices["out_hres"].model.output.diagnostic

    # 10u: 50 + 100 = 150, tp: 5 (unchanged)
    assert torch.allclose(result[..., prog_out], torch.tensor([[[[[150.0]]]]]))
    assert torch.allclose(result[..., diag_out], torch.tensor([[[[[5.0]]]]]))


# ============================================================
# Round-trip: compute_residuals -> add_interp_to_state recovers y
# ============================================================


def test_round_trip_residual_identity_processors():
    """compute_residuals followed by add_interp_to_state with identity processors recovers y exactly."""
    model = _make_mixed_model()
    identity = _IdentityProcessor()

    y = torch.tensor([[[[[150.0, 5.0]]]]])       # 10u=150 (prognostic), tp=5 (diagnostic)
    x_interp = torch.tensor([[[[[100.0]]]]])      # matched 10u=100 from in_lres

    residual_target = model.compute_residuals(
        y=y,
        x_interp=x_interp,
        pre_processors_state=identity,
        pre_processors_tendencies=identity,
    )

    y_reconstructed = model.add_interp_to_state(
        state_inp=torch.tensor([[[[[100.0, 200.0, 300.0]]]]]),  # in_lres channels
        model_output=residual_target,
        post_processors_state={"in_lres": identity, "out_hres": identity},
        post_processors_tendencies={"out_hres": identity},
    )

    assert torch.allclose(y_reconstructed, y, atol=1e-5), (
        f"Round-trip failed: expected {y}, got {y_reconstructed}"
    )


def test_round_trip_residual_scale_processors():
    """Round-trip holds when pre/post processors are exact inverses (scale x2 / x0.5)."""
    model = _make_mixed_model()
    identity = _IdentityProcessor()

    class _ScaleBy(torch.nn.Module):
        def __init__(self, factor):
            super().__init__()
            self.factor = factor

        def forward(self, x, in_place=False, **_kwargs):
            return x * self.factor

    y = torch.tensor([[[[[150.0, 5.0]]]]])
    x_interp = torch.tensor([[[[[100.0]]]]])

    residual_target = model.compute_residuals(
        y=y,
        x_interp=x_interp,
        pre_processors_state=identity,
        pre_processors_tendencies=_ScaleBy(2.0),   # normalize: multiply by 2
    )

    y_reconstructed = model.add_interp_to_state(
        state_inp=torch.tensor([[[[[100.0, 200.0, 300.0]]]]]),
        model_output=residual_target,
        post_processors_state={"in_lres": identity, "out_hres": identity},
        post_processors_tendencies={"out_hres": _ScaleBy(0.5)},   # denormalize: multiply by 0.5
    )

    assert torch.allclose(y_reconstructed, y, atol=1e-5), (
        f"Round-trip with scaled processors failed: expected {y}, got {y_reconstructed}"
    )


# ============================================================
# Noise schedule modes
# ============================================================


def _make_downscaler_noise():
    """Minimal GraphDiffusionDownscaler for noise tests (no full init needed)."""
    ds = GraphDiffusionDownscaler.__new__(GraphDiffusionDownscaler)
    ds.rho = 7.0
    ds.lognormal_mean = -1.2
    ds.lognormal_std = 1.2
    ds._residual_pairs = {"out_hres": "in_lres"}
    return ds


def test_noise_high_mode_sigma_in_range():
    """probabilistic_high_noise sigma stays within [sigma_min, sigma_max]."""
    ds = _make_downscaler_noise()
    ds.training_approach = "probabilistic_high_noise"

    sigma_min, sigma_max = 0.02, 100000.0
    sigma, _ = ds._get_noise_level(
        shape={"out_hres": (32, 1, 4, 1, 1)},
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        sigma_data=1.0,
        rho=7.0,
        device=torch.device("cpu"),
    )
    s = sigma["out_hres"]
    assert (s >= sigma_min).all() and (s <= sigma_max).all(), (
        f"sigma out of range: min={s.min():.4f}, max={s.max():.4f}"
    )


def test_noise_deterministic_mode_sigma_value():
    """deterministic mode returns sigma=500000 exactly."""
    ds = _make_downscaler_noise()
    ds.training_approach = "deterministic"

    sigma, _ = ds._get_noise_level(
        shape={"out_hres": (2, 1, 1, 1, 1)},
        sigma_max=100000.0,
        sigma_min=0.02,
        sigma_data=1.0,
        rho=7.0,
        device=torch.device("cpu"),
    )
    assert (sigma["out_hres"] == 500000.0).all()




# ============================================================
# _step smoke test: full GraphDiffusionDownscaler._step with mocked model
# ============================================================


class _MockD2Model(torch.nn.Module):
    """Minimal mock of AnemoiD2ModelEncProcDec for _step testing."""

    def __init__(self, n_vars_lres: int, n_vars_hres: int, n_vars_out: int, n_grid: int) -> None:
        super().__init__()
        self.sigma_max = 100000.0
        self.sigma_min = 0.02
        self.sigma_data = 1.0
        self._decoder_datasets = ["out_hres"]
        self._residual_pairs = {"out_hres": "in_lres"}
        self._n_vars_out = n_vars_out
        self._n_grid = n_grid

        # Matching channel indices: assume first n_vars_lres channels of out match in_lres
        self._matching_channel_indices_out_hres = torch.arange(n_vars_lres, dtype=torch.long)

    def get_matching_channel_indices(self, target_dataset: str) -> torch.Tensor:
        return self._matching_channel_indices_out_hres

    def compute_residuals(self, y, x_interp, pre_processors_state, pre_processors_tendencies, target_dataset, **_):
        # Return (y_prog - x_interp) for prog channels, y_diag for diagnostic
        result = y.clone()
        result[..., :x_interp.shape[-1]] = y[..., :x_interp.shape[-1]] - x_interp
        return result

    def add_interp_to_state(self, state_inp, model_output, post_processors_state, post_processors_tendencies,
                             target_dataset, source_dataset, **_):
        return model_output  # identity for test purposes

    def fwd_with_preconditioning(self, x, y_noised, sigma, model_comm_group=None, grid_shard_shapes=None):
        # Return zeros of the same shape as target
        return {"out_hres": torch.zeros_like(y_noised["out_hres"])}

    class _Residual(torch.nn.Module):
        def forward(self, x, grid_shard_shapes=None, model_comm_group=None):
            # InterpolationConnection selects last timestep and returns (batch, time=1, grid, vars)
            return x[:, -1:, ...]

    @property
    def residual(self):
        return {"in_lres": self._Residual()}


class _MockTaskModel(torch.nn.Module):
    """Wraps _MockD2Model with pre/post processors to satisfy GraphDiffusionDownscaler._step."""

    def __init__(self, d2_model: _MockD2Model) -> None:
        super().__init__()
        self.model = d2_model
        self.pre_processors = {
            "in_lres": _IdentityProcessor(),
            "in_hres": _IdentityProcessor(),
            "out_hres": _IdentityProcessor(),
        }
        self.post_processors = {
            "in_lres": _IdentityProcessor(),
            "out_hres": _IdentityProcessor(),
        }
        self.post_processors_tendencies = None


def _make_downscaler_task(n_vars_lres=2, n_vars_hres=1, n_vars_out=3, n_grid=4) -> GraphDiffusionDownscaler:
    """Build a minimal GraphDiffusionDownscaler without full LightningModule init.

    Calls nn.Module.__init__ so that checkpoint() and hooks work correctly,
    then assigns attributes directly.
    """
    task = GraphDiffusionDownscaler.__new__(GraphDiffusionDownscaler)
    torch.nn.Module.__init__(task)  # initializes _modules, _parameters, _backward_hooks, etc.

    d2_model = _MockD2Model(n_vars_lres, n_vars_hres, n_vars_out, n_grid)
    task.model = _MockTaskModel(d2_model)
    task.model_comm_group = None
    task.grid_shard_shapes = None
    task.rho = 7.0
    task.lognormal_mean = -1.2
    task.lognormal_std = 1.2
    task.training_approach = "probabilistic_low_noise"
    task._residual_pairs = {"out_hres": "in_lres"}
    task._residual_pre_processors = {}
    task._residual_post_processors = {}

    # Mock loss: returns mean of (pred - target)^2 weighted
    class _MockLoss:
        def __call__(self, pred, target, weights=None, grid_shard_slice=None, group=None):
            diff = (pred - target) ** 2
            if weights is not None:
                diff = diff * weights
            return diff.mean()

    task.loss = {"out_hres": _MockLoss()}
    task.target_dataset_names = ["out_hres"]

    # compute_loss_metrics: call _compute_loss directly for simplicity
    def _compute_loss_metrics(y_pred, y, validation_mode=False, weights=None):
        loss = task._compute_loss(
            y_pred["out_hres"], y["out_hres"],
            dataset_name="out_hres",
            weights=weights,
            grid_shard_slice=None,
        )
        return loss, {}, y_pred

    task.compute_loss_metrics = _compute_loss_metrics

    return task


def test_step_returns_scalar_loss():
    """_step returns a scalar loss and a list of predictions with correct shape."""
    n_grid, n_vars_lres, n_vars_hres, n_vars_out = 4, 2, 1, 3
    task = _make_downscaler_task(n_vars_lres, n_vars_hres, n_vars_out, n_grid)

    batch = {
        "in_lres":  torch.randn(2, 1, n_grid, n_vars_lres),    # (batch, time, grid, vars)
        "in_hres":  torch.randn(2, 1, 1, n_grid, n_vars_hres), # (batch, time, ensemble, grid, vars)
        "out_hres": torch.randn(2, 1, 1, n_grid, n_vars_out),  # (batch, time, ensemble, grid, vars)
    }

    loss, metrics, preds = task._step(batch, validation_mode=False)

    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert torch.isfinite(loss), "Loss should be finite"
    assert isinstance(preds, list) and len(preds) == 1


def test_step_on_after_batch_transfer_does_not_normalize():
    """on_after_batch_transfer returns raw (un-normalized) batch — needed for residual computation."""
    task = _make_downscaler_task()
    task._setup_batch_sharding = lambda b: b  # no-op sharding
    task._prepare_loss_scalers = lambda: None  # no-op

    raw_values = torch.tensor([[[1.0, 2.0]]])
    batch = {"in_lres": raw_values.clone(), "in_hres": raw_values.clone(), "out_hres": raw_values.clone()}

    returned = task.on_after_batch_transfer(batch, 0)

    # Batch must be returned unchanged (no normalization applied)
    torch.testing.assert_close(returned["in_lres"], batch["in_lres"])
    torch.testing.assert_close(returned["out_hres"], batch["out_hres"])

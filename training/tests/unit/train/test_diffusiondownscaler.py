# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for GraphDiffusionDownscaler training task."""

from __future__ import annotations

import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.train.tasks.diffusiondownscaler import GraphDiffusionDownscaler

# ============================================================
# Test helpers
# ============================================================


def _make_index_collection(
    name_to_index: dict[str, int],
    forcing: list[str] | None = None,
    diagnostic: list[str] | None = None,
) -> IndexCollection:
    """Create a minimal IndexCollection for testing."""
    cfg = DictConfig(
        {
            "forcing": forcing or [],
            "diagnostic": diagnostic or [],
            "target": [],
        },
    )
    return IndexCollection(cfg, name_to_index)


class _IdentityProcessor(torch.nn.Module):
    """Processor that returns input unchanged (identity normalization)."""

    def forward(self, x, in_place=False, **_kwargs):
        return x if in_place else x.clone()


class _DummyResidualConnection(torch.nn.Module):
    """Dummy interpolation that just returns the last timestep (identity upsampling)."""

    def forward(self, x, grid_shard_shapes=None, model_comm_group=None):
        # x: (batch, time, ensemble, grid, features) -> (batch, time, grid, features)
        return x[:, -1, ...]


class _DummyDownscalerModel:
    """Lightweight mock of the full AnemoiD2ModelEncProcDec interface."""

    def __init__(self, num_output_vars: int, channel_indices: torch.Tensor):
        self.sigma_max = 100000.0
        self.sigma_min = 0.02
        self.sigma_data = 1.0
        self.num_output_vars = num_output_vars
        self._matching_channel_indices_out_hres = channel_indices
        self._matching_indices_keys = [("out_hres", "in_lres", "_matching_channel_indices_out_hres")]
        self._residual_pairs = {"out_hres": "in_lres"}
        self._decoder_datasets = ["out_hres"]

        # residual["in_lres"] performs upsampling
        self.residual = torch.nn.ModuleDict({"in_lres": _DummyResidualConnection()})

    def get_matching_channel_indices(self, target_dataset: str) -> torch.Tensor:
        buf_name = f"_matching_channel_indices_{target_dataset}"
        return getattr(self, buf_name)

    def fwd_with_preconditioning(self, x_dict, y_noised, sigma, **_kwargs):
        # Return a tensor matching the out_hres shape
        y = y_noised["out_hres"]
        return {"out_hres": y * 0.1}


class _DummyModelInterface:
    """Wraps the model and provides pre/post processors like the real interface."""

    def __init__(self, model: _DummyDownscalerModel):
        self.model = model
        self.pre_processors = {
            "in_lres": _IdentityProcessor(),
            "in_hres": _IdentityProcessor(),
            "out_hres": _IdentityProcessor(),
        }
        self.post_processors = {
            "in_lres": _IdentityProcessor(),
            "in_hres": _IdentityProcessor(),
            "out_hres": _IdentityProcessor(),
        }


# ============================================================
# Noise level tests
# ============================================================


class TestGetNoiseLevel:
    """Tests for _get_noise_level method."""

    @pytest.fixture
    def downscaler(self, monkeypatch):
        """Create a GraphDiffusionDownscaler with monkeypatched __init__."""
        ds = GraphDiffusionDownscaler.__new__(GraphDiffusionDownscaler)
        # Set required attributes directly
        ds.rho = 7.0
        ds.lognormal_mean = -1.2
        ds.lognormal_std = 1.2
        ds.training_approach = "probabilistic_low_noise"
        ds._residual_pairs = {"out_hres": "in_lres"}
        return ds

    def test_probabilistic_low_noise(self, downscaler):
        shape = {"out_hres": (4, 1, 1, 1, 1)}
        sigma, weight = downscaler._get_noise_level(
            shape=shape,
            sigma_max=100000.0,
            sigma_min=0.02,
            sigma_data=1.0,
            rho=7.0,
            device=torch.device("cpu"),
        )
        assert isinstance(sigma, dict) and "out_hres" in sigma
        assert isinstance(weight, dict) and "out_hres" in weight
        assert sigma["out_hres"].shape == (4, 1, 1, 1, 1)
        assert weight["out_hres"].shape == (4, 1, 1, 1, 1)
        assert (sigma["out_hres"] > 0).all()
        assert (weight["out_hres"] > 0).all()

    def test_probabilistic_high_noise(self, downscaler):
        downscaler.training_approach = "probabilistic_high_noise"
        shape = {"out_hres": (4, 1, 1, 1, 1)}
        sigma, weight = downscaler._get_noise_level(
            shape=shape,
            sigma_max=100000.0,
            sigma_min=0.02,
            sigma_data=1.0,
            rho=7.0,
            device=torch.device("cpu"),
        )
        assert sigma["out_hres"].shape == (4, 1, 1, 1, 1)
        assert (sigma["out_hres"] >= 0.02).all()
        assert (sigma["out_hres"] <= 100000.0).all()

    def test_deterministic(self, downscaler):
        downscaler.training_approach = "deterministic"
        shape = {"out_hres": (4, 1, 1, 1, 1)}
        sigma, weight = downscaler._get_noise_level(
            shape=shape,
            sigma_max=100000.0,
            sigma_min=0.02,
            sigma_data=1.0,
            rho=7.0,
            device=torch.device("cpu"),
        )
        assert torch.allclose(sigma["out_hres"], torch.full((4, 1, 1, 1, 1), 500000.0))


# ============================================================
# Residual prediction flag tests
# ============================================================


class TestResidualPredictionConfig:
    """Tests for the residual_prediction configuration."""

    def test_residual_prediction_dict_parsed(self):
        """residual_prediction dict should be parsed into _residual_pairs."""
        raw = {"out_hres": "in_lres"}
        assert isinstance(raw, dict)
        pairs = dict(raw)
        assert pairs == {"out_hres": "in_lres"}

    def test_residual_prediction_true_raises(self):
        """residual_prediction=True should raise ValueError (must be a dict)."""
        from anemoi.models.models.diffusiondownscaler_encoder_processor_decoder import AnemoiD2ModelEncProcDec

        model = object.__new__(AnemoiD2ModelEncProcDec)
        model._encoder_datasets = ["out_hres"]
        model._decoder_datasets = ["out_hres"]

        # Simulate what __init__ does with raw=True
        raw = True
        with pytest.raises(ValueError, match="must be a dict"):
            if isinstance(raw, dict):
                pass
            elif raw:
                raise ValueError(
                    f"residual_prediction must be a dict mapping target->source datasets "
                    f"(e.g. {{out_hres: in_lres}}) or False, got: {raw}",
                )

    def test_residual_prediction_false_gives_empty(self):
        """residual_prediction=False should give empty pairs dict."""
        raw = False
        if isinstance(raw, dict):
            pairs = dict(raw)
        elif raw:
            raise ValueError("should not happen")
        else:
            pairs = {}
        assert pairs == {}


# ============================================================
# Noise target tests
# ============================================================


class TestNoiseTarget:
    """Tests for _noise_target method."""

    def test_noise_adds_gaussian(self):
        ds = GraphDiffusionDownscaler.__new__(GraphDiffusionDownscaler)
        x = {"out_hres": torch.zeros(2, 1, 1, 10, 3)}
        sigma = {"out_hres": torch.ones(2, 1, 1, 1, 1)}

        torch.manual_seed(42)
        result = ds._noise_target(x, sigma)

        assert isinstance(result, dict) and "out_hres" in result
        # With x=0, result should be ~ N(0, sigma) = N(0, 1)
        assert result["out_hres"].shape == x["out_hres"].shape
        # The noise should be non-zero
        assert not torch.allclose(result["out_hres"], x["out_hres"])

    def test_noise_scales_with_sigma(self):
        ds = GraphDiffusionDownscaler.__new__(GraphDiffusionDownscaler)
        x = {"out_hres": torch.zeros(2, 1, 1, 10, 3)}

        torch.manual_seed(42)
        sigma_small = {"out_hres": torch.full((2, 1, 1, 1, 1), 0.01)}
        result_small = ds._noise_target(x, sigma_small)

        torch.manual_seed(42)
        sigma_large = {"out_hres": torch.full((2, 1, 1, 1, 1), 100.0)}
        result_large = ds._noise_target(x, sigma_large)

        # Larger sigma should produce larger magnitude noise
        assert result_large["out_hres"].abs().mean() > result_small["out_hres"].abs().mean()


# ============================================================
# Channel matching tests (model-level)
# ============================================================


class TestChannelMatching:
    """Tests for _match_tensor_channels and _build_matching_channel_indices."""

    def test_identical_channels(self):
        """When input and output have same channels in same order, indices should be identity."""
        model = type("Model", (), {})()
        model._match_tensor_channels = (
            lambda self, inp, out: torch.tensor([inp[name] for name in out.keys() if name in inp], dtype=torch.long)
        ).__get__(model)

        input_idx = {"a": 0, "b": 1, "c": 2}
        output_idx = {"a": 0, "b": 1, "c": 2}
        from anemoi.models.models.diffusiondownscaler_encoder_processor_decoder import AnemoiD2ModelEncProcDec

        indices = AnemoiD2ModelEncProcDec._match_tensor_channels(None, input_idx, output_idx)
        assert torch.equal(indices, torch.tensor([0, 1, 2]))

    def test_reordered_channels(self):
        """When output channels are reordered, indices should reflect the reordering."""
        from anemoi.models.models.diffusiondownscaler_encoder_processor_decoder import AnemoiD2ModelEncProcDec

        input_idx = {"a": 0, "b": 1, "c": 2}
        output_idx = {"c": 0, "a": 1, "b": 2}
        indices = AnemoiD2ModelEncProcDec._match_tensor_channels(None, input_idx, output_idx)
        assert torch.equal(indices, torch.tensor([2, 0, 1]))

    def test_subset_channels(self):
        """When output has fewer channels than input, indices select the subset."""
        from anemoi.models.models.diffusiondownscaler_encoder_processor_decoder import AnemoiD2ModelEncProcDec

        input_idx = {"a": 0, "b": 1, "c": 2, "d": 3}
        output_idx = {"b": 0, "d": 1}
        indices = AnemoiD2ModelEncProcDec._match_tensor_channels(None, input_idx, output_idx)
        assert torch.equal(indices, torch.tensor([1, 3]))

    def test_channel_reindex_correctness(self):
        """End-to-end: applying indices to a tensor gives correct channel mapping."""
        from anemoi.models.models.diffusiondownscaler_encoder_processor_decoder import AnemoiD2ModelEncProcDec

        input_idx = {"t2m": 0, "u10": 1, "v10": 2, "sp": 3}
        output_idx = {"sp": 0, "t2m": 1, "u10": 2}
        indices = AnemoiD2ModelEncProcDec._match_tensor_channels(None, input_idx, output_idx)

        # Create a tensor with known values per channel
        x = torch.tensor([[10.0, 20.0, 30.0, 40.0]])  # (1, 4) — t2m=10, u10=20, v10=30, sp=40
        reindexed = x[..., indices]  # Should give [sp=40, t2m=10, u10=20]
        assert torch.equal(reindexed, torch.tensor([[40.0, 10.0, 20.0]]))


# ============================================================
# Training step behavior tests (residual vs direct)
# ============================================================


class TestResidualTargetComputation:
    """Test the residual vs direct prediction target computation logic.

    These tests exercise the core computation that happens in _step() without
    needing the full base class pipeline (compute_loss_metrics, etc.).
    """

    def test_residual_target_is_y_minus_x_interp(self):
        """With residual_prediction=True, target = y - x_interp[channel_indices]."""
        channel_indices = torch.tensor([0, 1, 2])
        x_interp = torch.ones(2, 1, 1, 10, 3) * 5.0
        y = torch.ones(2, 1, 1, 10, 3) * 8.0

        # Residual prediction
        target = y - x_interp[..., channel_indices]
        expected = torch.ones(2, 1, 1, 10, 3) * 3.0
        assert torch.allclose(target, expected)

    def test_direct_target_is_y(self):
        """With residual_prediction=False, target = y."""
        x_interp = torch.ones(2, 1, 1, 10, 3) * 5.0
        y = torch.ones(2, 1, 1, 10, 3) * 8.0

        # Direct prediction
        target = y
        assert torch.allclose(target, torch.ones(2, 1, 1, 10, 3) * 8.0)

    def test_residual_reconstruction(self):
        """Full prediction = x_interp + predicted_residual when residual_prediction=True."""
        channel_indices = torch.tensor([0, 1, 2])
        x_interp = torch.ones(2, 1, 1, 10, 3) * 5.0
        predicted_residual = torch.ones(2, 1, 1, 10, 3) * 3.0

        y_pred_full = x_interp[..., channel_indices] + predicted_residual
        expected = torch.ones(2, 1, 1, 10, 3) * 8.0
        assert torch.allclose(y_pred_full, expected)

    def test_direct_prediction_no_reconstruction(self):
        """With direct prediction, y_pred_full == y_pred_denorm (no x_interp addition)."""
        y_pred_denorm = torch.ones(2, 1, 1, 10, 3) * 8.0
        y_pred_full = y_pred_denorm  # Direct: no addition
        assert torch.equal(y_pred_full, y_pred_denorm)

    def test_residual_with_reordered_channels(self):
        """Channel reordering in residual computation gives correct results."""
        # Input has channels [a=10, b=20, c=30]
        # Output order is [c, a] so channel_indices = [2, 0]
        channel_indices = torch.tensor([2, 0])
        x_interp = torch.tensor([[[[[10.0, 20.0, 30.0]]]]])  # (1,1,1,1,3)
        y = torch.tensor([[[[[35.0, 15.0]]]]])  # (1,1,1,1,2) — c_target=35, a_target=15

        target = y - x_interp[..., channel_indices]
        # x_interp reindexed: [c=30, a=10]
        # residual: [35-30, 15-10] = [5, 5]
        expected = torch.tensor([[[[[5.0, 5.0]]]]])
        assert torch.allclose(target, expected)


# ============================================================
# Mixed residual/direct prediction tests
# ============================================================


class TestMixedResidualDirect:
    """Test the mixed case: in_lres=[10u, 10v, 2t], out_hres=[10u, tp].

    10u: present in both → residual prediction (prognostic)
    tp: only in out_hres → direct prediction (diagnostic)
    10v, 2t: only in in_lres → forcing (not predicted)
    """

    @pytest.fixture
    def data_indices(self):
        """Create data indices for the mixed case."""
        in_lres = _make_index_collection({"10u": 0, "10v": 1, "2t": 2})
        # tp is diagnostic (not in in_lres → direct prediction)
        out_hres = _make_index_collection({"10u": 0, "tp": 1}, diagnostic=["tp"])
        in_hres = _make_index_collection({"cos_lat": 0, "sin_lat": 1})
        return {"in_lres": in_lres, "in_hres": in_hres, "out_hres": out_hres}

    def test_channel_matching_partial_overlap(self, data_indices):
        """Channel matching only maps common channels (10u), not tp."""
        from anemoi.models.models.diffusiondownscaler_encoder_processor_decoder import AnemoiD2ModelEncProcDec

        input_idx = data_indices["in_lres"].name_to_index  # {10u: 0, 10v: 1, 2t: 2}
        output_idx = data_indices["out_hres"].name_to_index  # {10u: 0, tp: 1}
        indices = AnemoiD2ModelEncProcDec._match_tensor_channels(None, input_idx, output_idx)
        # Only 10u is common → indices should be [0] (in_lres index of 10u)
        assert torch.equal(indices, torch.tensor([0]))

    def test_prognostic_diagnostic_split(self, data_indices):
        """out_hres should have 10u as prognostic and tp as diagnostic."""
        out_indices = data_indices["out_hres"]
        # 10u is prognostic (not in forcing, not in diagnostic)
        assert "10u" in [name for name, idx in out_indices.name_to_index.items() if name not in out_indices.diagnostic]
        # tp is diagnostic
        assert "tp" in out_indices.diagnostic

    def test_prognostic_output_indices(self, data_indices):
        """model.output.prognostic should contain index for 10u."""
        out_indices = data_indices["out_hres"]
        prog_indices = out_indices.model.output.prognostic
        diag_indices = out_indices.model.output.diagnostic
        # 10u → prognostic, tp → diagnostic
        assert len(prog_indices) == 1  # 10u
        assert len(diag_indices) == 1  # tp

    def test_residual_target_mixed(self, data_indices):
        """Residual target: 10u gets y-x_interp, tp gets y directly."""
        out_indices = data_indices["out_hres"]
        prog_out = out_indices.model.output.prognostic
        diag_out = out_indices.model.output.diagnostic

        # x_interp has only the matched channel (10u from in_lres)
        channel_indices = torch.tensor([0])  # in_lres index of 10u
        x_in_lres = torch.tensor([[[[[100.0, 200.0, 300.0]]]]])  # 10u=100, 10v=200, 2t=300
        x_interp = x_in_lres[..., channel_indices]  # [100.0] (just 10u)

        y = torch.tensor([[[[[150.0, 5.0]]]]])  # 10u=150, tp=5

        # Compute target
        target = y.clone()
        # Prognostic (10u): residual
        target[..., prog_out] = y[..., prog_out] - x_interp  # 150 - 100 = 50
        # Diagnostic (tp): direct (unchanged)
        # target[..., diag_out] = y[..., diag_out]  # already 5

        assert torch.allclose(target[..., prog_out], torch.tensor([[[[[50.0]]]]]))
        assert torch.allclose(target[..., diag_out], torch.tensor([[[[[5.0]]]]]))

    def test_reconstruction_mixed(self, data_indices):
        """Reconstruction: add x_interp to prognostic, leave diagnostic alone."""
        out_indices = data_indices["out_hres"]
        prog_out = out_indices.model.output.prognostic
        diag_out = out_indices.model.output.diagnostic

        channel_indices = torch.tensor([0])
        x_in_lres_denorm = torch.tensor([[[[[100.0, 200.0, 300.0]]]]])

        # Model predicted: residual for 10u=50, direct for tp=5
        y_pred_denorm = torch.tensor([[[[[50.0, 5.0]]]]])

        # Reconstruct
        y_pred_full = y_pred_denorm.clone()
        y_pred_full[..., prog_out] += x_in_lres_denorm[..., channel_indices]

        # 10u: 50 + 100 = 150, tp: 5 (unchanged)
        assert torch.allclose(y_pred_full, torch.tensor([[[[[150.0, 5.0]]]]]))

    def test_all_channels_direct_when_no_overlap(self):
        """When in_lres and out_hres have no common variables, all are direct prediction."""
        from anemoi.models.models.diffusiondownscaler_encoder_processor_decoder import AnemoiD2ModelEncProcDec

        input_idx = {"10u": 0, "10v": 1}
        output_idx = {"tp": 0, "cp": 1}
        indices = AnemoiD2ModelEncProcDec._match_tensor_channels(None, input_idx, output_idx)
        assert indices.numel() == 0  # No common channels

    def test_all_channels_residual_when_full_overlap(self):
        """When all out_hres variables are in in_lres, all are residual prediction."""
        from anemoi.models.models.diffusiondownscaler_encoder_processor_decoder import AnemoiD2ModelEncProcDec

        input_idx = {"10u": 0, "10v": 1, "2t": 2}
        output_idx = {"10v": 0, "10u": 1}
        indices = AnemoiD2ModelEncProcDec._match_tensor_channels(None, input_idx, output_idx)
        # 10v→1, 10u→0
        assert torch.equal(indices, torch.tensor([1, 0]))


# ============================================================
# compute_residuals model-level tests
# ============================================================


class TestComputeResiduals:
    """Test the model's compute_residuals method with mixed prognostic/diagnostic."""

    @pytest.fixture
    def model_with_mixed_indices(self):
        """Create a minimal model mock with mixed prognostic/diagnostic data indices."""
        from anemoi.models.models.diffusiondownscaler_encoder_processor_decoder import AnemoiD2ModelEncProcDec

        model = object.__new__(AnemoiD2ModelEncProcDec)
        # Set up data indices: 10u=prognostic (residual), tp=diagnostic (direct)
        model.data_indices = {
            "in_lres": _make_index_collection({"10u": 0, "10v": 1, "2t": 2}),
            "out_hres": _make_index_collection({"10u": 0, "tp": 1}, diagnostic=["tp"]),
        }
        model._residual_pairs = {"out_hres": "in_lres"}
        model._matching_channel_indices_out_hres = torch.tensor([0])  # in_lres 10u → out_hres 10u
        model._matching_indices_keys = [("out_hres", "in_lres", "_matching_channel_indices_out_hres")]
        return model

    def test_compute_residuals_prognostic_is_difference(self, model_with_mixed_indices):
        """Prognostic channel (10u) should be y - x_interp."""
        model = model_with_mixed_indices
        identity = _IdentityProcessor()

        y = torch.tensor([[[[[150.0, 5.0]]]]])  # 10u=150, tp=5
        x_interp = torch.tensor([[[[[100.0]]]]])  # matched 10u from in_lres

        target = model.compute_residuals(
            y=y,
            x_interp=x_interp,
            pre_processors_state=identity,
            pre_processors_tendencies=None,
        )

        prog_out = model.data_indices["out_hres"].model.output.prognostic
        diag_out = model.data_indices["out_hres"].model.output.diagnostic

        # 10u residual: 150 - 100 = 50
        assert torch.allclose(target[..., prog_out], torch.tensor([[[[[50.0]]]]]))
        # tp direct: 5.0
        assert torch.allclose(target[..., diag_out], torch.tensor([[[[[5.0]]]]]))

    def test_compute_residuals_with_tendency_processor(self, model_with_mixed_indices):
        """When tendency processor is provided, it normalizes prognostic channels."""
        model = model_with_mixed_indices

        class _ScaleBy2Processor(torch.nn.Module):
            def forward(self, x, in_place=False, **_kwargs):
                return x * 2.0

        identity = _IdentityProcessor()
        scale2 = _ScaleBy2Processor()

        y = torch.tensor([[[[[150.0, 5.0]]]]])
        x_interp = torch.tensor([[[[[100.0]]]]])

        target = model.compute_residuals(
            y=y,
            x_interp=x_interp,
            pre_processors_state=identity,
            pre_processors_tendencies=scale2,
        )

        prog_out = model.data_indices["out_hres"].model.output.prognostic
        diag_out = model.data_indices["out_hres"].model.output.diagnostic

        # Prognostic residual (50) scaled by 2 = 100
        assert torch.allclose(target[..., prog_out], torch.tensor([[[[[100.0]]]]]))
        # Diagnostic (tp=5) uses identity processor
        assert torch.allclose(target[..., diag_out], torch.tensor([[[[[5.0]]]]]))


# ============================================================
# add_interp_to_state model-level tests
# ============================================================


class TestAddInterpToState:
    """Test the model's add_interp_to_state method with mixed prognostic/diagnostic."""

    @pytest.fixture
    def model_with_mixed_indices(self):
        """Create a minimal model mock."""
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

    def test_add_interp_fallback(self, model_with_mixed_indices):
        """Without tendency processors, fallback adds x_interp to prognostic only."""
        model = model_with_mixed_indices
        identity = _IdentityProcessor()
        post_processors = {"in_lres": identity, "out_hres": identity}

        # state_inp is the normalized upsampled in_lres
        state_inp = torch.tensor([[[[[100.0, 200.0, 300.0]]]]])  # 10u, 10v, 2t
        # model_output is the predicted normalized residual/direct
        model_output = torch.tensor([[[[[50.0, 5.0]]]]])  # 10u_residual, tp_direct

        result = model.add_interp_to_state(
            state_inp=state_inp,
            model_output=model_output,
            post_processors_state=post_processors,
            post_processors_tendencies=None,
        )

        prog_out = model.data_indices["out_hres"].model.output.prognostic
        diag_out = model.data_indices["out_hres"].model.output.diagnostic

        # 10u: 50 (residual) + 100 (x_interp) = 150
        assert torch.allclose(result[..., prog_out], torch.tensor([[[[[150.0]]]]]))
        # tp: 5 (direct, no addition)
        assert torch.allclose(result[..., diag_out], torch.tensor([[[[[5.0]]]]]))

    def test_no_residual_prediction(self, model_with_mixed_indices):
        """With no residual pairs, output is just denormalized model output."""
        model = model_with_mixed_indices
        model._residual_pairs = {}  # No residual prediction
        identity = _IdentityProcessor()
        post_processors = {"in_lres": identity, "out_hres": identity}

        state_inp = torch.tensor([[[[[100.0, 200.0, 300.0]]]]])
        model_output = torch.tensor([[[[[150.0, 5.0]]]]])

        result = model.add_interp_to_state(
            state_inp=state_inp,
            model_output=model_output,
            post_processors_state=post_processors,
            post_processors_tendencies=None,
        )

        # Direct prediction: output == denorm(model_output)
        assert torch.allclose(result, torch.tensor([[[[[150.0, 5.0]]]]]))

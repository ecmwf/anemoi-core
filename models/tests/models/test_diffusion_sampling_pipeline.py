# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import pytest
import torch

from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportModelEncProcDec
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportTendModelEncProcDec
from anemoi.models.samplers import transport_samplers
from anemoi.models.transport import DiffusionModelObjective
from anemoi.models.transport import EdmSettings
from anemoi.models.transport import StochasticInterpolantModelObjective
from anemoi.models.transport import TransportSourceBuilder
from anemoi.models.transport import TransportSourceRequest
from anemoi.models.transport import TransportSourceSettings


class IdentityProcessor(torch.nn.Module):
    def forward(self, x: torch.Tensor, in_place: bool = True, inverse: bool = False, **kwargs):
        del inverse, kwargs
        if not in_place:
            x = x.clone()
        return x


def _transport_model_stub() -> AnemoiTransportModelEncProcDec:
    model = AnemoiTransportModelEncProcDec.__new__(AnemoiTransportModelEncProcDec)
    model.transport_source = TransportSourceBuilder()
    return model


def test_transport_conditioning_embedding_uses_compact_condition_width() -> None:
    model = _transport_model_stub()
    model._graph_name_hidden = "hidden"
    model.node_attributes = SimpleNamespace(num_nodes={"data": 4, "hidden": 5})
    cond_dim = 8
    model._embed_noise_conditioning = lambda sigma: torch.ones(
        (*sigma.shape[:-1], cond_dim),
        device=sigma.device,
        dtype=sigma.dtype,
    )

    x = {"data": torch.empty(2, 2, 3, 7, 1)}
    condition = {"data": torch.zeros(2, 1, 3, 1, 1)}

    fwd_mapper_kwargs, processor_kwargs, bwd_mapper_kwargs = model._build_conditioning_kwargs(x, condition)

    data_cond, hidden_cond = fwd_mapper_kwargs["data"]["cond"]
    hidden_back_cond, data_back_cond = bwd_mapper_kwargs["data"]["cond"]
    assert data_cond.shape == data_back_cond.shape == (2 * 3 * 4, cond_dim)
    assert hidden_cond.shape == hidden_back_cond.shape == processor_kwargs["cond"].shape == (2 * 3 * 5, cond_dim)


def test_transport_conditioning_rejects_expanded_condition_shape() -> None:
    model = _transport_model_stub()
    expanded_condition = {"data": torch.zeros(2, 4, 3, 7, 1)}

    with pytest.raises(AssertionError, match="Expected condition to have shape"):
        model._assert_condition_shapes(expanded_condition)


def test_before_sampling_non_sharded_returns_none_grid_shapes() -> None:
    model = _transport_model_stub()

    batch = {"data": torch.randn(2, 4, 3, 2)}
    pre_processors = {"data": IdentityProcessor()}

    (xs,), grid_shard_sizes = model._before_sampling(
        batch,
        pre_processors,
        n_step_input=3,
        model_comm_group=None,
    )

    assert grid_shard_sizes is None
    assert xs["data"].shape == (2, 3, 1, 3, 2)


def test_predict_step_iterates_items_and_casts_each_dataset_dtype() -> None:
    model = _transport_model_stub()

    batch = {
        "ds_a": torch.randn(1, 3, 4, 2, dtype=torch.float32),
        "ds_b": torch.randn(1, 3, 4, 2, dtype=torch.bfloat16),
    }

    x_for_sampling = {
        "ds_a": torch.randn(1, 2, 1, 4, 2, dtype=torch.float32),
        "ds_b": torch.randn(1, 2, 1, 4, 2, dtype=torch.bfloat16),
    }

    model._before_sampling = lambda *_args, **_kwargs: ((x_for_sampling,), None)
    model.sample = lambda *_args, **_kwargs: {
        "ds_a": torch.randn(1, 2, 1, 4, 3, dtype=torch.float64),
        "ds_b": torch.randn(1, 2, 1, 4, 3, dtype=torch.float64),
    }

    def _after_sampling_spy(
        out,
        _post_processors,
        _before_sampling_data,
        _model_comm_group,
        _grid_shard_sizes,
        _gather_out,
        **_kwargs,
    ):
        assert out["ds_a"].dtype == batch["ds_a"].dtype
        assert out["ds_b"].dtype == batch["ds_b"].dtype
        return out

    model._after_sampling = _after_sampling_spy

    out = model.predict_step(
        batch=batch,
        pre_processors={"ds_a": IdentityProcessor(), "ds_b": IdentityProcessor()},
        post_processors={"ds_a": IdentityProcessor(), "ds_b": IdentityProcessor()},
        n_step_input=2,
    )

    assert out["ds_a"].dtype == torch.float32
    assert out["ds_b"].dtype == torch.bfloat16


def test_sample_passes_zero_terminated_schedule_to_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyScheduler(transport_samplers.NoiseScheduler):
        def __init__(self, sigma_max: float, sigma_min: float, num_steps: int, **kwargs):
            super().__init__(sigma_max=sigma_max, sigma_min=sigma_min, num_steps=num_steps)
            del kwargs

        def _build_schedule(self, device=None, dtype_compute: torch.dtype = torch.float64, **kwargs):
            del kwargs
            return torch.linspace(1.0, 0.1, self.num_steps, device=device, dtype=dtype_compute)

    class DummySampler:
        def __init__(self, dtype: torch.dtype = torch.float64, **kwargs):
            del kwargs
            self.dtype = dtype

        def sample(
            self,
            x: dict[str, torch.Tensor],
            y: dict[str, torch.Tensor],
            sigmas: torch.Tensor,
            denoising_fn,
            model_comm_group=None,
            grid_shard_sizes=None,
            **kwargs,
        ):
            del denoising_fn, model_comm_group, grid_shard_sizes, kwargs
            assert isinstance(sigmas, torch.Tensor)
            assert sigmas.shape == (5,)
            assert sigmas[-1] == 0.0
            for dataset_name, y_data in y.items():
                assert y_data.dtype == sigmas.dtype
                assert y_data.shape[:4] == (
                    x[dataset_name].shape[0],
                    2,
                    x[dataset_name].shape[2],
                    x[dataset_name].shape[-2],
                )
            return y

    model = _transport_model_stub()
    model.inference_defaults = SimpleNamespace(
        noise_scheduler={
            "schedule_type": "dummy",
            "sigma_max": 1.0,
            "sigma_min": 0.1,
            "num_steps": 4,
        },
        diffusion_sampler={"sampler": "dummy"},
    )
    model.n_step_output = 2
    model.num_output_channels = {"ds_a": 3, "ds_b": 4}
    model.transport_objective = DiffusionModelObjective()
    model.edm = EdmSettings(sigma_data=1.0)
    model.forward_network = lambda *_args, **_kwargs: None

    monkeypatch.setitem(transport_samplers.NOISE_SCHEDULERS, "dummy", DummyScheduler)
    monkeypatch.setitem(transport_samplers.DIFFUSION_SAMPLERS, "dummy", DummySampler)

    x = {
        "ds_a": torch.randn(1, 3, 1, 5, 6, dtype=torch.float32),
        "ds_b": torch.randn(1, 3, 1, 7, 5, dtype=torch.float32),
    }

    out = model.sample(x)
    assert set(out.keys()) == {"ds_a", "ds_b"}


def test_sample_dispatches_stochastic_interpolant_to_euler_maruyama_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyStochasticInterpolantSampler:
        def __init__(self, dtype: torch.dtype = torch.float64, **kwargs):
            del kwargs
            self.dtype = dtype

        def sample(
            self,
            x: dict[str, torch.Tensor],
            y: dict[str, torch.Tensor],
            times: torch.Tensor,
            vector_field_fn,
            model_comm_group=None,
            grid_shard_sizes=None,
            **kwargs,
        ):
            sigma_fn = kwargs["sigma_fn"]
            assert times.shape == (4,)
            assert times[0] == 0.0
            assert times[-1] == 1.0
            torch.testing.assert_close(sigma_fn(torch.tensor(0.0)), torch.tensor(0.0))
            torch.testing.assert_close(sigma_fn(torch.tensor(0.5)), torch.sqrt(torch.tensor(0.5)))
            torch.testing.assert_close(y["ds_a"], torch.zeros_like(y["ds_a"]))
            return vector_field_fn(
                x,
                y,
                {"ds_a": torch.zeros(1, 1, 1, 1, 1)},
                model_comm_group,
                grid_shard_sizes,
            )

    model = _transport_model_stub()
    model.transport_objective = StochasticInterpolantModelObjective()
    model.inference_defaults = SimpleNamespace(noise_scheduler={"num_steps": 3})
    model.stochastic_interpolant = SimpleNamespace(sigma_schedule="brownian_bridge", noise_scale=1.0)
    model.n_step_output = 2
    model.num_output_channels = {"ds_a": 3}
    model.forward_network = lambda _x, y, *_args, **_kwargs: y
    model.build_sampling_source = lambda x, **_kwargs: {
        "ds_a": torch.zeros(
            x["ds_a"].shape[0],
            1,
            x["ds_a"].shape[2],
            x["ds_a"].shape[-2],
            model.num_output_channels["ds_a"],
            device=x["ds_a"].device,
            dtype=x["ds_a"].dtype,
        )
    }

    monkeypatch.setitem(
        transport_samplers.STOCHASTIC_INTERPOLANT_SAMPLERS, "euler_maruyama", DummyStochasticInterpolantSampler
    )

    x = {"ds_a": torch.randn(1, 3, 1, 5, 6, dtype=torch.float32)}

    out = model.sample(x)

    assert set(out.keys()) == {"ds_a"}
    assert out["ds_a"].shape == (1, 2, 1, 5, 3)


def test_sample_can_use_deterministic_vector_field_sampler_for_stochastic_interpolant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyVectorFieldSampler:
        def __init__(self, dtype: torch.dtype = torch.float64, **kwargs):
            del kwargs
            self.dtype = dtype

        def sample(
            self,
            x: dict[str, torch.Tensor],
            y: dict[str, torch.Tensor],
            times: torch.Tensor,
            vector_field_fn,
            model_comm_group=None,
            grid_shard_sizes=None,
            **kwargs,
        ):
            del kwargs
            assert times.shape == (4,)
            assert times[0] == 0.0
            assert times[-1] == 1.0
            torch.testing.assert_close(y["ds_a"], torch.zeros_like(y["ds_a"]))
            return vector_field_fn(
                x,
                y,
                {"ds_a": torch.zeros(1, 1, 1, 1, 1)},
                model_comm_group,
                grid_shard_sizes,
            )

    model = _transport_model_stub()
    model.transport_objective = StochasticInterpolantModelObjective()
    model.inference_defaults = SimpleNamespace()
    model.n_step_output = 2
    model.num_output_channels = {"ds_a": 3}
    model.forward_network = lambda _x, y, *_args, **_kwargs: y
    model.build_sampling_source = lambda x, **_kwargs: {
        "ds_a": torch.zeros(
            x["ds_a"].shape[0],
            1,
            x["ds_a"].shape[2],
            x["ds_a"].shape[-2],
            model.num_output_channels["ds_a"],
            device=x["ds_a"].device,
            dtype=x["ds_a"].dtype,
        )
    }

    monkeypatch.setitem(transport_samplers.VECTOR_FIELD_SAMPLERS, "dummy_vector", DummyVectorFieldSampler)

    x = {"ds_a": torch.randn(1, 3, 1, 5, 6, dtype=torch.float32)}

    out = model.sample(x, sampler_params={"sampler": "dummy_vector", "num_steps": 3})

    assert set(out.keys()) == {"ds_a"}
    assert out["ds_a"].shape == (1, 2, 1, 5, 3)


def test_transport_source_settings_reads_source_from_config() -> None:
    config = {"source": {"kind": "gaussian", "scale": 0.5, "noise_scale": 0.1}}

    settings = TransportSourceSettings.from_config(config)

    assert settings.kind == "gaussian"
    assert settings.scale == 0.5
    assert settings.noise_scale == 0.1


def test_transport_source_builder_does_not_build_unselected_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = TransportSourceBuilder(TransportSourceSettings(kind="gaussian", scale=2.0))
    target = {"data": torch.zeros(1, 1, 1, 2, 1)}

    def reference_source_factory() -> dict[str, torch.Tensor]:
        raise AssertionError("reference source should not be built")

    def fake_randn(shape, device=None, dtype=None):
        return torch.full(shape, 3.0, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randn", fake_randn)

    source = builder.build(
        TransportSourceRequest.from_tensors(
            target,
            default_kind="reference_state",
            source_factories={"reference_state": reference_source_factory},
        )
    )

    torch.testing.assert_close(source["data"], torch.full_like(target["data"], 6.0))


def test_transport_source_builder_postprocesses_reference_source(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = TransportSourceBuilder(TransportSourceSettings(kind="reference_state", scale=0.5, noise_scale=0.25))
    target = {"data": torch.zeros(1, 1, 1, 2, 1, dtype=torch.float64)}
    reference = {"data": torch.full_like(target["data"], 4.0)}

    monkeypatch.setattr(
        torch, "randn", lambda shape, device=None, dtype=None: torch.full(shape, 2.0, device=device, dtype=dtype)
    )

    source = builder.build(
        TransportSourceRequest.from_tensors(
            target,
            default_kind="gaussian",
            source_factories={"reference_state": lambda: reference},
        )
    )

    assert source["data"].dtype == target["data"].dtype
    torch.testing.assert_close(source["data"], torch.full_like(target["data"], 2.5))
    torch.testing.assert_close(reference["data"], torch.full_like(target["data"], 4.0))


def test_transport_source_builder_postprocesses_gaussian_source(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = TransportSourceBuilder(TransportSourceSettings(kind="gaussian", scale=2.0, noise_scale=0.25))
    target = {"data": torch.zeros(1, 1, 1, 2, 1)}

    random_values = iter((3.0, 4.0))
    monkeypatch.setattr(
        torch,
        "randn",
        lambda shape, device=None, dtype=None: torch.full(shape, next(random_values), device=device, dtype=dtype),
    )

    source = builder.build(TransportSourceRequest.from_tensors(target, default_kind="zero"))

    torch.testing.assert_close(source["data"], torch.full_like(target["data"], 7.0))


def test_transport_source_builder_can_jitter_zero_source(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = TransportSourceBuilder(TransportSourceSettings(kind="zero", scale=2.0, noise_scale=0.25))
    target = {"data": torch.zeros(1, 1, 1, 2, 1)}

    monkeypatch.setattr(
        torch, "randn", lambda shape, device=None, dtype=None: torch.full(shape, 2.0, device=device, dtype=dtype)
    )

    source = builder.build(TransportSourceRequest.from_tensors(target, default_kind="gaussian"))

    torch.testing.assert_close(source["data"], torch.full_like(target["data"], 0.5))


def test_tendency_sampling_source_can_be_gaussian(monkeypatch: pytest.MonkeyPatch) -> None:
    model = AnemoiTransportTendModelEncProcDec.__new__(AnemoiTransportTendModelEncProcDec)
    model.transport_source = TransportSourceBuilder(TransportSourceSettings(kind="gaussian"))
    model.n_step_output = 2
    model.num_output_channels = {"ds_a": 3}

    def fake_randn(shape, device=None, dtype=None):
        return torch.full(shape, 2.0, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randn", fake_randn)

    x = {"ds_a": torch.zeros(1, 3, 1, 5, 6, dtype=torch.float32)}

    source = model.build_sampling_source(x)

    assert source["ds_a"].shape == (1, 2, 1, 5, 3)
    torch.testing.assert_close(source["ds_a"], torch.full_like(source["ds_a"], 2.0))


def test_tendency_sampling_source_can_use_reference_state() -> None:
    model = AnemoiTransportTendModelEncProcDec.__new__(AnemoiTransportTendModelEncProcDec)
    model.transport_source = TransportSourceBuilder(TransportSourceSettings(kind="reference_state"))
    model.n_step_output = 2
    model.num_output_channels = {"ds_a": 2}
    model.data_indices = {
        "ds_a": SimpleNamespace(
            model=SimpleNamespace(
                output=SimpleNamespace(ordered_names=("a", "b")),
                input=SimpleNamespace(positions_for_names=lambda names: [0, 2]),
            ),
        ),
    }
    x_data = torch.arange(1 * 3 * 1 * 5 * 4, dtype=torch.float32).reshape(1, 3, 1, 5, 4)
    x = {"ds_a": x_data}

    source = model.build_sampling_source(x)

    expected = x_data[:, -1:, :, :, :].index_select(-1, torch.tensor([0, 2])).expand(-1, 2, -1, -1, -1)
    torch.testing.assert_close(source["ds_a"], expected)


def test_stochastic_interpolant_objective_returns_raw_drift_prediction() -> None:
    """The stochastic-interpolant model objective leaves drift predictions in model-output space."""
    interpolant = {"data": torch.full((1, 1, 1, 2, 1), 2.0)}
    time_level = {"data": torch.full_like(interpolant["data"], 0.25)}
    drift = {"data": torch.full_like(interpolant["data"], 0.5)}

    def _forward_network(
        _x: dict[str, torch.Tensor],
        conditioned_target: dict[str, torch.Tensor],
        condition: dict[str, torch.Tensor],
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        assert conditioned_target is interpolant
        assert condition is time_level
        return drift

    model = SimpleNamespace(forward_network=_forward_network)

    out = StochasticInterpolantModelObjective().forward(
        model,
        {"data": torch.zeros_like(interpolant["data"])},
        interpolant,
        time_level,
    )

    assert out is drift


@pytest.mark.parametrize(
    ("sampler_name", "sampler_config"),
    [
        ("heun", {"S_churn": 0.0, "S_min": 0.0, "S_max": float("inf"), "S_noise": 1.0}),
        ("dpmpp_2m", {}),
    ],
)
def test_sample_end_to_end_multi_dataset_real_sampler(
    sampler_name: str,
    sampler_config: dict[str, float],
) -> None:
    model = _transport_model_stub()
    model.inference_defaults = SimpleNamespace(
        noise_scheduler={
            "schedule_type": "linear",
            "sigma_max": 1.0,
            "sigma_min": 0.02,
            "num_steps": 6,
        },
        diffusion_sampler={"sampler": sampler_name, **sampler_config},
    )
    model.n_step_output = 2
    model.num_output_channels = {"dataset_a": 3, "dataset_b": 2}
    model.transport_objective = DiffusionModelObjective()
    model.edm = EdmSettings(sigma_data=1.0)

    def _network(
        x: dict[str, torch.Tensor],
        conditioned_target: dict[str, torch.Tensor],
        condition: dict[str, torch.Tensor],
        model_comm_group=None,
        grid_shard_sizes=None,
    ) -> dict[str, torch.Tensor]:
        del model_comm_group, grid_shard_sizes
        out = {}
        for dataset_name, target_data in conditioned_target.items():
            condition_data = condition[dataset_name]
            assert condition_data.shape == (
                target_data.shape[0],
                1,
                target_data.shape[2],
                1,
                1,
            )
            assert condition_data.dtype == target_data.dtype == x[dataset_name].dtype
            out[dataset_name] = 0.8 * target_data + 0.02 * condition_data
        return out

    model.forward_network = _network

    x = {
        "dataset_a": torch.randn(2, 3, 1, 5, 4, dtype=torch.float32),
        "dataset_b": torch.randn(2, 2, 1, 7, 6, dtype=torch.bfloat16),
    }

    out = model.sample(x)

    assert set(out.keys()) == set(x.keys())
    assert out["dataset_a"].shape == (2, 2, 1, 5, 3)
    assert out["dataset_b"].shape == (2, 2, 1, 7, 2)
    assert out["dataset_a"].dtype == x["dataset_a"].dtype
    assert out["dataset_b"].dtype == x["dataset_b"].dtype
    assert torch.isfinite(out["dataset_a"]).all()
    assert torch.isfinite(out["dataset_b"]).all()

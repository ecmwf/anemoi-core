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

from anemoi.models.models.diffusion_encoder_processor_decoder import AnemoiDiffusionModelEncProcDec
from anemoi.models.samplers import diffusion_samplers


class IdentityProcessor(torch.nn.Module):
    def forward(self, x: torch.Tensor, in_place: bool = True, inverse: bool = False, **kwargs):
        del inverse, kwargs
        if not in_place:
            x = x.clone()
        return x


def test_before_sampling_non_sharded_returns_none_grid_shapes() -> None:
    model = AnemoiDiffusionModelEncProcDec.__new__(AnemoiDiffusionModelEncProcDec)

    batch = {"data": torch.randn(2, 4, 3, 2)}
    pre_processors = {"data": IdentityProcessor()}

    (xs,), grid_shard_shapes = model._before_sampling(
        batch,
        pre_processors,
        n_step_input=3,
        model_comm_group=None,
    )

    assert grid_shard_shapes is None
    assert xs["data"].shape == (2, 3, 1, 3, 2)


def test_predict_step_iterates_items_and_casts_each_dataset_dtype() -> None:
    model = AnemoiDiffusionModelEncProcDec.__new__(AnemoiDiffusionModelEncProcDec)

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
        _grid_shard_shapes,
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


def test_sample_initialization_uses_per_dataset_schedule_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyScheduler:
        def __init__(self, sigma_max: float, sigma_min: float, num_steps: int, **kwargs):
            del sigma_max, sigma_min, kwargs
            self.num_steps = num_steps

        def get_schedule(self, device=None, dtype_compute: torch.dtype = torch.float64, **kwargs):
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
            grid_shard_shapes=None,
            **kwargs,
        ):
            del denoising_fn, model_comm_group, grid_shard_shapes, kwargs
            assert isinstance(sigmas, torch.Tensor)
            for dataset_name, y_data in y.items():
                assert y_data.dtype == sigmas.dtype
                assert y_data.shape[:4] == (
                    x[dataset_name].shape[0],
                    2,
                    x[dataset_name].shape[2],
                    x[dataset_name].shape[-2],
                )
            return y

    model = AnemoiDiffusionModelEncProcDec.__new__(AnemoiDiffusionModelEncProcDec)
    model.inference_defaults = SimpleNamespace(
        noise_scheduler={"schedule_type": "dummy", "sigma_max": 1.0, "sigma_min": 0.1, "num_steps": 4},
        diffusion_sampler={"sampler": "dummy"},
    )
    model.n_step_output = 2
    model.num_output_channels = {"ds_a": 3, "ds_b": 4}
    model.fwd_with_preconditioning = lambda *_args, **_kwargs: None

    monkeypatch.setitem(diffusion_samplers.NOISE_SCHEDULERS, "dummy", DummyScheduler)
    monkeypatch.setitem(diffusion_samplers.DIFFUSION_SAMPLERS, "dummy", DummySampler)

    x = {
        "ds_a": torch.randn(1, 3, 1, 5, 6, dtype=torch.float32),
        "ds_b": torch.randn(1, 3, 1, 7, 5, dtype=torch.float32),
    }

    out = model.sample(x)
    assert set(out.keys()) == {"ds_a", "ds_b"}


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
    model = AnemoiDiffusionModelEncProcDec.__new__(AnemoiDiffusionModelEncProcDec)
    model.inference_defaults = SimpleNamespace(
        noise_scheduler={"schedule_type": "linear", "sigma_max": 1.0, "sigma_min": 0.0, "num_steps": 6},
        diffusion_sampler={"sampler": sampler_name, **sampler_config},
    )
    model.n_step_output = 2
    model.num_output_channels = {"dataset_a": 3, "dataset_b": 2}

    def _denoiser(
        x: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        sigma: dict[str, torch.Tensor],
        model_comm_group=None,
        grid_shard_shapes=None,
    ) -> dict[str, torch.Tensor]:
        del model_comm_group, grid_shard_shapes
        out = {}
        for dataset_name, y_data in y.items():
            sigma_data = sigma[dataset_name]
            assert sigma_data.shape == (y_data.shape[0], 1, y_data.shape[2], 1, 1)
            assert sigma_data.dtype == y_data.dtype == x[dataset_name].dtype
            out[dataset_name] = 0.8 * y_data + 0.02 * sigma_data
        return out

    model.fwd_with_preconditioning = _denoiser

    x = {
        "dataset_a": torch.randn(2, 3, 1, 5, 4, dtype=torch.float32),
        "dataset_b": torch.randn(1, 2, 4, 7, 6, dtype=torch.bfloat16),
    }

    out = model.sample(x)

    assert set(out.keys()) == set(x.keys())
    assert out["dataset_a"].shape == (2, 2, 1, 5, 3)
    assert out["dataset_b"].shape == (1, 2, 4, 7, 2)
    assert out["dataset_a"].dtype == x["dataset_a"].dtype
    assert out["dataset_b"].dtype == x["dataset_b"].dtype
    assert torch.isfinite(out["dataset_a"]).all()
    assert torch.isfinite(out["dataset_b"]).all()

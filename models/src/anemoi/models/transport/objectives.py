# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from typing import Any
from typing import Optional

import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.samplers import transport_samplers
from anemoi.models.transport.paths import unit_time_grid

LOGGER = logging.getLogger(__name__)


def _get_inference_defaults_section(model: Any, *names: str) -> dict:
    defaults = getattr(model, "inference_defaults", {})
    for name in names:
        section = defaults.get(name) if isinstance(defaults, dict) else getattr(defaults, name, None)
        if section is not None:
            return dict(section)
    return {}


def _get_default_num_steps(model: Any, fallback: int = 50) -> int:
    noise_scheduler_config = _get_inference_defaults_section(model, "noise_scheduler")
    return int(noise_scheduler_config.get("num_steps", fallback))


class TransportModelObjective:
    """How a transport model runs its forward pass and inference sampler."""

    def forward(
        self,
        model: Any,
        x: dict[str, torch.Tensor],
        conditioned_target: dict[str, torch.Tensor],
        condition: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def sample(
        self,
        model: Any,
        x: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        noise_scheduler_params: Optional[dict] = None,
        sampler_params: Optional[dict] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class EDMDiffusionModelObjective(TransportModelObjective):
    """EDM diffusion model objective."""

    def forward(
        self,
        model: Any,
        x: dict[str, torch.Tensor],
        y_noised: dict[str, torch.Tensor],
        sigma: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        c_skip, c_out, c_in, c_noise = self._get_preconditioning(model, sigma, model.edm.sigma_data)
        pred = model._forward_transport_network(
            x,
            {key: c_in[key] * y_noised[key] for key in y_noised.keys()},
            c_noise,
            model_comm_group=model_comm_group,
            grid_shard_sizes=grid_shard_sizes,
            **kwargs,
        )
        return {key: c_skip[key] * y_noised[key] + c_out[key] * pred[key] for key in y_noised.keys()}

    def sample(
        self,
        model: Any,
        x: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        noise_scheduler_params: Optional[dict] = None,
        sampler_params: Optional[dict] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Sample from an EDM diffusion model."""
        x_device = next(iter(x.values())).device

        noise_scheduler_config = dict(model.inference_defaults.noise_scheduler)
        if noise_scheduler_params is not None:
            noise_scheduler_config.update(noise_scheduler_params)

        LOGGER.debug("noise_scheduler_config: %s", noise_scheduler_config)

        actual_schedule_type = noise_scheduler_config.pop("schedule_type")
        if actual_schedule_type not in transport_samplers.NOISE_SCHEDULERS:
            raise ValueError(f"Unknown schedule type: {actual_schedule_type}")

        scheduler_cls = transport_samplers.NOISE_SCHEDULERS[actual_schedule_type]
        scheduler = scheduler_cls(**noise_scheduler_config)

        source = model.build_sampling_source(
            x,
            model_comm_group=model_comm_group,
            grid_shard_sizes=grid_shard_sizes,
        )
        sigmas, y_init = {}, {}
        for dataset_name in x:
            sigma_i = scheduler.get_schedule(x_device, torch.float64)
            sigmas[dataset_name] = sigma_i
            y_init[dataset_name] = source[dataset_name].to(dtype=sigma_i.dtype) * sigma_i[0]

        edm_diffusion_sampler_config = _get_inference_defaults_section(model, "edm_diffusion_sampler")
        if sampler_params is not None:
            edm_diffusion_sampler_config.update(sampler_params)

        LOGGER.debug("edm_diffusion_sampler_config: %s", edm_diffusion_sampler_config)

        actual_sampler = edm_diffusion_sampler_config.pop("sampler")
        if actual_sampler not in transport_samplers.DIFFUSION_SAMPLERS:
            raise ValueError(f"Unknown sampler: {actual_sampler}")

        sampler_cls = transport_samplers.DIFFUSION_SAMPLERS[actual_sampler]
        sampler_instance = sampler_cls(dtype=next(iter(sigmas.values())).dtype, **edm_diffusion_sampler_config)

        sigmas_ref = next(iter(sigmas.values()))
        for dataset_name, sigmas_i in sigmas.items():
            if not torch.allclose(sigmas_i, sigmas_ref):
                LOGGER.warning(
                    "Sigma schedules differ between datasets. Dataset '%s' has a different schedule.",
                    dataset_name,
                )

        def denoising_fn(
            x_arg: dict[str, torch.Tensor],
            y_arg: dict[str, torch.Tensor],
            sigma_arg: dict[str, torch.Tensor],
            comm_arg: Optional[ProcessGroup] = None,
            shard_sizes_arg: DatasetShardSizes | None = None,
        ) -> dict[str, torch.Tensor]:
            return self.forward(
                model,
                x_arg,
                y_arg,
                sigma_arg,
                model_comm_group=comm_arg,
                grid_shard_sizes=shard_sizes_arg,
            )

        return sampler_instance.sample(
            x,
            y_init,
            sigmas_ref,
            denoising_fn,
            model_comm_group,
            grid_shard_sizes=grid_shard_sizes,
        )

    @staticmethod
    def _get_preconditioning(
        model: Any,
        sigma: dict[str, torch.Tensor],
        sigma_data: float,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        """Compute EDM preconditioning factors."""
        batch_size, ensemble_size = model._assert_condition_shapes(sigma)
        dataset_names = list(sigma)
        base_view = (batch_size, 1, ensemble_size, 1, 1)

        c_skip, c_out, c_in, c_noise = {}, {}, {}, {}
        # this is per dataset ; future proof but currently not used -> we assume the same sigma for each dataset later
        for dataset_name in dataset_names:
            sigma_dataset = sigma[dataset_name]
            sigma_base = sigma_dataset[:, 0, :, 0, 0]
            c_skip_base = sigma_data**2 / (sigma_base**2 + sigma_data**2)
            c_out_base = sigma_base * sigma_data / (sigma_base**2 + sigma_data**2) ** 0.5
            c_in_base = 1.0 / (sigma_data**2 + sigma_base**2) ** 0.5
            c_noise_base = sigma_base.log() / 4.0
            shape_x = sigma_dataset.shape
            c_skip[dataset_name] = c_skip_base.view(base_view).expand(shape_x)
            c_out[dataset_name] = c_out_base.view(base_view).expand(shape_x)
            c_in[dataset_name] = c_in_base.view(base_view).expand(shape_x)
            c_noise[dataset_name] = c_noise_base.view(base_view).expand(shape_x)

        return c_skip, c_out, c_in, c_noise


class StochasticInterpolantModelObjective(TransportModelObjective):
    """Stochastic-interpolant model objective that predicts bridge drift."""

    def forward(
        self,
        model: Any,
        x: dict[str, torch.Tensor],
        interpolant_state: dict[str, torch.Tensor],
        time_level: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        return model._forward_transport_network(
            x,
            interpolant_state,
            time_level,
            model_comm_group=model_comm_group,
            grid_shard_sizes=grid_shard_sizes,
            **kwargs,
        )

    def sample(
        self,
        model: Any,
        x: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        noise_scheduler_params: Optional[dict] = None,
        sampler_params: Optional[dict] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        x_device = next(iter(x.values())).device

        si_sampler_config = _get_inference_defaults_section(model, "stochastic_interpolant_sampler")
        if noise_scheduler_params is not None:
            si_sampler_config.update(noise_scheduler_params)
        if sampler_params is not None:
            si_sampler_config.update(sampler_params)

        actual_sampler = si_sampler_config.pop("sampler", "heun")
        num_steps = si_sampler_config.pop("num_steps", _get_default_num_steps(model))

        source = model.build_sampling_source(
            x,
            model_comm_group=model_comm_group,
            grid_shard_sizes=grid_shard_sizes,
        )
        y_init = source

        times = unit_time_grid(int(num_steps), device=x_device, dtype=torch.float64)

        def transport_fn(
            x_arg: dict[str, torch.Tensor],
            y_arg: dict[str, torch.Tensor],
            time_arg: dict[str, torch.Tensor],
            comm_arg: Optional[ProcessGroup] = None,
            shard_sizes_arg: DatasetShardSizes | None = None,
        ) -> dict[str, torch.Tensor]:
            return self.forward(
                model,
                x_arg,
                y_arg,
                time_arg,
                model_comm_group=comm_arg,
                grid_shard_sizes=shard_sizes_arg,
            )

        if actual_sampler not in transport_samplers.VECTOR_FIELD_SAMPLERS:
            raise ValueError(f"Unknown stochastic-interpolant sampler: {actual_sampler}")

        sampler_cls = transport_samplers.VECTOR_FIELD_SAMPLERS[actual_sampler]
        sampler_instance = sampler_cls(dtype=times.dtype, **si_sampler_config)
        return sampler_instance.sample(
            x,
            y_init,
            times,
            transport_fn,
            model_comm_group,
            grid_shard_sizes=grid_shard_sizes,
        )


TRANSPORT_MODEL_OBJECTIVES = {
    "edm_diffusion": EDMDiffusionModelObjective,
    "stochastic_interpolant": StochasticInterpolantModelObjective,
}


def get_transport_model_objective(name: str) -> TransportModelObjective:
    try:
        return TRANSPORT_MODEL_OBJECTIVES[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown transport model objective: {name}") from exc

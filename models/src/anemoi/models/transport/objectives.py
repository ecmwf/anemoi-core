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
from anemoi.models.transport.schedules import SIGMA_SCHEDULES
from anemoi.models.transport.schedules import TIME_SCHEDULES

LOGGER = logging.getLogger(__name__)


def _get_inference_defaults_section(model: Any, name: str) -> dict:
    section = model.inference_defaults.get(name)
    return dict(section) if section is not None else {}


def _get_inference_config(model: Any, section: str, overrides: Optional[dict]) -> dict:
    config = _get_inference_defaults_section(model, section)
    if overrides is not None:
        config.update(overrides)
    LOGGER.debug("%s_config: %s", section, config)
    return config


def _resolve_registry_entry(config: dict, selector: str, registry: dict, context: str) -> type:
    if selector not in config:
        raise ValueError(f"{context} must define '{selector}'.")
    entry_name = config.pop(selector)
    if entry_name not in registry:
        raise ValueError(f"Unknown {context}: {entry_name}")
    return registry[entry_name]


def _build_inference_schedule(
    model: Any,
    registry: dict,
    context: str,
    schedule_params: Optional[dict],
    device: torch.device,
) -> torch.Tensor:
    schedule_config = _get_inference_config(model, "sampling_schedule", schedule_params)
    scheduler_cls = _resolve_registry_entry(schedule_config, "schedule_type", registry, context)
    scheduler = scheduler_cls(**schedule_config)
    return scheduler.get_schedule(device, torch.float64)


def _build_inference_sampler(
    model: Any,
    registry: dict,
    context: str,
    sampler_params: Optional[dict],
    dtype: torch.dtype,
) -> Any:
    sampler_config = _get_inference_config(model, "sampler", sampler_params)
    sampler_cls = _resolve_registry_entry(sampler_config, "sampler", registry, context)
    return sampler_cls(dtype=dtype, **sampler_config)


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
        schedule_params: Optional[dict] = None,
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
        schedule_params: Optional[dict] = None,
        sampler_params: Optional[dict] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Sample from an EDM diffusion model."""
        x_device = next(iter(x.values())).device

        source = model.build_sampling_source(
            x,
            model_comm_group=model_comm_group,
            grid_shard_sizes=grid_shard_sizes,
        )
        sigma_schedule = _build_inference_schedule(
            model,
            SIGMA_SCHEDULES,
            "EDM sampling schedule",
            schedule_params,
            x_device,
        )
        # Sample only the datasets that seeded a source; conditioning-only datasets are absent from
        # ``source`` (build_sampling_source drops them) and must never enter the denoising loop.
        y_init = {
            dataset_name: source_data.to(dtype=sigma_schedule.dtype) * sigma_schedule[0]
            for dataset_name, source_data in source.items()
        }

        sampler_instance = _build_inference_sampler(
            model,
            transport_samplers.DIFFUSION_SAMPLERS,
            "EDM sampler",
            sampler_params,
            sigma_schedule.dtype,
        )

        def denoising_fn(
            x_arg: dict[str, torch.Tensor],
            y_arg: dict[str, torch.Tensor],
            sigma_arg: dict[str, torch.Tensor],
            comm_arg: Optional[ProcessGroup] = None,
            shard_sizes_arg: DatasetShardSizes | None = None,
        ) -> dict[str, torch.Tensor]:
            # Drop history-less datasets before the network forward: at sampling time ``x`` may carry
            # a template tensor for a history-less predicted dataset (used above to size its source),
            # but the encoder path must match training, where history-less datasets have no input.
            return self.forward(
                model,
                model._encoder_inputs(x_arg),
                y_arg,
                sigma_arg,
                model_comm_group=comm_arg,
                grid_shard_sizes=shard_sizes_arg,
            )

        return sampler_instance.sample(
            x,
            y_init,
            sigma_schedule,
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
        schedule_params: Optional[dict] = None,
        sampler_params: Optional[dict] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        x_device = next(iter(x.values())).device

        source = model.build_sampling_source(
            x,
            model_comm_group=model_comm_group,
            grid_shard_sizes=grid_shard_sizes,
        )
        y_init = source

        time_schedule = _build_inference_schedule(
            model,
            TIME_SCHEDULES,
            "stochastic-interpolant schedule",
            schedule_params,
            x_device,
        )

        def transport_fn(
            x_arg: dict[str, torch.Tensor],
            y_arg: dict[str, torch.Tensor],
            time_arg: dict[str, torch.Tensor],
            comm_arg: Optional[ProcessGroup] = None,
            shard_sizes_arg: DatasetShardSizes | None = None,
        ) -> dict[str, torch.Tensor]:
            # Drop history-less datasets before the network forward (see EDM objective for rationale).
            return self.forward(
                model,
                model._encoder_inputs(x_arg),
                y_arg,
                time_arg,
                model_comm_group=comm_arg,
                grid_shard_sizes=shard_sizes_arg,
            )

        sampler_instance = _build_inference_sampler(
            model,
            transport_samplers.VECTOR_FIELD_SAMPLERS,
            "stochastic-interpolant sampler",
            sampler_params,
            time_schedule.dtype,
        )
        return sampler_instance.sample(
            x,
            y_init,
            time_schedule,
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

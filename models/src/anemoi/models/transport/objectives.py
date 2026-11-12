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
import warnings
from typing import TYPE_CHECKING
from typing import Any
from typing import Optional

import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.data import Batch
from anemoi.models.samplers import transport_samplers
from anemoi.models.transport.schedules import SIGMA_SCHEDULES
from anemoi.models.transport.schedules import TIME_SCHEDULES

if TYPE_CHECKING:
    from anemoi.models.data.sources import BaseTemplate

LOGGER = logging.getLogger(__name__)


def _multiply(data: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    return data * factor


def _add(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left + right


def _get_inference_defaults_section(model: Any, name: str) -> dict:
    section = model.inference_defaults.get(name)
    return dict(section) if section is not None else {}


def _get_inference_config(model: Any, section: str, overrides: Optional[dict], selector: str) -> dict:
    """Return the settings of the sampler or schedule: the model's defaults, updated with ``overrides``.

    Overrides that choose a different sampler or schedule (``selector``) replace the defaults, which belong to the
    default choice and may not apply to the new one.
    """
    config = _get_inference_defaults_section(model, section)
    if overrides is not None:
        if selector in overrides and overrides[selector] != config.get(selector):
            config = {}
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


def _sampling_dtype(device: torch.device) -> torch.dtype:
    """Return the dtype of the sampling schedule and solver state on ``device``: float64, or float32 on MPS."""
    if device.type != "mps":
        return torch.float64
    warnings.warn(
        "Transport sampling runs in float32 on MPS, which has no float64; predictions differ slightly from "
        "those of the float64 sampler on CPU or CUDA.",
        stacklevel=2,
    )
    return torch.float32


def _build_inference_schedule(
    model: Any,
    registry: dict,
    context: str,
    schedule_params: Optional[dict],
    device: torch.device,
) -> torch.Tensor:
    schedule_config = _get_inference_config(model, "sampling_schedule", schedule_params, "schedule_type")
    scheduler_cls = _resolve_registry_entry(schedule_config, "schedule_type", registry, context)
    scheduler = scheduler_cls(**schedule_config)
    return scheduler.get_schedule(device, _sampling_dtype(device))


def _build_inference_sampler(
    model: Any,
    registry: dict,
    context: str,
    sampler_params: Optional[dict],
    dtype: torch.dtype,
) -> Any:
    sampler_config = _get_inference_config(model, "sampler", sampler_params, "sampler")
    sampler_cls = _resolve_registry_entry(sampler_config, "sampler", registry, context)
    return sampler_cls(dtype=dtype, **sampler_config)


class TransportModelObjective:
    """How a transport model runs its forward pass and inference sampler."""

    def forward(
        self,
        model: Any,
        x: Batch,
        conditioned_target: Batch,
        condition: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        **kwargs: Any,
    ) -> Batch:
        raise NotImplementedError

    def sample(
        self,
        model: Any,
        x: Batch,
        *,
        target_template: dict[str, BaseTemplate],
        model_comm_group: Optional[ProcessGroup] = None,
        schedule_params: Optional[dict] = None,
        sampler_params: Optional[dict] = None,
        target_forcing: Optional[Batch] = None,
    ) -> Batch:
        raise NotImplementedError


class EDMDiffusionModelObjective(TransportModelObjective):
    """EDM diffusion model objective."""

    def forward(
        self,
        model: Any,
        x: Batch,
        y_noised: Batch,
        sigma: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        **kwargs: Any,
    ) -> Batch:
        c_skip, c_out, c_in, c_noise = self._get_preconditioning(model, sigma, model.edm.sigma_data)
        scaled_noised = y_noised.with_sources(
            {name: noised.map_with_condition(_multiply, c_in[name]) for name, noised in y_noised.items()},
        )
        pred = model._forward_transport_network(
            x,
            scaled_noised,
            c_noise,
            model_comm_group=model_comm_group,
            **kwargs,
        )
        return y_noised.with_sources(
            {
                name: noised.map_with_condition(_multiply, c_skip[name]).zip_map_data(
                    _add,
                    pred[name].map_with_condition(_multiply, c_out[name]),
                )
                for name, noised in y_noised.items()
            },
        )

    def sample(
        self,
        model: Any,
        x: Batch,
        *,
        target_template: dict[str, BaseTemplate],
        model_comm_group: Optional[ProcessGroup] = None,
        schedule_params: Optional[dict] = None,
        sampler_params: Optional[dict] = None,
        target_forcing: Optional[Batch] = None,
    ) -> Batch:
        """Sample from an EDM diffusion model."""
        x_device = x.device

        source = model.build_sampling_source(
            x,
            target_template=target_template,
            model_comm_group=model_comm_group,
        )
        sigma_schedule = _build_inference_schedule(
            model,
            SIGMA_SCHEDULES,
            "EDM sampling schedule",
            schedule_params,
            x_device,
        )
        # The source is already described in model-output space, like the sampled target.
        y_batch = source.map_data(lambda data: data.to(dtype=sigma_schedule.dtype) * sigma_schedule[0])

        sampler_instance = _build_inference_sampler(
            model,
            transport_samplers.DIFFUSION_SAMPLERS,
            "EDM sampler",
            sampler_params,
            sigma_schedule.dtype,
        )

        def denoising_fn(
            x_arg: Batch,
            y_arg: Batch,
            sigma_arg: dict[str, torch.Tensor],
            comm_arg: Optional[ProcessGroup] = None,
        ) -> Batch:
            return self.forward(
                model,
                x_arg,
                y_arg,
                sigma_arg,
                model_comm_group=comm_arg,
                target_forcing=target_forcing,
            )

        return sampler_instance.sample(
            x,
            y_batch,
            sigma_schedule,
            denoising_fn,
            model_comm_group,
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


class FlowMatchingModelObjective(TransportModelObjective):
    """Flow-matching model objective that predicts the velocity from the source field to the target."""

    def forward(
        self,
        model: Any,
        x: Batch,
        state: Batch,
        time_level: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        **kwargs: Any,
    ) -> Batch:
        return model._forward_transport_network(
            x,
            state,
            time_level,
            model_comm_group=model_comm_group,
            **kwargs,
        )

    def sample(
        self,
        model: Any,
        x: Batch,
        *,
        target_template: dict[str, BaseTemplate],
        model_comm_group: Optional[ProcessGroup] = None,
        schedule_params: Optional[dict] = None,
        sampler_params: Optional[dict] = None,
        target_forcing: Optional[Batch] = None,
    ) -> Batch:
        x_device = x.device

        # The source is already described in model-output space, like the sampled target.
        y_init = model.build_sampling_source(
            x,
            target_template=target_template,
            model_comm_group=model_comm_group,
        )

        time_schedule = _build_inference_schedule(
            model,
            TIME_SCHEDULES,
            "flow-matching schedule",
            schedule_params,
            x_device,
        )

        def transport_fn(
            x_arg: Batch,
            y_arg: Batch,
            time_arg: dict[str, torch.Tensor],
            comm_arg: Optional[ProcessGroup] = None,
        ) -> Batch:
            return self.forward(
                model,
                x_arg,
                y_arg,
                time_arg,
                model_comm_group=comm_arg,
                target_forcing=target_forcing,
            )

        sampler_instance = _build_inference_sampler(
            model,
            transport_samplers.VECTOR_FIELD_SAMPLERS,
            "flow-matching sampler",
            sampler_params,
            time_schedule.dtype,
        )
        return sampler_instance.sample(
            x,
            y_init,
            time_schedule,
            transport_fn,
            model_comm_group,
        )


TRANSPORT_MODEL_OBJECTIVES = {
    "edm_diffusion": EDMDiffusionModelObjective,
    "flow_matching": FlowMatchingModelObjective,
}


def get_transport_model_objective(name: str) -> TransportModelObjective:
    try:
        return TRANSPORT_MODEL_OBJECTIVES[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown transport model objective: {name}") from exc

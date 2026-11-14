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
from collections.abc import Callable
from collections.abc import Mapping
from typing import Any
from typing import ClassVar
from typing import Optional

import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.data import Batch
from anemoi.models.samplers import transport_samplers
from anemoi.models.transport.registry import build_registered
from anemoi.models.transport.schedules import SIGMA_SCHEDULES
from anemoi.models.transport.schedules import TIME_SCHEDULES
from anemoi.models.transport.settings import EdmSettings

LOGGER = logging.getLogger(__name__)


def _multiply(data: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    return data * factor


def _add(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left + right


TransportNetwork = Callable[[Batch, Batch, dict[str, torch.Tensor]], Batch]
"""The model's encoder, processor and decoder: ``network(x, conditioned_target, condition) -> prediction``."""


def condition_batch_shape(condition: dict[str, torch.Tensor]) -> tuple[int, int]:
    """Check that ``condition`` holds one value per sample and member, and return (batch, ensemble).

    Every dataset's condition has shape (batch, 1, ensemble, 1, 1), with the same batch and ensemble sizes.
    """
    dataset_names = list(condition)
    condition_ref = condition[dataset_names[0]]
    assert condition_ref.ndim == 5, "Expected condition to have shape (batch, 1, ensemble, 1, 1)."
    batch_size, _, ensemble_size = condition_ref.shape[:3]
    for dataset_name in dataset_names:
        condition_shape = condition[dataset_name].shape
        assert len(condition_shape) == 5, (
            f"Expected condition to have shape (batch, 1, ensemble, 1, 1) for '{dataset_name}'."
        )
        assert condition_shape[1] == condition_shape[3] == condition_shape[4] == 1, (
            f"Expected condition to have shape (batch, 1, ensemble, 1, 1) for '{dataset_name}'."
        )
        assert condition_shape[0] == batch_size and condition_shape[2] == ensemble_size, (
            "Batch or ensemble dimension mismatch across datasets for conditioned inputs."
        )
    return batch_size, ensemble_size


def edm_preconditioning(
    sigma: dict[str, torch.Tensor],
    sigma_data: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return the EDM preconditioning factors c_skip, c_out, c_in and c_noise for each dataset.

    Karras et al. (2022), table 1: the network sees its input scaled to unit variance, learns a target of unit
    variance, and is told the noise level as log(sigma) / 4.
    """
    batch_size, ensemble_size = condition_batch_shape(sigma)
    base_view = (batch_size, 1, ensemble_size, 1, 1)

    c_skip, c_out, c_in, c_noise = {}, {}, {}, {}
    for dataset_name, sigma_dataset in sigma.items():
        sigma_base = sigma_dataset[:, 0, :, 0, 0]
        shape_x = sigma_dataset.shape
        c_skip[dataset_name] = (sigma_data**2 / (sigma_base**2 + sigma_data**2)).view(base_view).expand(shape_x)
        c_out[dataset_name] = (
            (sigma_base * sigma_data / (sigma_base**2 + sigma_data**2) ** 0.5).view(base_view).expand(shape_x)
        )
        c_in[dataset_name] = (1.0 / (sigma_data**2 + sigma_base**2) ** 0.5).view(base_view).expand(shape_x)
        c_noise[dataset_name] = (sigma_base.log() / 4.0).view(base_view).expand(shape_x)
    return c_skip, c_out, c_in, c_noise


def _get_inference_config(inference_defaults: Mapping, section: str, overrides: Optional[dict]) -> dict:
    """Return the settings of the sampler or schedule: the model's defaults, updated with ``overrides``."""
    defaults = inference_defaults.get(section)
    config = dict(defaults) if defaults is not None else {}
    if overrides is not None:
        config.update(overrides)
    LOGGER.debug("%s_config: %s", section, config)
    return config


def _build_from_config(config: dict, selector: str, registry: dict, context: str, **fixed: Any) -> Any:
    """Build the registry entry that ``config[selector]`` names, from the other settings in ``config``."""
    config = dict(config)
    if selector not in config:
        raise ValueError(f"{context} must define '{selector}'.")
    return build_registered(registry, config.pop(selector), config, context, **fixed)


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
    inference_defaults: Mapping,
    registry: dict,
    context: str,
    schedule_params: Optional[dict],
    device: torch.device,
) -> torch.Tensor:
    schedule_config = _get_inference_config(inference_defaults, "sampling_schedule", schedule_params)
    scheduler = _build_from_config(schedule_config, "schedule_type", registry, context)
    return scheduler.get_schedule(device, _sampling_dtype(device))


def _build_inference_sampler(
    inference_defaults: Mapping,
    registry: dict,
    context: str,
    sampler_params: Optional[dict],
    dtype: torch.dtype,
) -> Any:
    sampler_config = _get_inference_config(inference_defaults, "sampler", sampler_params)
    return _build_from_config(sampler_config, "sampler", registry, context, dtype=dtype)


class TransportModelObjective:
    """How a transport model runs its network and its inference sampler.

    The model hands the objective its network as a callable, so the objective needs nothing else from the
    model. Subclasses name the schedules and samplers they offer and the state sampling starts from.
    """

    schedules: ClassVar[dict[str, type]]
    samplers: ClassVar[dict[str, type]]
    name: ClassVar[str]

    def forward(
        self,
        network: TransportNetwork,
        x: Batch,
        conditioned_target: Batch,
        condition: dict[str, torch.Tensor],
    ) -> Batch:
        raise NotImplementedError

    def initial_state(self, noise: Batch, schedule: torch.Tensor) -> Batch:
        """Return the state sampling starts from, given unit Gaussian noise and the schedule."""
        del schedule
        return noise

    def sample(
        self,
        network: TransportNetwork,
        x: Batch,
        noise: Batch,
        inference_defaults: Mapping,
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        schedule_params: Optional[dict] = None,
        sampler_params: Optional[dict] = None,
    ) -> Batch:
        """Integrate from Gaussian noise to a sample of the target.

        ``inference_defaults`` holds the model's ``sampling_schedule`` and ``sampler`` settings, which
        ``schedule_params`` and ``sampler_params`` update.
        """
        schedule = _build_inference_schedule(
            inference_defaults,
            self.schedules,
            f"{self.name} sampling schedule",
            schedule_params,
            x.device,
        )
        sampler = _build_inference_sampler(
            inference_defaults,
            self.samplers,
            f"{self.name} sampler",
            sampler_params,
            schedule.dtype,
        )

        def step(
            x_arg: Batch,
            y_arg: Batch,
            condition: dict[str, torch.Tensor],
            comm_arg: Optional[ProcessGroup] = None,
        ) -> Batch:
            # The network is already bound to the model communication group.
            del comm_arg
            return self.forward(network, x_arg, y_arg, condition)

        return sampler.sample(x, self.initial_state(noise, schedule), schedule, step, model_comm_group)


class EDMDiffusionModelObjective(TransportModelObjective):
    """EDM diffusion: the network, wrapped in the EDM preconditioning, predicts the clean target."""

    schedules = SIGMA_SCHEDULES
    samplers = transport_samplers.DIFFUSION_SAMPLERS
    name = "EDM"

    def __init__(self, sigma_data: float = 1.0) -> None:
        self.sigma_data = float(sigma_data)

    def forward(
        self,
        network: TransportNetwork,
        x: Batch,
        y_noised: Batch,
        sigma: dict[str, torch.Tensor],
    ) -> Batch:
        c_skip, c_out, c_in, c_noise = edm_preconditioning(sigma, self.sigma_data)
        scaled_noised = y_noised.with_sources(
            {name: noised.map_with_condition(_multiply, c_in[name]) for name, noised in y_noised.items()},
        )
        pred = network(x, scaled_noised, c_noise)
        return y_noised.with_sources(
            {
                name: noised.map_with_condition(_multiply, c_skip[name]).zip_map_data(
                    _add,
                    pred[name].map_with_condition(_multiply, c_out[name]),
                )
                for name, noised in y_noised.items()
            },
        )

    def initial_state(self, noise: Batch, schedule: torch.Tensor) -> Batch:
        """Scale unit noise to the largest noise level, in the schedule's precision."""
        return noise.map_data(lambda data: data.to(dtype=schedule.dtype) * schedule[0])


class FlowMatchingModelObjective(TransportModelObjective):
    """Flow matching: the network predicts the velocity from Gaussian noise to the target."""

    schedules = TIME_SCHEDULES
    samplers = transport_samplers.VECTOR_FIELD_SAMPLERS
    name = "flow-matching"

    def forward(
        self,
        network: TransportNetwork,
        x: Batch,
        state: Batch,
        time_level: dict[str, torch.Tensor],
    ) -> Batch:
        return network(x, state, time_level)


def get_transport_model_objective(name: str, edm: EdmSettings) -> TransportModelObjective:
    """Return the model-side objective ``name``, set up with the model's EDM settings where it uses them."""
    if name == "edm_diffusion":
        return EDMDiffusionModelObjective(sigma_data=edm.sigma_data)
    if name == "flow_matching":
        return FlowMatchingModelObjective()
    msg = f"Unknown transport model objective: {name!r}. Available: edm_diffusion, flow_matching."
    raise ValueError(msg)

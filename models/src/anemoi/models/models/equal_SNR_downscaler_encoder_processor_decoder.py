# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import time
import warnings
from typing import Callable
from typing import Optional
from typing import Union

import einops
import torch
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.models.downscaler_encoder_processor_decoder import AnemoiDownscalingModelEncProcDec
from anemoi.models.samplers import diffusion_samplers
from anemoi.utils.config import DotDict
from anemoi.utils.spectral import DCT2D, InverseDCT2D

LOGGER = logging.getLogger(__name__)


class AnemoiEqualSNRDownscalingModelEncProcDec(AnemoiDownscalingModelEncProcDec):
    """Downscaling Model."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
        truncation_data: dict,
    ) -> None:
        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
            truncation_data=truncation_data,
        )

    def sample(
        self,
        x_in_interp_to_hres: torch.Tensor,
        x_in_hres: torch.Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        noise_scheduler_params: Optional[dict] = None,
        sampler_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from the diffusion model.

        Parameters
        ----------
        x : torch.Tensor
            Input conditioning data with shape (batch, time, ensemble, grid, vars)
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training
        grid_shard_shapes : Optional[list]
            Grid shard shapes for distributed processing
        noise_scheduler_params : Optional[dict]
            Dictionary of noise scheduler parameters (schedule_type, num_steps, sigma_max, etc.) to override defaults
        sampler_params : Optional[dict]
            Dictionary of sampler parameters (sampler, S_churn, S_min, etc.) to override defaults
        **kwargs
            Additional sampler-specific arguments

        Returns
        -------
        torch.Tensor
            Sampled output with shape (batch, ensemble, grid, vars)
        """

        # Start with inference defaults
        if hasattr(self.inference_defaults, "noise_scheduler"):
            noise_scheduler_config = dict(self.inference_defaults.noise_scheduler)
        else:

            if self.training_approach == "probabilistic_high_noise":
                default_scheduler = self.DEFAULT_HIGH_NOISE_SCHEDULER_PARAMS
            else:
                default_scheduler = self.DEFAULT_LOW_NOISE_SCHEDULER_PARAMS
            noise_scheduler_config = dict(default_scheduler)
        print(f"default {noise_scheduler_config=}")

        # Override config with provided noise scheduler parameters
        if noise_scheduler_params is not None:
            noise_scheduler_config.update(noise_scheduler_params)

        warnings.warn(f"noise_scheduler_config: {noise_scheduler_config}")
        print(f"noise_scheduler_config: {noise_scheduler_config}")

        # Remove schedule_type (used for class selection, not constructor)
        actual_schedule_type = noise_scheduler_config.pop("schedule_type")

        if actual_schedule_type not in diffusion_samplers.NOISE_SCHEDULERS:
            raise ValueError(f"Unknown schedule type: {actual_schedule_type}")

        scheduler_cls = diffusion_samplers.NOISE_SCHEDULERS[actual_schedule_type]
        scheduler = scheduler_cls(**noise_scheduler_config)
        sigmas = scheduler.get_schedule(x_in_interp_to_hres.device, torch.float64)

        # Initialize output with noise
        batch_size, ensemble_size, grid_size = (
            x_in_interp_to_hres.shape[0],
            x_in_interp_to_hres.shape[2],
            x_in_interp_to_hres.shape[-2],
        )
        time_size = 1
        shape = (
            batch_size,
            time_size,
            ensemble_size,
            grid_size,
            self.num_output_channels,
        )

        itransform = kwargs["itransform"]
        variance = kwargs["variance"]

        y_init = sigmas[0] * itransform(torch.randn(shape, device=variance.device, dtype=sigmas.dtype) * torch.pow(variance, 1/2))
        #y_init = sigmas[0] * itransform(torch.randn(shape, device=variance.device, dtype=sigmas.dtype) * variance)

        print("sigmas", sigmas)

        # Build diffusion sampler config dict from all inference defaults
        if hasattr(self.inference_defaults, "diffusion_sampler"):
            diffusion_sampler_config = dict(self.inference_defaults.diffusion_sampler)
        else:

            if self.training_approach == "probabilistic_high_noise":
                default_sampler = self.DEFAULT_HIGH_NOISE_SAMPLER_PARAMS
            else:
                default_sampler = self.DEFAULT_LOW_NOISE_SAMPLER_PARAMS
            diffusion_sampler_config = dict(default_sampler)
        print(f"default {diffusion_sampler_config=}")

        # Override config with provided sampler parameters
        if sampler_params is not None:
            diffusion_sampler_config.update(sampler_params)

        warnings.warn(f"diffusion_sampler_config: {diffusion_sampler_config}")
        print(f"diffusion_sampler_config: {diffusion_sampler_config}")

        # Remove sampler name (used for class selection, not constructor)
        actual_sampler = diffusion_sampler_config.pop("sampler")

        if actual_sampler not in diffusion_samplers.DIFFUSION_SAMPLERS:
            raise ValueError(f"Unknown sampler: {actual_sampler}")

        sampler_cls = diffusion_samplers.DIFFUSION_SAMPLERS[actual_sampler]

        diffusion_sampler_config["itransform"]=itransform
        diffusion_sampler_config["variance"]=variance
        sampler_instance = sampler_cls(dtype=sigmas.dtype, **diffusion_sampler_config)

        return sampler_instance.sample(
            x_in_interp_to_hres,
            x_in_hres,
            y_init,
            sigmas,
            self.fwd_with_preconditioning,
            grid_shard_shapes=grid_shard_shapes,
            model_comm_group=model_comm_group
        )

    

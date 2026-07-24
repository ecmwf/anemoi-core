# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from abc import ABC
from abc import abstractmethod
from typing import Callable
from typing import Optional

import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.data import Batch
from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.transport.data_helpers import Data
from anemoi.models.transport.data_helpers import add_data
from anemoi.models.transport.data_helpers import condition_shape
from anemoi.models.transport.data_helpers import data_dtype
from anemoi.models.transport.data_helpers import map_data
from anemoi.models.transport.data_helpers import randn_like_data
from anemoi.models.transport.data_helpers import scale_data
from anemoi.models.transport.data_helpers import zip_map_data

TransportModelFunction = Callable[
    [
        Batch,
        Batch,
        dict[str, torch.Tensor],
        Optional[ProcessGroup],
        DatasetShardSizes | None,
    ],
    Batch,
]
DenoisingFunction = TransportModelFunction
VectorFieldFunction = TransportModelFunction


def _map_data_dict(data: dict[str, Data], fn: Callable[[torch.Tensor], torch.Tensor]) -> dict[str, Data]:
    return {dataset_name: map_data(dataset_data, fn) for dataset_name, dataset_data in data.items()}


def _expand_scalar_condition(value: torch.Tensor, y: Batch) -> dict[str, torch.Tensor]:
    """Expand one scalar condition so each dataset can pass it to the model."""
    condition = {}
    for dataset_name, y_data in y.data.items():
        shape = condition_shape(y_data) if isinstance(y_data, list) else (y_data.shape[0], 1, y_data.shape[2], 1, 1)
        condition[dataset_name] = value.view(1, 1, 1, 1, 1).expand(shape).to(data_dtype(y_data))
    return condition


class EDMDiffusionSampler(ABC):
    """Base class for EDM diffusion samplers."""

    @abstractmethod
    def sample(
        self,
        x: Batch,
        y: Batch,
        sigmas: torch.Tensor,
        denoising_fn: DenoisingFunction,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        **kwargs,
    ) -> Batch:
        """Run EDM diffusion sampling from the initial noisy field to a clean prediction.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input conditioning data with shape (batch, time, ensemble, grid, vars).
        y : dict[str, torch.Tensor]
            Initial noise tensor with shape (batch, time, ensemble, grid, vars).
        sigmas : torch.Tensor
            Noise schedule with shape (num_steps + 1,). The final value is
            expected to be exact zero after sigma schedule finalization.
        denoising_fn : Callable
            Function that performs denoising.
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training.
        grid_shard_sizes : DatasetShardSizes, optional
            Per-dataset shard sizes for the grid dimension. ``None`` means the
            corresponding dataset is replicated, not sharded.
        **kwargs
            Additional sampler-specific parameters.

        Returns
        -------
        dict[str, torch.Tensor]
            Sampled output with shape (batch, time, ensemble, grid, vars).
        """
        pass


class EDMHeunSampler(EDMDiffusionSampler):
    """EDM Heun sampler with stochastic churn following Karras et al."""

    def __init__(
        self,
        S_churn: float = 0.0,
        S_min: float = 0.0,
        S_max: float = float("inf"),
        S_noise: float = 1.0,
        dtype: torch.dtype = torch.float64,
        eps_prec: float = 1e-10,
    ):
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.dtype = dtype
        self.eps_prec = eps_prec

    def sample(
        self,
        x: Batch,
        y: Batch,
        sigmas: torch.Tensor,
        denoising_fn: DenoisingFunction,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        **kwargs,
    ) -> Batch:
        # Override instance defaults with any kwargs
        S_churn = kwargs.get("S_churn", self.S_churn)
        S_min = kwargs.get("S_min", self.S_min)
        S_max = kwargs.get("S_max", self.S_max)
        S_noise = kwargs.get("S_noise", self.S_noise)
        dtype = kwargs.get("dtype", self.dtype)
        eps_prec = kwargs.get("eps_prec", self.eps_prec)
        sigmas = sigmas.to(dtype)

        num_steps = len(sigmas) - 1
        # Persistent dtype-precision solver state; all Heun update arithmetic uses this buffer.
        y_solver = _map_data_dict(y.data, lambda sample: sample.to(dtype))

        # Heun sampling loop
        for i in range(num_steps):
            sigma_i = sigmas[i]
            sigma_next = sigmas[i + 1]

            apply_churn = S_min <= sigma_i <= S_max and S_churn > 0.0
            if apply_churn:
                gamma = min(
                    S_churn / num_steps,
                    torch.sqrt(torch.tensor(2.0, dtype=sigma_i.dtype)) - 1,
                )
                sigma_effective = sigma_i + gamma * sigma_i

                for dataset_name, y_solver_data in y_solver.items():
                    dataset_grid_shard_sizes = (
                        grid_shard_sizes.get(dataset_name) if grid_shard_sizes is not None else None
                    )
                    epsilon = scale_data(
                        randn_like_data(
                            y_solver_data,
                            model_comm_group=model_comm_group,
                            grid_shard_sizes=dataset_grid_shard_sizes,
                        ),
                        S_noise,
                    )
                    y_solver[dataset_name] = add_data(
                        y_solver_data,
                        scale_data(epsilon, torch.sqrt(sigma_effective**2 - sigma_i**2)),
                    )
            else:
                sigma_effective = sigma_i

            # Cast for model evaluation: run denoiser in model/input dtype.
            y_model = y.with_data(
                {
                    dataset_name: map_data(
                        y_data, lambda sample, name=dataset_name: sample.to(data_dtype(x.data[name]))
                    )
                    for dataset_name, y_data in y_solver.items()
                },
            )

            sigma_effective_expanded = _expand_scalar_condition(sigma_effective, y_model)

            D1 = denoising_fn(
                x,
                y_model,
                sigma_effective_expanded,
                model_comm_group,
                grid_shard_sizes,
            )
            D1_solver = _map_data_dict(D1.data, lambda sample: sample.to(dtype))

            # Predictor state in solver precision; for Heun corrector evaluation.
            update_direction, y_next_solver = {}, {}
            for dataset_name in y_solver:
                update_direction[dataset_name] = zip_map_data(
                    y_solver[dataset_name],
                    D1_solver[dataset_name],
                    lambda y_sample, denoised_sample: (y_sample - denoised_sample) / (sigma_effective + eps_prec),
                )
                y_next_solver[dataset_name] = add_data(
                    y_solver[dataset_name],
                    scale_data(update_direction[dataset_name], sigma_next - sigma_effective),
                )

            if sigma_next != 0:
                y_next_model = y.with_data(
                    {
                        dataset_name: map_data(
                            y_next_data,
                            lambda sample, name=dataset_name: sample.to(data_dtype(x.data[name])),
                        )
                        for dataset_name, y_next_data in y_next_solver.items()
                    },
                )
                sigma_next_expanded = _expand_scalar_condition(sigma_next, y_next_model)

                D2 = denoising_fn(
                    x,
                    y_next_model,
                    sigma_next_expanded,
                    model_comm_group,
                    grid_shard_sizes,
                )
                D2_solver = _map_data_dict(D2.data, lambda sample: sample.to(dtype))

                for dataset_name in y_solver:
                    corrected_update_direction = zip_map_data(
                        y_next_solver[dataset_name],
                        D2_solver[dataset_name],
                        lambda y_sample, denoised_sample: (y_sample - denoised_sample) / (sigma_next + eps_prec),
                    )
                    combined_direction = zip_map_data(
                        update_direction[dataset_name],
                        corrected_update_direction,
                        lambda update_sample, corrected_sample: (update_sample + corrected_sample) / 2,
                    )
                    y_solver[dataset_name] = add_data(
                        y_solver[dataset_name],
                        scale_data(combined_direction, sigma_next - sigma_effective),
                    )
            else:
                y_solver = y_next_solver

        return y.with_data(
            {
                dataset_name: map_data(y_data, lambda sample, name=dataset_name: sample.to(data_dtype(x.data[name])))
                for dataset_name, y_data in y_solver.items()
            },
        )


class DPMpp2MSampler(EDMDiffusionSampler):
    """DPM++ 2M sampler (DPM-Solver++ with 2nd order multistep)."""

    def __init__(
        self,
        dtype: torch.dtype = torch.float64,
    ):
        self.dtype = dtype
        pass  # No parameters needed for DPM++ 2M

    def sample(
        self,
        x: Batch,
        y: Batch,
        sigmas: torch.Tensor,
        denoising_fn: DenoisingFunction,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        **kwargs,
    ) -> Batch:
        dtype = kwargs.get("dtype", self.dtype)

        # Keep model evaluations in model dtype, but run solver updates in sampler dtype.
        y_model = y.with_data(
            {
                dataset_name: map_data(y_data, lambda sample, name=dataset_name: sample.to(data_dtype(x.data[name])))
                for dataset_name, y_data in y.data.items()
            },
        )
        sigmas = sigmas.to(dtype)

        num_steps = len(sigmas) - 1

        # Storage for previous denoised predictions
        old_denoised = None

        # DPM++ 2M sampling loop
        for i in range(num_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            sigma_expanded = _expand_scalar_condition(sigma, y_model)
            denoised = denoising_fn(x, y_model, sigma_expanded, model_comm_group, grid_shard_sizes)
            denoised_solver = _map_data_dict(denoised.data, lambda sample: sample.to(dtype))

            if sigma_next == 0:
                y_model = y.with_data(
                    {
                        dataset_name: map_data(
                            den,
                            lambda sample, name=dataset_name: sample.to(data_dtype(x.data[name])),
                        )
                        for dataset_name, den in denoised_solver.items()
                    },
                )
                break

            y_solver = _map_data_dict(y_model.data, lambda sample: sample.to(dtype))
            t = -torch.log(sigma + 1e-10)
            t_next = -torch.log(sigma_next + 1e-10) if sigma_next != 0 else float("inf")
            h = t_next - t

            if old_denoised is None:
                for dataset_name in y.data:
                    y_solver[dataset_name] = zip_map_data(
                        y_solver[dataset_name],
                        denoised_solver[dataset_name],
                        lambda y_sample, denoised_sample: (sigma_next / sigma) * y_sample
                        - (torch.exp(-h) - 1) * denoised_sample,
                    )
            else:
                # Second order multistep
                h_last = t - (-torch.log(sigmas[i - 1] + 1e-10)) if i > 0 else h
                r = h_last / h

                coeff1 = 1 + 1 / (2 * r)
                coeff2 = -1 / (2 * r)

                for dataset_name in y.data:
                    direction = zip_map_data(
                        denoised_solver[dataset_name],
                        old_denoised[dataset_name],
                        lambda denoised_sample, old_sample: coeff1 * denoised_sample + coeff2 * old_sample,
                    )
                    y_solver[dataset_name] = zip_map_data(
                        y_solver[dataset_name],
                        direction,
                        lambda y_sample, direction_sample: (sigma_next / sigma) * y_sample
                        - (torch.exp(-h) - 1) * direction_sample,
                    )

            old_denoised = denoised_solver
            y_model = y.with_data(
                {
                    dataset_name: map_data(
                        y_data, lambda sample, name=dataset_name: sample.to(data_dtype(x.data[name]))
                    )
                    for dataset_name, y_data in y_solver.items()
                },
            )

        return y_model


DIFFUSION_SAMPLERS = {
    "heun": EDMHeunSampler,
    "dpmpp_2m": DPMpp2MSampler,
}


class VectorFieldSampler(ABC):
    """Base class for ODE samplers that integrate a learned vector field."""

    @abstractmethod
    def sample(
        self,
        x: Batch,
        y: Batch,
        times: torch.Tensor,
        vector_field_fn: VectorFieldFunction,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        **kwargs,
    ) -> Batch:
        """Move the field along the provided time grid."""
        pass


class VectorFieldEulerSampler(VectorFieldSampler):
    """First-order Euler sampler for learned ODE vector fields."""

    def __init__(
        self,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.dtype = dtype

    def sample(
        self,
        x: Batch,
        y: Batch,
        times: torch.Tensor,
        vector_field_fn: VectorFieldFunction = None,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        **kwargs,
    ) -> Batch:
        if vector_field_fn is None:
            raise ValueError("VectorFieldEulerSampler requires a vector_field_fn callable.")
        dtype = kwargs.get("dtype", self.dtype)
        times = times.to(dtype)
        y_solver = _map_data_dict(y.data, lambda sample: sample.to(dtype))

        for i in range(len(times) - 1):
            time_i = times[i]
            time_next = times[i + 1]
            dt = time_next - time_i

            y_model = y.with_data(
                {
                    dataset_name: map_data(
                        y_data, lambda sample, name=dataset_name: sample.to(data_dtype(x.data[name]))
                    )
                    for dataset_name, y_data in y_solver.items()
                },
            )
            time_expanded = _expand_scalar_condition(time_i, y_model)
            vector_field = vector_field_fn(
                x,
                y_model,
                time_expanded,
                model_comm_group,
                grid_shard_sizes,
            )

            for dataset_name in y_solver:
                y_solver[dataset_name] = add_data(
                    y_solver[dataset_name],
                    scale_data(map_data(vector_field.data[dataset_name], lambda sample: sample.to(dtype)), dt),
                )

        return y.with_data(
            {
                dataset_name: map_data(y_data, lambda sample, name=dataset_name: sample.to(data_dtype(x.data[name])))
                for dataset_name, y_data in y_solver.items()
            },
        )


class VectorFieldHeunSampler(VectorFieldSampler):
    """Second-order Heun sampler for deterministic bridge models."""

    def __init__(
        self,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.dtype = dtype

    def sample(
        self,
        x: Batch,
        y: Batch,
        times: torch.Tensor,
        vector_field_fn: VectorFieldFunction = None,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        **kwargs,
    ) -> Batch:
        if vector_field_fn is None:
            raise ValueError("VectorFieldHeunSampler requires a vector_field_fn callable.")
        dtype = kwargs.get("dtype", self.dtype)
        times = times.to(dtype)
        y_solver = _map_data_dict(y.data, lambda sample: sample.to(dtype))

        for i in range(len(times) - 1):
            time_i = times[i]
            time_next = times[i + 1]
            dt = time_next - time_i

            y_model = y.with_data(
                {
                    dataset_name: map_data(
                        y_data, lambda sample, name=dataset_name: sample.to(data_dtype(x.data[name]))
                    )
                    for dataset_name, y_data in y_solver.items()
                },
            )
            time_i_expanded = _expand_scalar_condition(time_i, y_model)
            vector_field_1 = vector_field_fn(
                x,
                y_model,
                time_i_expanded,
                model_comm_group,
                grid_shard_sizes,
            )

            y_predictor = {
                dataset_name: add_data(
                    y_solver[dataset_name],
                    scale_data(map_data(vector_field_1.data[dataset_name], lambda sample: sample.to(dtype)), dt),
                )
                for dataset_name in y_solver
            }
            y_next_model = y.with_data(
                {
                    dataset_name: map_data(
                        y_data,
                        lambda sample, name=dataset_name: sample.to(data_dtype(x.data[name])),
                    )
                    for dataset_name, y_data in y_predictor.items()
                },
            )
            time_next_expanded = _expand_scalar_condition(time_next, y_next_model)
            vector_field_2 = vector_field_fn(
                x,
                y_next_model,
                time_next_expanded,
                model_comm_group,
                grid_shard_sizes,
            )

            for dataset_name in y_solver:
                combined_field = zip_map_data(
                    map_data(vector_field_1.data[dataset_name], lambda sample: sample.to(dtype)),
                    map_data(vector_field_2.data[dataset_name], lambda sample: sample.to(dtype)),
                    lambda first_sample, second_sample: (first_sample + second_sample) / 2,
                )
                y_solver[dataset_name] = add_data(
                    y_solver[dataset_name],
                    scale_data(combined_field, dt),
                )

        return y.with_data(
            {
                dataset_name: map_data(y_data, lambda sample, name=dataset_name: sample.to(data_dtype(x.data[name])))
                for dataset_name, y_data in y_solver.items()
            },
        )


VECTOR_FIELD_SAMPLERS = {
    "euler": VectorFieldEulerSampler,
    "heun": VectorFieldHeunSampler,
}

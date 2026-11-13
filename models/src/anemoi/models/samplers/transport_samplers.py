# (C) Copyright 2025-2026 Anemoi contributors.
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

TransportModelFunction = Callable[
    [
        Batch,
        Batch,
        dict[str, torch.Tensor],
        Optional[ProcessGroup],
    ],
    Batch,
]
DenoisingFunction = TransportModelFunction
VectorFieldFunction = TransportModelFunction


def _in_dtype(batch: Batch, dtype: torch.dtype) -> Batch:
    """Return ``batch`` with every payload cast to ``dtype`` (solver precision)."""
    return batch.map_data(lambda data: data.to(dtype))


def _in_model_dtype(batch: Batch, x: Batch) -> Batch:
    """Return ``batch`` with each dataset's payload cast to the dtype of the matching input."""
    return batch.with_sources(
        {name: source.map_data(lambda data, dtype=x[name].dtype: data.to(dtype)) for name, source in batch.items()},
    )


def _axpy(y: Batch, direction: Batch, step: torch.Tensor | float) -> Batch:
    """Return ``y + step * direction``, dataset by dataset."""
    return y.zip_map_data(lambda y_data, direction_data: y_data + direction_data * step, direction)


def _expand_scalar_condition(value: torch.Tensor, y: Batch) -> dict[str, torch.Tensor]:
    """Expand one scalar condition so each dataset can pass it to the model.

    The condition keeps at least float32 precision, also when the data is in a lower precision:
    the model embeds it at the frequencies of its noise or time embedding.
    """
    return {
        dataset_name: value.view(1, 1, 1, 1, 1)
        .expand(source.condition_shape)
        .to(torch.promote_types(source.dtype, torch.float32))
        for dataset_name, source in y.items()
    }


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
    ):
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.dtype = dtype

    def sample(
        self,
        x: Batch,
        y: Batch,
        sigmas: torch.Tensor,
        denoising_fn: DenoisingFunction,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> Batch:
        S_churn, S_min, S_max, S_noise, dtype = self.S_churn, self.S_min, self.S_max, self.S_noise, self.dtype
        sigmas = sigmas.to(dtype)

        num_steps = len(sigmas) - 1
        # Persistent dtype-precision solver state; all Heun update arithmetic uses this buffer.
        y_solver = _in_dtype(y, dtype)

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

                # Noise is drawn consistently across the grid shards each source records.
                epsilon = y_solver.with_sources(
                    {name: source.randn_like(model_comm_group) for name, source in y_solver.items()},
                ).map_data(lambda noise: noise * S_noise)
                y_solver = _axpy(y_solver, epsilon, torch.sqrt(sigma_effective**2 - sigma_i**2))
                del epsilon
            else:
                sigma_effective = sigma_i

            # The denoiser runs in the model dtype; the update arithmetic stays in the solver dtype.
            # Each field is dropped as soon as it has been used, so at most the state, the first update
            # direction and the predictor state are held in solver precision while the network runs.
            y_model = _in_model_dtype(y_solver, x)
            denoised = denoising_fn(x, y_model, _expand_scalar_condition(sigma_effective, y_model), model_comm_group)
            del y_model
            update_direction = y_solver.zip_map_data(
                lambda y_sample, denoised_sample: (y_sample - denoised_sample) / sigma_effective,
                _in_dtype(denoised, dtype),
            )
            del denoised
            y_next_solver = _axpy(y_solver, update_direction, sigma_next - sigma_effective)

            if sigma_next != 0:
                y_next_model = _in_model_dtype(y_next_solver, x)
                denoised = denoising_fn(
                    x,
                    y_next_model,
                    _expand_scalar_condition(sigma_next, y_next_model),
                    model_comm_group,
                )
                del y_next_model
                corrected_update_direction = y_next_solver.zip_map_data(
                    lambda y_sample, denoised_sample: (y_sample - denoised_sample) / sigma_next,
                    _in_dtype(denoised, dtype),
                )
                del denoised, y_next_solver
                combined_direction = update_direction.zip_map_data(
                    lambda update_sample, corrected_sample: (update_sample + corrected_sample) / 2,
                    corrected_update_direction,
                )
                del update_direction, corrected_update_direction
                y_solver = _axpy(y_solver, combined_direction, sigma_next - sigma_effective)
                del combined_direction
            else:
                y_solver = y_next_solver

        return _in_model_dtype(y_solver, x)


class DPMpp2MSampler(EDMDiffusionSampler):
    """DPM++ 2M sampler (DPM-Solver++ with 2nd order multistep)."""

    def __init__(
        self,
        dtype: torch.dtype = torch.float64,
    ):
        self.dtype = dtype

    def sample(
        self,
        x: Batch,
        y: Batch,
        sigmas: torch.Tensor,
        denoising_fn: DenoisingFunction,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> Batch:
        dtype = self.dtype

        # The state stays in the solver dtype for the whole run; only the copy handed to the denoiser is cast
        # to the model dtype, so the multistep update does not lose precision between steps.
        y_solver = _in_dtype(y, dtype)
        sigmas = sigmas.to(dtype)

        num_steps = len(sigmas) - 1

        # Storage for previous denoised predictions
        old_denoised = None

        # DPM++ 2M sampling loop
        for i in range(num_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            y_model = _in_model_dtype(y_solver, x)
            denoised = denoising_fn(x, y_model, _expand_scalar_condition(sigma, y_model), model_comm_group)
            del y_model
            denoised_solver = _in_dtype(denoised, dtype)
            del denoised

            if sigma_next == 0:
                # The final state is the denoised field, described like the sampled target.
                return _in_model_dtype(y_solver.with_data({n: s.data for n, s in denoised_solver.items()}), x)

            t = -torch.log(sigma)
            t_next = -torch.log(sigma_next)
            h = t_next - t

            if old_denoised is None:
                y_solver = y_solver.zip_map_data(
                    lambda y_sample, denoised_sample: (
                        (sigma_next / sigma) * y_sample - (torch.exp(-h) - 1) * denoised_sample
                    ),
                    denoised_solver,
                )
            else:
                # Second order multistep
                h_last = t + torch.log(sigmas[i - 1])
                r = h_last / h

                coeff1 = 1 + 1 / (2 * r)
                coeff2 = -1 / (2 * r)

                direction = denoised_solver.zip_map_data(
                    lambda denoised_sample, old_sample: coeff1 * denoised_sample + coeff2 * old_sample,
                    old_denoised,
                )
                y_solver = y_solver.zip_map_data(
                    lambda y_sample, direction_sample: (
                        (sigma_next / sigma) * y_sample - (torch.exp(-h) - 1) * direction_sample
                    ),
                    direction,
                )

            old_denoised = denoised_solver

        return _in_model_dtype(y_solver, x)


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
        vector_field_fn: VectorFieldFunction,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> Batch:
        dtype = self.dtype
        times = times.to(dtype)
        y_solver = _in_dtype(y, dtype)

        for i in range(len(times) - 1):
            time_i = times[i]
            time_next = times[i + 1]
            dt = time_next - time_i

            y_model = _in_model_dtype(y_solver, x)
            time_expanded = _expand_scalar_condition(time_i, y_model)
            vector_field = vector_field_fn(
                x,
                y_model,
                time_expanded,
                model_comm_group,
            )

            y_solver = _axpy(y_solver, _in_dtype(vector_field, dtype), dt)

        return _in_model_dtype(y_solver, x)


class VectorFieldHeunSampler(VectorFieldSampler):
    """Second-order Heun sampler for learned ODE vector fields."""

    def __init__(
        self,
        dtype: torch.dtype = torch.float64,
        euler_final_step: bool = True,
    ) -> None:
        self.dtype = dtype
        self.euler_final_step = euler_final_step

    def sample(
        self,
        x: Batch,
        y: Batch,
        times: torch.Tensor,
        vector_field_fn: VectorFieldFunction,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> Batch:
        dtype = self.dtype
        times = times.to(dtype)
        y_solver = _in_dtype(y, dtype)

        num_steps = len(times) - 1
        for i in range(num_steps):
            time_i = times[i]
            time_next = times[i + 1]
            dt = time_next - time_i

            y_model = _in_model_dtype(y_solver, x)
            time_i_expanded = _expand_scalar_condition(time_i, y_model)
            vector_field_1 = vector_field_fn(
                x,
                y_model,
                time_i_expanded,
                model_comm_group,
            )

            vector_field_1_solver = _in_dtype(vector_field_1, dtype)
            y_predictor = _axpy(y_solver, vector_field_1_solver, dt)
            if self.euler_final_step and i == num_steps - 1:
                y_solver = y_predictor
                continue

            y_next_model = _in_model_dtype(y_predictor, x)
            time_next_expanded = _expand_scalar_condition(time_next, y_next_model)
            vector_field_2 = vector_field_fn(
                x,
                y_next_model,
                time_next_expanded,
                model_comm_group,
            )

            combined_field = vector_field_1_solver.zip_map_data(
                lambda first_sample, second_sample: (first_sample + second_sample) / 2,
                _in_dtype(vector_field_2, dtype),
            )
            y_solver = _axpy(y_solver, combined_field, dt)

        return _in_model_dtype(y_solver, x)


VECTOR_FIELD_SAMPLERS = {
    "euler": VectorFieldEulerSampler,
    "heun": VectorFieldHeunSampler,
}

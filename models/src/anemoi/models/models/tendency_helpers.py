# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import Callable
from typing import Optional

import torch
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes


class TendencyPredictHelperMixin:
    """Helper-only mixin for shared tendency prediction utilities."""

    @staticmethod
    def _tendency_apply_imputer_inverse(
        post_processors: dict[str, nn.Module],
        dataset_name: str,
        x: torch.Tensor,
    ) -> torch.Tensor:
        processors = post_processors[dataset_name]
        if not hasattr(processors, "processors"):
            return x
        for processor in processors.processors.values():
            if getattr(processor, "supports_skip_imputation", False):
                x = processor(x, in_place=False, inverse=True, skip_imputation=False)
        return x

    def _tendency_compute_tendency(
        self,
        x_t1: dict[str, torch.Tensor],
        x_t0: dict[str, torch.Tensor],
        pre_processors_state: dict[str, Callable],
        pre_processors_tendencies: dict[str, Callable],
        input_post_processor: Optional[dict[str, Callable]] = None,
        skip_imputation: bool = False,
    ) -> dict[str, torch.Tensor]:
        tendencies = {}

        assert set(x_t1.keys()) == set(x_t0.keys()), "x_t1 and x_t0 must have the same dataset keys."

        for dataset_name in x_t1.keys():
            if input_post_processor is not None and input_post_processor[dataset_name] is not None:
                x_t1[dataset_name] = input_post_processor[dataset_name](
                    x_t1[dataset_name],
                    in_place=False,
                    data_index=self.data_indices[dataset_name].data.output.full,
                    skip_imputation=skip_imputation,
                )
                x_t0[dataset_name] = input_post_processor[dataset_name](
                    x_t0[dataset_name],
                    in_place=False,
                    data_index=self.data_indices[dataset_name].data.input.prognostic,
                    skip_imputation=skip_imputation,
                )

            tendency = x_t1[dataset_name].clone()
            tendency[..., self.data_indices[dataset_name].model.output.prognostic] = pre_processors_tendencies[
                dataset_name
            ](
                x_t1[dataset_name][..., self.data_indices[dataset_name].model.output.prognostic] - x_t0[dataset_name],
                in_place=False,
                data_index=self.data_indices[dataset_name].data.output.prognostic,
                skip_imputation=skip_imputation,
            )
            tendency[..., self.data_indices[dataset_name].model.output.diagnostic] = pre_processors_state[dataset_name](
                x_t1[dataset_name][..., self.data_indices[dataset_name].model.output.diagnostic],
                in_place=False,
                data_index=self.data_indices[dataset_name].data.output.diagnostic,
                skip_imputation=skip_imputation,
            )
            tendencies[dataset_name] = tendency

        return tendencies

    def _tendency_add_tendency_to_state(
        self,
        state_inp: dict[str, torch.Tensor],
        tendency: dict[str, torch.Tensor],
        post_processors_state: dict[str, Callable],
        post_processors_tendencies: dict[str, Callable],
        output_pre_processor: Optional[dict[str, Callable]] = None,
        skip_imputation: bool = False,
    ) -> dict[str, torch.Tensor]:
        state_outp = {}

        for dataset_name in tendency.keys():
            state_outp[dataset_name] = post_processors_tendencies[dataset_name](
                tendency[dataset_name],
                in_place=False,
                data_index=self.data_indices[dataset_name].data.output.full,
                skip_imputation=skip_imputation,
            )

            state_outp[dataset_name][
                ..., self.data_indices[dataset_name].model.output.diagnostic
            ] = post_processors_state[dataset_name](
                tendency[dataset_name][..., self.data_indices[dataset_name].model.output.diagnostic],
                in_place=False,
                data_index=self.data_indices[dataset_name].data.output.diagnostic,
                skip_imputation=skip_imputation,
            )

            state_outp[dataset_name][
                ..., self.data_indices[dataset_name].model.output.prognostic
            ] += post_processors_state[dataset_name](
                state_inp[dataset_name],
                in_place=False,
                data_index=self.data_indices[dataset_name].data.input.prognostic,
                skip_imputation=skip_imputation,
            )

            if output_pre_processor is not None and output_pre_processor[dataset_name] is not None:
                state_outp[dataset_name] = output_pre_processor[dataset_name](
                    state_outp[dataset_name],
                    in_place=False,
                    data_index=self.data_indices[dataset_name].data.output.full,
                    skip_imputation=skip_imputation,
                )

        return state_outp

    def _tendency_prepare_predict_inputs(
        self,
        batch: dict[str, torch.Tensor],
        pre_processors: dict[str, nn.Module],
        multi_step: int,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]], dict[str, Optional[list]]]:
        xs = {}
        x_t0s = {}
        grid_shard_shapes = dict.fromkeys(batch, None)

        for dataset_name, x in batch.items():
            x_in = x[:, 0:multi_step, None, ...]
            x_t0 = x[:, -1, None, ...]

            if model_comm_group is not None:
                shard_shapes = get_shard_shapes(x_in, -2, model_comm_group=model_comm_group)
                grid_shard_shapes[dataset_name] = [shape[-2] for shape in shard_shapes]
                x_in = shard_tensor(x_in, -2, shard_shapes, model_comm_group)
                shard_shapes = get_shard_shapes(x_t0, -2, model_comm_group=model_comm_group)
                x_t0 = shard_tensor(x_t0, -2, shard_shapes, model_comm_group)

            x_in = pre_processors[dataset_name](x_in, in_place=False)
            x_t0 = pre_processors[dataset_name](x_t0, in_place=False)

            xs[dataset_name] = x_in
            x_t0s[dataset_name] = x_t0

        return (xs, x_t0s), grid_shard_shapes

    def _tendency_finalize_predict_outputs(
        self,
        out: dict[str, torch.Tensor],
        post_processors: dict[str, nn.Module],
        prepared_inputs: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: dict[str, Optional[list]] = None,
        gather_out: bool = True,
        post_processors_tendencies: Optional[dict[str, nn.Module]] = None,
    ) -> dict[str, torch.Tensor]:
        if isinstance(prepared_inputs, tuple) and len(prepared_inputs) >= 2:
            x_t0s = prepared_inputs[1]
        else:
            raise AssertionError("Expected prepared_inputs to contain x_t0s")

        x_t0s = self._tendency_apply_reference_state_truncation(x_t0s, grid_shard_shapes, model_comm_group)
        assert post_processors_tendencies is not None, "Per-step tendency processors must be provided."

        for dataset_name, out_dataset in out.items():
            post_tend = post_processors_tendencies[dataset_name]
            assert post_tend is not None, "Per-step tendency processors must be provided per dataset."
            assert (
                len(post_tend) == out_dataset.shape[1]
            ), "Per-step tendency processors must match the number of output steps."

            states = []
            for step, post_proc in enumerate(post_tend):
                out_step = out_dataset[:, step : step + 1]
                state_step = self._tendency_add_tendency_to_state(
                    {dataset_name: x_t0s[dataset_name]},
                    {dataset_name: out_step},
                    {dataset_name: post_processors[dataset_name]},
                    {dataset_name: post_proc},
                    skip_imputation=True,
                )[dataset_name]
                states.append(state_step)

            out_dataset = torch.cat(states, dim=1)
            out_dataset = self._tendency_apply_imputer_inverse(post_processors, dataset_name, out_dataset)
            if gather_out and model_comm_group is not None:
                out_dataset = gather_tensor(
                    out_dataset,
                    -2,
                    apply_shard_shapes(out_dataset, -2, shard_shapes_dim=grid_shard_shapes[dataset_name]),
                    model_comm_group,
                )
            out[dataset_name] = out_dataset

        return out

    def _tendency_apply_reference_state_truncation(
        self,
        x: dict[str, torch.Tensor],
        grid_shard_shapes: dict[str, list],
        model_comm_group: ProcessGroup,
    ) -> dict[str, torch.Tensor]:
        x_skips = {}

        for dataset_name, in_x in x.items():
            x_skip = self.residual[dataset_name](
                in_x, grid_shard_shapes[dataset_name], model_comm_group, multi_out=self.multi_out
            )
            assert x_skip.ndim == 5, "Residual must be (batch, time, ensemble, grid, vars)."
            x_skips[dataset_name] = x_skip[..., self.data_indices[dataset_name].model.input.prognostic]

        return x_skips

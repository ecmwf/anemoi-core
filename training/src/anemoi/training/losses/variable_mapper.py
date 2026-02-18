# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import functools
from collections.abc import Callable
from typing import Any

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.index_space import IndexSpace
from anemoi.training.losses.scaler_tensor import ScaleTensor
from anemoi.training.utils.enums import TensorDim


class LossVariableMapper(BaseLoss):
    """Loss wrapper to filter variables to compute the loss on."""

    def __init__(
        self,
        loss: dict[str, Any] | Callable | BaseLoss,
        predicted_variables: list[str] | None = None,
        target_variables: list[str] | None = None,
    ):
        """Loss wrapper to filter variables to compute the loss on.

        Parameters
        ----------
        loss : Type[torch.nn.Module] | dict[str, Any]
            wrapped loss
        predicted_variables : list[str] | None
            predicted variables to keep, if None, all variables are kept
        target_variables : list[str] | None
            target variables to keep, if None, all variables are kept
        """
        if predicted_variables and target_variables:
            assert len(predicted_variables) == len(
                target_variables,
            ), "predicted and target variables must have the same length for loss computation"

        super().__init__()

        self._loss_scaler_specification = {}
        assert isinstance(
            loss,
            BaseLoss,
        ), f"Invalid loss type provided: {type(loss)}. Expected a str or dict or BaseLoss."
        self.loss = loss
        if hasattr(self.loss, "scaler"):
            # Share the inner loss scaler so scaler membership and updates remain visible
            # to training/task utilities that inspect `loss.scaler`.
            self.scaler = self.loss.scaler
        self.predicted_variables = predicted_variables
        self.target_variables = target_variables
        self.predicted_indices_by_layout: dict[IndexSpace, list[int]] = {}
        self.target_indices_by_layout: dict[IndexSpace, list[int]] = {}

    @staticmethod
    def _ordered_names_by_index(name_to_index: dict[str, int]) -> list[str]:
        return [name for name, _ in sorted(name_to_index.items(), key=lambda item: item[1])]

    def _get_predicted_indices_for_scaler_variable_axis(self, variable_size: int) -> list[int] | None:
        if variable_size == 1:
            # Broadcast scalers do not need filtering.
            return None
        if not self.predicted_indices_by_layout:
            msg = "LossVariableMapper data indices must be set before adding variable scalers."
            raise RuntimeError(msg)

        layout_variable_sizes: dict[IndexSpace, int] = {
            IndexSpace.MODEL_OUTPUT: len(self.data_indices.model.output.full),
            IndexSpace.DATA_OUTPUT: len(self.data_indices.data.output.full),
            IndexSpace.DATA_FULL: len(self.data_indices.name_to_index),
        }

        matches: dict[IndexSpace, list[int]] = {}
        for layout, layout_size in layout_variable_sizes.items():
            if layout not in self.predicted_indices_by_layout or layout_size != variable_size:
                continue
            indices = self.predicted_indices_by_layout[layout]
            if indices and max(indices) >= variable_size:
                continue
            matches[layout] = indices

        for preferred_layout in (IndexSpace.MODEL_OUTPUT, IndexSpace.DATA_OUTPUT, IndexSpace.DATA_FULL):
            if preferred_layout in matches:
                indices = matches[preferred_layout]
                if indices == list(range(variable_size)):
                    return None
                return indices

        if self.predicted_variables is not None and variable_size == len(self.predicted_variables):
            # Scaler may already be pre-filtered to the requested variable subset.
            return None

        known_sizes = {layout.value: size for layout, size in layout_variable_sizes.items()}
        msg = (
            "Cannot map VARIABLE-axis scaler to a known index space. "
            f"Variable axis size: {variable_size}, "
            f"known sizes: {known_sizes}."
        )
        raise ValueError(msg)

    def _filter_variable_axis_scaler(
        self,
        dimension: int | tuple[int],
        scaler: torch.Tensor,
    ) -> torch.Tensor:
        dims = (dimension,) if isinstance(dimension, int) else tuple(dimension)
        if TensorDim.VARIABLE not in dims:
            return scaler

        # Filter any scaler carrying VARIABLE dim to the selected prediction variables.
        # This supports both pure VARIABLE scalers and mixed-dimension scalers like
        # (BATCH, GRID, VARIABLE).
        variable_axis = dims.index(TensorDim.VARIABLE)
        predicted_indices = self._get_predicted_indices_for_scaler_variable_axis(scaler.shape[variable_axis])
        if predicted_indices is None:
            return scaler

        predicted_indices_tensor = torch.as_tensor(
            predicted_indices,
            device=scaler.device,
            dtype=torch.long,
        )
        return scaler.index_select(variable_axis, predicted_indices_tensor)

    @functools.wraps(ScaleTensor.add_scaler)
    def add_scaler(self, dimension: int | tuple[int], scaler: torch.Tensor, *, name: str | None = None) -> None:
        scaler = self._filter_variable_axis_scaler(dimension, scaler)
        # Pass scalers to the inner loss so they are actually applied during loss computation
        self.loss.add_scaler(dimension=dimension, scaler=scaler, name=name)

    @functools.wraps(ScaleTensor.update_scaler)
    def update_scaler(self, name: str, scaler: torch.Tensor, *, override: bool = False) -> None:
        # Keep update behavior consistent with add_scaler for VARIABLE-axis scalers.
        if hasattr(self.loss, "scaler") and name in self.loss.scaler.tensors:
            dimension = self.loss.scaler.tensors[name][0]
            scaler = self._filter_variable_axis_scaler(dimension, scaler)
        self.loss.update_scaler(name=name, scaler=scaler, override=override)

    @functools.wraps(ScaleTensor.has_scaler_for_dim)
    def has_scaler_for_dim(self, dim: TensorDim) -> bool:
        return self.loss.has_scaler_for_dim(dim=dim)

    @staticmethod
    def _to_layout(layout: IndexSpace | str, *, layout_name: str) -> IndexSpace:
        if isinstance(layout, IndexSpace):
            return layout
        try:
            return IndexSpace(layout)
        except ValueError as e:
            msg = f"Invalid {layout_name}: {layout!r}. Expected one of {[item.value for item in IndexSpace]}"
            raise ValueError(msg) from e

    @staticmethod
    def _build_data_output_positions(data_indices: IndexCollection) -> dict[str, int]:
        data_full_to_pos = {int(var_idx): pos for pos, var_idx in enumerate(data_indices.data.output.full.tolist())}
        return {
            name: data_full_to_pos[idx] for name, idx in data_indices.name_to_index.items() if idx in data_full_to_pos
        }

    @staticmethod
    def _resolve_indices(
        variables: list[str],
        lookup: dict[str, int],
        *,
        layout: IndexSpace,
        role: str,
    ) -> list[int]:
        missing = [name for name in variables if name not in lookup]
        if missing:
            msg = (
                f"Cannot resolve {role} variables {missing} for layout '{layout.value}'. "
                "Check that the configured variables are compatible with this layout."
            )
            raise ValueError(msg)
        return [lookup[name] for name in variables]

    def set_data_indices(self, data_indices: IndexCollection) -> BaseLoss:
        """Hook to set the data indices for the loss."""
        self.data_indices = data_indices
        model_name_to_index = data_indices.model.output.name_to_index
        data_full_name_to_index = data_indices.name_to_index
        data_output_name_to_pos = self._build_data_output_positions(data_indices)

        if self.predicted_variables is None:
            # Preserve tensor index order, not alphabetical includes order.
            self.predicted_variables = self._ordered_names_by_index(model_name_to_index)
        if self.target_variables is None:
            # Default to one-to-one mapping with preserved order.
            self.target_variables = list(self.predicted_variables)

        assert len(self.predicted_variables) == len(
            self.target_variables,
        ), "predicted and target variables must have the same length for loss computation"

        self.predicted_indices_by_layout = {
            IndexSpace.MODEL_OUTPUT: self._resolve_indices(
                self.predicted_variables,
                model_name_to_index,
                layout=IndexSpace.MODEL_OUTPUT,
                role="predicted",
            ),
            IndexSpace.DATA_OUTPUT: self._resolve_indices(
                self.predicted_variables,
                data_output_name_to_pos,
                layout=IndexSpace.DATA_OUTPUT,
                role="predicted",
            ),
            IndexSpace.DATA_FULL: self._resolve_indices(
                self.predicted_variables,
                data_full_name_to_index,
                layout=IndexSpace.DATA_FULL,
                role="predicted",
            ),
        }
        self.target_indices_by_layout = {
            IndexSpace.DATA_OUTPUT: self._resolve_indices(
                self.target_variables,
                data_output_name_to_pos,
                layout=IndexSpace.DATA_OUTPUT,
                role="target",
            ),
            IndexSpace.DATA_FULL: self._resolve_indices(
                self.target_variables,
                data_full_name_to_index,
                layout=IndexSpace.DATA_FULL,
                role="target",
            ),
        }
        if all(name in model_name_to_index for name in self.target_variables):
            self.target_indices_by_layout[IndexSpace.MODEL_OUTPUT] = self._resolve_indices(
                self.target_variables,
                model_name_to_index,
                layout=IndexSpace.MODEL_OUTPUT,
                role="target",
            )
        return self

    @staticmethod
    def _maybe_to_index_list(indexer: Any) -> list[int] | None:
        if isinstance(indexer, int):
            return [indexer]
        if isinstance(indexer, range):
            return list(indexer)
        if isinstance(indexer, list | tuple):
            return [int(idx) for idx in indexer]
        if isinstance(indexer, torch.Tensor):
            if indexer.dtype == torch.bool:
                return torch.nonzero(indexer, as_tuple=False).reshape(-1).tolist()
            return indexer.reshape(-1).tolist()
        return None

    @staticmethod
    def _restore_indexer_type(mapped_indices: list[int], original_indexer: Any) -> Any:
        if isinstance(original_indexer, int):
            return mapped_indices[0] if len(mapped_indices) == 1 else mapped_indices
        if isinstance(original_indexer, torch.Tensor):
            return torch.as_tensor(mapped_indices, device=original_indexer.device, dtype=torch.long)
        return mapped_indices

    def _remap_scaler_indices_for_filtered_pred(
        self,
        scaler_indices: tuple[Any, ...],
        pred_indices: list[int],
    ) -> tuple[tuple[Any, ...], bool]:
        if len(scaler_indices) == 0:
            return scaler_indices, False

        variable_indexer = scaler_indices[-1]
        requested_indices = self._maybe_to_index_list(variable_indexer)
        if requested_indices is None:
            return scaler_indices, False

        pred_index_to_local = {index: pos for pos, index in enumerate(pred_indices)}
        mapped_indices = [pred_index_to_local[index] for index in requested_indices if index in pred_index_to_local]
        remapped_scaler_indices = (*scaler_indices[:-1], self._restore_indexer_type(mapped_indices, variable_indexer))
        return remapped_scaler_indices, len(mapped_indices) == 0

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        pred_layout = kwargs.pop("pred_layout", None)
        target_layout = kwargs.pop("target_layout", None)
        if pred_layout is None or target_layout is None:
            msg = "LossVariableMapper requires both 'pred_layout' and 'target_layout' kwargs."
            raise ValueError(msg)
        pred_layout = self._to_layout(pred_layout, layout_name="pred_layout")
        target_layout = self._to_layout(target_layout, layout_name="target_layout")

        if pred_layout not in self.predicted_indices_by_layout:
            msg = (
                f"pred_layout '{pred_layout.value}' is not available for this filtering configuration. "
                f"Available: {[layout.value for layout in self.predicted_indices_by_layout]}"
            )
            raise ValueError(msg)
        if target_layout not in self.target_indices_by_layout:
            msg = (
                f"target_layout '{target_layout.value}' is not available for this filtering configuration. "
                f"Available: {[layout.value for layout in self.target_indices_by_layout]}"
            )
            raise ValueError(msg)

        pred_indices = self.predicted_indices_by_layout[pred_layout]
        target_indices = self.target_indices_by_layout[target_layout]

        pred_filtered = pred[..., pred_indices]
        target_filtered = target[..., target_indices]

        scaler_indices = kwargs.get("scaler_indices")
        empty_metric_selection = False
        if isinstance(scaler_indices, tuple):
            kwargs["scaler_indices"], empty_metric_selection = self._remap_scaler_indices_for_filtered_pred(
                scaler_indices,
                pred_indices,
            )

        squash = kwargs.get("squash", True)
        if empty_metric_selection:
            if squash:
                return torch.zeros((), dtype=pred.dtype, device=pred.device, requires_grad=False)
            len_model_output = pred.shape[-1]
            return torch.zeros(len_model_output, dtype=pred.dtype, device=pred.device, requires_grad=False)

        if squash:
            return self.loss(pred_filtered, target_filtered, **kwargs)
        len_model_output = pred.shape[-1]
        loss = torch.zeros(len_model_output, dtype=pred.dtype, device=pred.device, requires_grad=False)
        loss_per_variable = self.loss(
            pred_filtered,
            target_filtered,
            **kwargs,
        )
        loss[pred_indices] = loss_per_variable
        return loss

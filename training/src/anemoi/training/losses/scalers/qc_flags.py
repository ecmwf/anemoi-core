# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from collections.abc import Mapping

import torch

from anemoi.models.interface import AnemoiModelInterface
from anemoi.training.losses.scalers.base_scaler import BaseUpdatingScaler
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


class QCFlagsLossScaler(BaseUpdatingScaler):
    """Loss scaler derived from packed ``qc_flags`` values.

    The scaler expects a pre-processor to expose a tensor attribute containing
    packed QC flags from the current batch (default attribute name:
    ``last_qc_flags``). The provided flags are converted to per-point weights
    and then broadcast over the output-variable dimension.

    Intended usage:
    - keep NaN imputation separate from loss weighting (e.g. ``NoMaskImputer``)
    - apply quality-based down-weighting directly in the loss stack
    """

    scale_dims: tuple[TensorDim] = (TensorDim.BATCH_SIZE, TensorDim.GRID, TensorDim.VARIABLE)

    def __init__(
        self,
        qc_var_name: str = "qc_flags",
        bit_weights: Mapping[int | str, float] | None = None,
        bit_names: Mapping[int | str, str] | None = None,
        invalid_bit_indices: list[int] | None = None,
        invalid_weight: float = 0.1,
        default_weight: float = 1.0,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        source_attribute: str = "last_qc_flags",
        norm: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(norm=norm)
        self.qc_var_name = str(qc_var_name)
        self.source_attribute = str(source_attribute)
        self.default_weight = float(default_weight)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)

        parsed_bit_weights: dict[int, float] = {}
        if bit_weights is not None:
            parsed_bit_weights.update({int(bit): float(weight) for bit, weight in bit_weights.items()})

        if invalid_bit_indices is not None:
            for bit in invalid_bit_indices:
                parsed_bit_weights.setdefault(int(bit), float(invalid_weight))

        self.bit_weights = parsed_bit_weights
        self.bit_names = {int(bit): str(name) for bit, name in (bit_names or {}).items()}
        self._logged_once_datasets: set[str] = set()
        del kwargs

    def _get_qc_flags(self, model: AnemoiModelInterface, dataset_name: str | None = None) -> torch.Tensor | None:
        if dataset_name is None:
            return None
        if not hasattr(model, "pre_processors") or dataset_name not in model.pre_processors:
            return None

        dataset_preprocessors = model.pre_processors[dataset_name]
        processors = getattr(dataset_preprocessors, "processors", {})
        if not hasattr(processors, "values"):
            return None

        for processor in processors.values():
            qc_flags = getattr(processor, self.source_attribute, None)
            if isinstance(qc_flags, torch.Tensor) and qc_flags.numel() > 0:
                qc_flags = qc_flags.to(torch.int64)
                while qc_flags.ndim > 2:
                    qc_flags = qc_flags.select(1, 0)
                if qc_flags.ndim == 2:
                    return qc_flags

        return None

    def _weights_from_qc_flags(self, qc_flags: torch.Tensor) -> torch.Tensor:
        weights = torch.full_like(qc_flags, self.default_weight, dtype=torch.float32)

        if len(self.bit_weights) == 0:
            return weights.clamp(min=self.min_weight, max=self.max_weight)

        for bit, bit_weight in self.bit_weights.items():
            bit_is_set = ((qc_flags >> int(bit)) & 1) != 0
            if not torch.any(bit_is_set):
                continue

            candidate = torch.full_like(weights, float(bit_weight))
            if float(bit_weight) <= self.default_weight:
                weights = torch.where(bit_is_set, torch.minimum(weights, candidate), weights)
            else:
                can_upweight = bit_is_set & (weights >= self.default_weight)
                weights = torch.where(can_upweight, torch.maximum(weights, candidate), weights)

        return weights.clamp(min=self.min_weight, max=self.max_weight)

    def _format_flag_value(self, flag: int) -> str:
        if len(self.bit_names) == 0:
            return str(flag)

        active_names = [name for bit, name in sorted(self.bit_names.items()) if ((flag >> bit) & 1) != 0]
        if not active_names:
            return f"{flag}[none]"
        return f"{flag}[{'|'.join(active_names)}]"

    def _log_once(self, dataset_name: str, qc_flags: torch.Tensor, weights_bg: torch.Tensor) -> None:
        if dataset_name in self._logged_once_datasets:
            return

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                self._logged_once_datasets.add(dataset_name)
                return

        flat_flags = qc_flags.detach().reshape(-1)
        flat_weights = weights_bg.detach().reshape(-1)

        uniq_flags, counts = torch.unique(flat_flags, return_counts=True, sorted=True)
        max_items = 12
        uniq_flags = uniq_flags[:max_items]
        counts = counts[:max_items]
        uniq_text = ", ".join(
            [f"{self._format_flag_value(int(flag))}:{int(count)}" for flag, count in zip(uniq_flags, counts)]
        )

        bit_hit_text = ""
        if len(self.bit_weights) > 0:
            numel = max(int(flat_flags.numel()), 1)
            bit_hits = []
            for bit in sorted(self.bit_weights):
                hit_count = int((((flat_flags >> int(bit)) & 1) != 0).sum().item())
                bit_label = f"b{int(bit)}"
                if int(bit) in self.bit_names:
                    bit_label = f"{bit_label}/{self.bit_names[int(bit)]}"
                bit_hits.append(f"{bit_label}={hit_count / numel:.4f}")
            bit_hit_text = " bit_hit_rate(" + ", ".join(bit_hits) + ")"

        LOGGER.info(
            "QCFlagsLossScaler[%s] source=%s shape=%s unique_flags(count<=%d)=[%s] weight[min=%.4f mean=%.4f max=%.4f]%s",
            dataset_name,
            self.source_attribute,
            tuple(qc_flags.shape),
            max_items,
            uniq_text,
            float(flat_weights.min().item()),
            float(flat_weights.mean().item()),
            float(flat_weights.max().item()),
            bit_hit_text,
        )
        self._logged_once_datasets.add(dataset_name)

    def on_batch_start(self, model: AnemoiModelInterface, dataset_name: str | None = None) -> torch.Tensor | None:
        qc_flags = self._get_qc_flags(model=model, dataset_name=dataset_name)
        if qc_flags is None:
            return None

        if dataset_name is None:
            return None

        n_outputs = len(model.data_indices[dataset_name].model.output.name_to_index)
        if n_outputs == 0:
            return None

        weights_bg = self._weights_from_qc_flags(qc_flags)
        self._log_once(dataset_name=dataset_name, qc_flags=qc_flags, weights_bg=weights_bg)
        return weights_bg.unsqueeze(-1).expand(-1, -1, n_outputs)

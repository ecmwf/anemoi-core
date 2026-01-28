# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.training.losses.loss import get_metric_ranges
from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel

from .losses import build_loss_from_config
from .scalers import build_scalers_from_config

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.config_types import Settings
    from anemoi.training.losses.base import BaseLoss
    from anemoi.training.losses.scaler_tensor import TENSOR_SPEC
    from anemoi.training.losses.scalers.base_scaler import BaseUpdatingScaler


def build_training_components_from_config(
    config: Settings,
    *,
    graph_data: Mapping[str, HeteroData],
    statistics: Mapping[str, Any],
    statistics_tendencies: Mapping[str, Any] | None,
    data_indices: Mapping[str, IndexCollection],
    metadata: Mapping[str, Any],
    output_masks: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, TENSOR_SPEC]],
    dict[str, dict[str, BaseUpdatingScaler]],
    dict[str, BaseLoss],
    dict[str, dict[str, BaseLoss]],
    dict[str, dict[str, list[int]]],
]:
    """Build scalers, losses, metrics, and metric ranges from configuration."""
    dataset_variable_groups = get_multiple_datasets_config(config.training.variable_groups)
    loss_configs = get_multiple_datasets_config(config.training.training_loss)
    scalers_configs = get_multiple_datasets_config(config.training.scalers)
    val_metrics_configs = get_multiple_datasets_config(config.training.validation_metrics)
    metrics_to_log = get_multiple_datasets_config(config.training.metrics)

    scalers: dict[str, dict[str, TENSOR_SPEC]] = {}
    updating_scalars: dict[str, dict[str, BaseUpdatingScaler]] = {}
    losses: dict[str, BaseLoss] = {}
    metrics: dict[str, dict[str, BaseLoss]] = {}
    val_metric_ranges: dict[str, dict[str, list[int]]] = {}

    for dataset_name in graph_data:
        metadata_extractor = ExtractVariableGroupAndLevel(
            variable_groups=dataset_variable_groups[dataset_name],
            metadata_variables=metadata["dataset"][dataset_name].get("variables_metadata"),
        )

        dataset_scalers, dataset_updating_scalars = build_scalers_from_config(
            scalers_configs[dataset_name],
            data_indices=data_indices[dataset_name],
            graph_data=graph_data[dataset_name],
            statistics=statistics[dataset_name],
            statistics_tendencies=(statistics_tendencies[dataset_name] if statistics_tendencies is not None else None),
            metadata_extractor=metadata_extractor,
            output_mask=output_masks[dataset_name],
        )
        scalers[dataset_name] = dataset_scalers
        updating_scalars[dataset_name] = dataset_updating_scalars

        val_metric_ranges[dataset_name] = get_metric_ranges(
            metadata_extractor,
            output_data_indices=data_indices[dataset_name].model.output,
            metrics_to_log=metrics_to_log[dataset_name],
        )

        losses[dataset_name] = build_loss_from_config(
            loss_configs[dataset_name],
            scalers=dataset_scalers,
            data_indices=data_indices[dataset_name],
        )

        metrics[dataset_name] = {
            metric_name: build_loss_from_config(
                metric_config,
                scalers=dataset_scalers,
                data_indices=data_indices[dataset_name],
            )
            for metric_name, metric_config in val_metrics_configs[dataset_name].items()
        }

    return scalers, updating_scalars, losses, metrics, val_metric_ranges


__all__ = ["build_training_components_from_config"]

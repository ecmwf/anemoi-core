# (C) Copyright 2024 Anemoi contributors.
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

from anemoi.models.config_types import Settings as ModelSettings
from anemoi.models.interface import AnemoiModelInterface
from anemoi.models.preprocessing import Processors
from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.training.builders.components import build_component
from anemoi.training.config_types import Settings
from anemoi.training.config_types import to_container

if TYPE_CHECKING:

    import torch
    from torch_geometric.data import HeteroData


def _build_processors_for_dataset(
    processors_configs: dict,
    statistics: dict,
    data_indices: dict,
    statistics_tendencies: dict | None = None,
) -> tuple[Processors, Processors, Processors | None, Processors | None]:
    processors = [
        [name, build_component(processor, data_indices=data_indices, statistics=statistics)]
        for name, processor in processors_configs.items()
    ]

    pre_processors = Processors(processors)
    post_processors = Processors(processors, inverse=True)

    pre_processors_tendencies = None
    post_processors_tendencies = None
    if statistics_tendencies is not None:
        processors_tendencies = [
            [name, build_component(processor, data_indices=data_indices, statistics=statistics_tendencies)]
            for name, processor in processors_configs.items()
        ]
        pre_processors_tendencies = Processors(processors_tendencies)
        post_processors_tendencies = Processors(processors_tendencies, inverse=True)

    return pre_processors, post_processors, pre_processors_tendencies, post_processors_tendencies


def build_processors_from_config(
    config: Settings,
    *,
    statistics: dict,
    data_indices: dict,
    statistics_tendencies: dict | None = None,
) -> tuple[dict[str, Processors], dict[str, Processors], dict[str, Processors] | None, dict[str, Processors] | None]:
    data_config = get_multiple_datasets_config(config.data)

    pre_processors: dict[str, Processors] = {}
    post_processors: dict[str, Processors] = {}
    pre_processors_tendencies: dict[str, Processors] = {}
    post_processors_tendencies: dict[str, Processors] = {}

    for dataset_name in statistics:
        pre, post, pre_tend, post_tend = _build_processors_for_dataset(
            data_config[dataset_name]["processors"],
            statistics[dataset_name],
            data_indices[dataset_name],
            statistics_tendencies[dataset_name] if statistics_tendencies is not None else None,
        )
        pre_processors[dataset_name] = pre
        post_processors[dataset_name] = post
        if pre_tend is not None and post_tend is not None:
            pre_processors_tendencies[dataset_name] = pre_tend
            post_processors_tendencies[dataset_name] = post_tend

    return (
        pre_processors,
        post_processors,
        pre_processors_tendencies or None,
        post_processors_tendencies or None,
    )


def _to_model_settings(config: Settings) -> ModelSettings:
    return ModelSettings.model_validate(to_container(config))


def build_model_from_config(
    config: Settings,
    *,
    graph_data: HeteroData | dict[str, HeteroData],
    data_indices: dict,
    statistics: dict,
) -> torch.nn.Module:
    model_config = _to_model_settings(config)
    model_instantiate_config = {
        "_target_": model_config.model.model._target_,
        "_convert_": getattr(model_config.model.model, "_convert_", "all"),
    }
    return build_component(
        model_instantiate_config,
        model_config=model_config,
        data_indices=data_indices,
        statistics=statistics,
        graph_data=graph_data,
    )


def build_model_interface_from_config(
    config: Settings,
    *,
    graph_data: HeteroData | dict[str, HeteroData],
    statistics: dict,
    data_indices: dict,
    metadata: dict[str, Any],
    statistics_tendencies: dict | None = None,
    supporting_arrays: dict | None = None,
) -> AnemoiModelInterface:
    model_config = _to_model_settings(config)
    pre, post, pre_tend, post_tend = build_processors_from_config(
        config,
        statistics=statistics,
        data_indices=data_indices,
        statistics_tendencies=statistics_tendencies,
    )
    model = build_model_from_config(
        config,
        graph_data=graph_data,
        data_indices=data_indices,
        statistics=statistics,
    )
    return AnemoiModelInterface(
        config=model_config,
        graph_data=graph_data,
        statistics=statistics,
        data_indices=data_indices,
        metadata=metadata,
        statistics_tendencies=statistics_tendencies,
        supporting_arrays=supporting_arrays,
        model=model,
        pre_processors=pre,
        post_processors=post,
        pre_processors_tendencies=pre_tend,
        post_processors_tendencies=post_tend,
    )

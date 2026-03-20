# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Factory function for building ModelInterface via Hydra instantiate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing import StepwiseProcessors
from anemoi.models.utils.config import get_multiple_datasets_config

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

    from anemoi.models.interface import ModelInterface
    from anemoi.utils.config import DotDict


@dataclass(frozen=True)
class ModelRuntimeArtifacts:
    """Runtime-built artifacts required to assemble a model."""

    graph_data: HeteroData
    statistics: dict
    statistics_tendencies: dict | None
    data_indices: dict
    metadata: dict
    supporting_arrays: dict


# ---------------------------------------------------------------------------
# Processor-building helpers
# ---------------------------------------------------------------------------


def _build_processor_pair(
    processors_configs: dict,
    data_indices: dict,
    statistics: dict,
) -> tuple[Processors, Processors]:
    processors = [
        [name, instantiate(processor, data_indices=data_indices, statistics=statistics)]
        for name, processor in processors_configs.items()
    ]
    return Processors(processors), Processors(processors, inverse=True)


def _build_tendency_processors(
    processors_configs: dict,
    data_indices: dict,
    statistics_tendencies: dict | None,
    n_step_output: int | None,
) -> tuple[Processors | StepwiseProcessors | None, Processors | StepwiseProcessors | None]:
    if statistics_tendencies is None:
        return None, None

    if "lead_times" not in statistics_tendencies:
        return _build_processor_pair(processors_configs, data_indices, statistics_tendencies)

    lead_times = list(statistics_tendencies.get("lead_times") or [])
    if n_step_output == 1:
        step_stats = statistics_tendencies.get(lead_times[0]) if lead_times else None
        stats_for_tendencies = step_stats or statistics_tendencies
        return _build_processor_pair(processors_configs, data_indices, stats_for_tendencies)

    pre_processors_tendencies = StepwiseProcessors(lead_times)
    post_processors_tendencies = StepwiseProcessors(lead_times)
    for lead_time in lead_times:
        step_stats = statistics_tendencies.get(lead_time)
        if step_stats is None:
            continue
        pre_step, post_step = _build_processor_pair(processors_configs, data_indices, step_stats)
        pre_processors_tendencies.set(lead_time, pre_step)
        post_processors_tendencies.set(lead_time, post_step)
    return pre_processors_tendencies, post_processors_tendencies


def _build_processors_for_dataset(
    processors_configs: dict,
    statistics: dict,
    data_indices: dict,
    statistics_tendencies: dict | None,
    n_step_output: int | None,
) -> tuple:
    """Return (pre, post, pre_tend, post_tend) for one dataset."""
    pre, post = _build_processor_pair(processors_configs, data_indices, statistics)
    pre_tend, post_tend = _build_tendency_processors(
        processors_configs,
        data_indices,
        statistics_tendencies,
        n_step_output,
    )
    return pre, post, pre_tend, post_tend


def _build_processors(
    config: DotDict,
    statistics: dict,
    data_indices: dict,
    statistics_tendencies: dict | None,
) -> tuple[torch.nn.ModuleDict, torch.nn.ModuleDict, torch.nn.ModuleDict, torch.nn.ModuleDict]:
    """Build pre/post processors for all datasets."""
    n_step_output = getattr(config.training, "multistep_output", None)
    data_config = get_multiple_datasets_config(config.data)
    pre_processors = torch.nn.ModuleDict()
    post_processors = torch.nn.ModuleDict()
    pre_processors_tendencies = torch.nn.ModuleDict()
    post_processors_tendencies = torch.nn.ModuleDict()
    for dataset_name in statistics:
        pre, post, pre_tend, post_tend = _build_processors_for_dataset(
            data_config[dataset_name].processors,
            statistics[dataset_name],
            data_indices[dataset_name],
            (statistics_tendencies[dataset_name] if statistics_tendencies is not None else None),
            n_step_output,
        )
        pre_processors[dataset_name] = pre
        post_processors[dataset_name] = post
        if pre_tend is not None:
            pre_processors_tendencies[dataset_name] = pre_tend
            post_processors_tendencies[dataset_name] = post_tend
    return (
        pre_processors,
        post_processors,
        pre_processors_tendencies,
        post_processors_tendencies,
    )


# ---------------------------------------------------------------------------
# Main factory
# ---------------------------------------------------------------------------


def build_anemoi_model(
    *,
    wrapper: DictConfig | None = None,
    backbone: DictConfig,
    training_config: DictConfig,
    data_config: DictConfig,
    dataloader_config: DictConfig,
    graph_config: DictConfig,
    system_config: DictConfig,
    runtime_artifacts: ModelRuntimeArtifacts,
    **model_arch_kwargs,
) -> ModelInterface:
    """Build and return a fully constructed ModelInterface.

    Called by Hydra instantiate(config.model, runtime_artifacts=...) from
    train.py. The builder is a pure assembler: runtime-built artifacts are
    supplied explicitly by the trainer through ``ModelRuntimeArtifacts``.
    """

    def _to_container(v: Any) -> Any:
        if isinstance(v, DictConfig):
            return OmegaConf.to_container(v, resolve=True)
        return v

    full_config_dict = {
        "training": _to_container(training_config),
        "data": _to_container(data_config),
        "dataloader": _to_container(dataloader_config),
        "graph": _to_container(graph_config),
        "system": _to_container(system_config),
        "model": {
            **({"wrapper": _to_container(wrapper)} if wrapper is not None else {}),
            "backbone": _to_container(backbone),
            **{k: _to_container(v) for k, v in model_arch_kwargs.items()},
        },
    }
    config = OmegaConf.create(full_config_dict)

    # Build processors
    (
        pre_processors,
        post_processors,
        pre_processors_tendencies,
        post_processors_tendencies,
    ) = _build_processors(
        config=config,
        statistics=runtime_artifacts.statistics,
        data_indices=runtime_artifacts.data_indices,
        statistics_tendencies=runtime_artifacts.statistics_tendencies,
    )

    wrapper_config = wrapper or OmegaConf.create({"_target_": "anemoi.models.models.AnemoiModel"})

    return instantiate(
        wrapper_config,
        model_config=config,
        graph_data=runtime_artifacts.graph_data,
        statistics=runtime_artifacts.statistics,
        statistics_tendencies=runtime_artifacts.statistics_tendencies,
        data_indices=runtime_artifacts.data_indices,
        metadata=runtime_artifacts.metadata,
        supporting_arrays=runtime_artifacts.supporting_arrays,
        pre_processors=pre_processors,
        post_processors=post_processors,
        pre_processors_tendencies=pre_processors_tendencies,
        post_processors_tendencies=post_processors_tendencies,
    )

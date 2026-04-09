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

import logging

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.models.models import AnemoiDiffusionModel
from anemoi.models.models import AnemoiDiffusionTendencyModel
from anemoi.models.models import AnemoiModel
from anemoi.models.models.diffusion_encoder_processor_decoder import AnemoiDiffusionModelEncProcDec
from anemoi.models.models.diffusion_encoder_processor_decoder import AnemoiDiffusionTendModelEncProcDec
from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing import StepwiseProcessors
from anemoi.models.utils.runtime_artifacts import RuntimeArtifacts
from anemoi.models.utils.supporting_arrays import build_combined_supporting_arrays

LOGGER = logging.getLogger(__name__)


def _build_backbone(backbone_cfg: DictConfig, model_config, data_indices: dict, statistics: dict, graph_data) -> object:
    """Instantiate the backbone NN from its Hydra config."""
    cfg = {
        "_target_": backbone_cfg._target_,
        "_convert_": getattr(backbone_cfg, "_convert_", "none"),
    }
    return instantiate(
        cfg,
        model_config=model_config,
        data_indices=data_indices,
        statistics=statistics,
        graph_data=graph_data,
        _recursive_=False,
    )


def _dispatch_model_class(backbone: object) -> type:
    """Return the AnemoiModel wrapper class appropriate for *backbone*."""
    if isinstance(backbone, AnemoiDiffusionTendModelEncProcDec):
        return AnemoiDiffusionTendencyModel
    if isinstance(backbone, AnemoiDiffusionModelEncProcDec):
        return AnemoiDiffusionModel
    return AnemoiModel


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
    processors_config: DictConfig,
    statistics: dict,
    data_indices: dict,
    statistics_tendencies: dict | None,
    n_step_output: int | None,
) -> tuple[torch.nn.ModuleDict, torch.nn.ModuleDict, torch.nn.ModuleDict, torch.nn.ModuleDict]:
    """Build pre/post processors for all datasets."""
    pre_processors = torch.nn.ModuleDict()
    post_processors = torch.nn.ModuleDict()
    pre_processors_tendencies = torch.nn.ModuleDict()
    post_processors_tendencies = torch.nn.ModuleDict()
    for dataset_name in statistics:
        pre, post, pre_tend, post_tend = _build_processors_for_dataset(
            processors_config[dataset_name],
            statistics[dataset_name],
            data_indices[dataset_name],
            statistics_tendencies[dataset_name] if statistics_tendencies is not None else None,
            n_step_output,
        )
        pre_processors[dataset_name] = pre
        post_processors[dataset_name] = post
        if pre_tend is not None:
            pre_processors_tendencies[dataset_name] = pre_tend
            post_processors_tendencies[dataset_name] = post_tend
    return pre_processors, post_processors, pre_processors_tendencies, post_processors_tendencies


# ---------------------------------------------------------------------------
# Main factory
# ---------------------------------------------------------------------------


def build_anemoi_model(
    *,
    runtime_artifacts: RuntimeArtifacts,
    backbone: DictConfig,
    processors: DictConfig,
    multistep_input: int,
    multistep_output: int | None = None,
    **model_arch_kwargs,
) -> AnemoiModel:
    """Build and return a fully constructed AnemoiModel.

    Called via instantiate_model() from train.py.
    YAML kwargs (backbone, processors, multistep_input, multistep_output, **model_arch_kwargs)
    come from the model YAML. runtime_artifacts is injected from Python.
    """

    def _to_container(v):
        if isinstance(v, DictConfig):
            return OmegaConf.to_container(v, resolve=True)
        return v

    model_config = OmegaConf.create(
        {
            "model": {
                "backbone": _to_container(backbone),
                "multistep_input": multistep_input,
                "multistep_output": multistep_output if multistep_output is not None else 1,
                **{k: _to_container(v) for k, v in model_arch_kwargs.items()},
            },
        }
    )

    # Build processors
    pre_processors, post_processors, pre_processors_tendencies, post_processors_tendencies = _build_processors(
        processors_config=processors,
        statistics=runtime_artifacts.statistics,
        data_indices=runtime_artifacts.data_indices,
        statistics_tendencies=runtime_artifacts.statistics_tendencies,
        n_step_output=multistep_output,
    )

    supporting_arrays = build_combined_supporting_arrays(
        config=model_config,
        graph_data=runtime_artifacts.graph_data,
        supporting_arrays=runtime_artifacts.supporting_arrays,
        dataset_names=list(runtime_artifacts.statistics.keys()),
    )

    backbone_instance = _build_backbone(
        backbone,
        model_config,
        runtime_artifacts.data_indices,
        runtime_artifacts.statistics,
        runtime_artifacts.graph_data,
    )
    model_cls = _dispatch_model_class(backbone_instance)

    return model_cls(
        backbone=backbone_instance,
        n_step_input=multistep_input,
        graph_data=runtime_artifacts.graph_data,
        statistics=runtime_artifacts.statistics,
        statistics_tendencies=runtime_artifacts.statistics_tendencies,
        data_indices=runtime_artifacts.data_indices,
        metadata=runtime_artifacts.metadata,
        supporting_arrays=supporting_arrays,
        pre_processors=pre_processors,
        post_processors=post_processors,
        pre_processors_tendencies=pre_processors_tendencies,
        post_processors_tendencies=post_processors_tendencies,
    )

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

import datetime
import logging
import uuid as _uuid_module
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.models.interface import ModelInterface
from anemoi.models.models import AnemoiModel
from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing import StepwiseProcessors
from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
from anemoi.training.utils.jsonify import map_config_to_primitives
from anemoi.utils.provenance import gather_provenance_info

LOGGER = logging.getLogger(__name__)


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
    backbone: DictConfig,
    training_config: DictConfig,
    data_config: DictConfig,
    dataloader_config: DictConfig,
    graph_config: DictConfig,
    system_config: DictConfig,
    **model_arch_kwargs,
) -> ModelInterface:
    """Build and return a fully constructed ModelInterface.

    Called by Hydra instantiate(config.model) from train.py. All inputs come
    from OmegaConf interpolations in the model yaml — no kwargs from train.py.
    """

    def _to_container(v):
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
            "backbone": _to_container(backbone),
            **{k: _to_container(v) for k, v in model_arch_kwargs.items()},
        },
    }
    config = OmegaConf.create(full_config_dict)

    # Build datamodule to obtain statistics, data_indices, supporting_arrays
    datamodule = AnemoiDatasetsDataModule(config)

    # Build processors
    pre_processors, post_processors, pre_processors_tendencies, post_processors_tendencies = _build_processors(
        config=config,
        statistics=datamodule.statistics,
        data_indices=datamodule.data_indices,
        statistics_tendencies=datamodule.statistics_tendencies,
    )

    # Build graph (load from file or create)
    graph_data = _build_graph(config)

    # Combine supporting arrays with output-mask arrays
    from anemoi.training.utils.supporting_arrays import build_combined_supporting_arrays

    supporting_arrays = build_combined_supporting_arrays(
        config=config,
        graph_data=graph_data,
        supporting_arrays=datamodule.supporting_arrays,
    )

    # Build metadata
    metadata = _build_metadata(config, datamodule)

    return AnemoiModel(
        model_config=config,
        graph_data=graph_data,
        statistics=datamodule.statistics,
        statistics_tendencies=datamodule.statistics_tendencies,
        data_indices=datamodule.data_indices,
        metadata=metadata,
        supporting_arrays=supporting_arrays,
        pre_processors=pre_processors,
        post_processors=post_processors,
        pre_processors_tendencies=pre_processors_tendencies,
        post_processors_tendencies=post_processors_tendencies,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_graph(config: DotDict) -> dict:
    """Load or create graph data."""
    graphs = {}
    dataset_configs = get_multiple_datasets_config(config.dataloader.training)
    for dataset_name, dataset_config in dataset_configs.items():
        graph_path = getattr(config.system.input, "graph", None)
        if graph_path and not getattr(config.graph, "overwrite", False):
            graph_filename = Path(graph_path)
            if graph_filename.name.endswith(".pt"):
                graph_name = graph_filename.name.replace(".pt", f"_{dataset_name}.pt")
                graph_filename = graph_filename.parent / graph_name
            if graph_filename.exists():
                from anemoi.graphs.utils import get_distributed_device

                LOGGER.info("Loading graph data from %s", graph_filename)
                graphs[dataset_name] = torch.load(
                    graph_filename,
                    map_location=get_distributed_device(),
                    weights_only=False,
                )
                continue

        # Create new graph
        from anemoi.graphs.create import GraphCreator

        graph_config = config.graph
        dataset_reader_config = dataset_config.dataset_config
        if isinstance(dataset_reader_config, dict):
            dataset_source = dataset_reader_config.get("dataset")
        else:
            dataset_source = dataset_reader_config
        if (
            dataset_source is not None
            and hasattr(graph_config.nodes, "data")
            and hasattr(graph_config.nodes.data.node_builder, "dataset")
        ):
            graph_config.nodes.data.node_builder.dataset = dataset_source

        save_path = None
        if graph_path:
            save_path = Path(graph_path)
            if save_path.name.endswith(".pt"):
                graph_name = save_path.name.replace(".pt", f"_{dataset_name}.pt")
                save_path = save_path.parent / graph_name

        graphs[dataset_name] = GraphCreator(config=graph_config).create(
            save_path=save_path,
            overwrite=getattr(config.graph, "overwrite", False),
        )

    return graphs


def _build_metadata(config: DotDict, datamodule: AnemoiDatasetsDataModule) -> dict:
    """Build inference/provenance metadata."""
    metadata_inference = {
        "dataset_names": None,  # populated by fill_metadata
        "task": None,  # set by train.py after build
    }
    md_dict = {
        "version": "2.0",
        "config": config,
        "run_id": str(_uuid_module.uuid4()),
        "dataset": None,
        "data_indices": None,
        "provenance_training": gather_provenance_info(),
        "timestamp": datetime.datetime.now(tz=datetime.UTC),
        "metadata_inference": metadata_inference,
        "uuid": None,
    }
    datamodule.fill_metadata(md_dict)
    return map_config_to_primitives(md_dict)

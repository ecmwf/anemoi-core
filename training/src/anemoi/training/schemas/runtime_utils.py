# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Union

from omegaconf import DictConfig

from .system import SystemSchema


def expand_paths(config_system: Union[SystemSchema, DictConfig]) -> Union[SystemSchema, DictConfig]:
    """Expand output paths relative to the configured root directory.

    This runs after the config is loaded so downstream code can always work
    with fully expanded paths instead of rebuilding them in multiple places.
    """
    output_config = config_system.output
    root_output_path = Path(output_config.root) if output_config.root else Path()
    if output_config.plots:
        config_system.output.plots = root_output_path / output_config.plots
    if output_config.profiler:
        config_system.output.profiler = root_output_path / output_config.profiler

    config_system.output.logs.root = (
        root_output_path / output_config.logs.root if output_config.logs.root else root_output_path
    )
    base = config_system.output.logs.root

    output_config.logs.wandb = base / "wandb" if output_config.logs.wandb is None else base / output_config.logs.wandb
    output_config.logs.mlflow = (
        base / "mlflow" if output_config.logs.mlflow is None else base / output_config.logs.mlflow
    )
    output_config.checkpoints.root = (
        root_output_path / output_config.checkpoints.root if output_config.checkpoints.root else root_output_path
    )

    return config_system


def resolve_lineage_run(
    run_id: str | None,
    fork_run_id: str | None,
    parent_run_server2server: str | None,
) -> str | None:
    """Resolve the effective lineage run used for output directories."""
    if run_id:
        return parent_run_server2server or run_id
    if fork_run_id:
        return parent_run_server2server or fork_run_id
    return None


def build_runtime_system(
    system_config: SystemSchema | DictConfig,
    run_id: str | None,
    fork_run_id: str | None,
    parent_run_server2server: str | None,
) -> SystemSchema | DictConfig:
    """Return the finalized system config used by the trainer at runtime.

    This is the runtime step that happens after the base schema has already
    expanded output paths. It appends the effective run lineage to the output
    directories used by checkpoints and plots.
    """
    system_with_lineage = deepcopy(system_config)
    lineage_run = resolve_lineage_run(run_id, fork_run_id, parent_run_server2server)
    if lineage_run is not None:
        system_with_lineage.output.checkpoints.root = Path(system_with_lineage.output.checkpoints.root, lineage_run)
        if system_with_lineage.output.plots is not None:
            system_with_lineage.output.plots = Path(system_with_lineage.output.plots, lineage_run)

    return system_with_lineage

# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import asyncio
import datetime
import logging
from abc import ABC
from abc import abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from packaging import version
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch_geometric.data import HeteroData

from anemoi.graphs.create import GraphCreator
from anemoi.graphs.create import load_graph_from_file
from anemoi.graphs.create import validate_loaded_graph
from anemoi.graphs.projection_helpers import DEFAULT_DATASET_NAME
from anemoi.graphs.projection_helpers import uses_fused_dataset_graph
from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
from anemoi.training.diagnostics.callbacks import CallbacksContext
from anemoi.training.diagnostics.callbacks import get_callbacks
from anemoi.training.diagnostics.logger import get_mlflow_logger
from anemoi.training.diagnostics.logger import get_wandb_logger
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.schemas.base_schema import UnvalidatedBaseSchema
from anemoi.training.schemas.base_schema import convert_to_omegaconf
from anemoi.training.schemas.dataloader import DatasetConfigSchema
from anemoi.training.tasks.base import BaseTask
from anemoi.training.utils.compile import prepare_compilation
from anemoi.training.utils.hydra import instantiate_with_runtime_kwargs
from anemoi.training.utils.jsonify import map_config_to_primitives
from anemoi.training.utils.seeding import SeedContext
from anemoi.training.utils.seeding import derive_seed
from anemoi.training.utils.seeding import get_base_seed
from anemoi.utils.provenance import gather_provenance_info

LOGGER = logging.getLogger(__name__)

PL_VERSION = version.parse(pl.__version__)


def _write_run_identity(config: DictConfig, run_id: str | None, fork_run_id: str | None) -> None:
    """Serialize the resolved run identity onto the config for the MLflow-sync contract.

    ``training.run_id`` / ``training.fork_run_id`` were removed from the schema. Internal
    consumers (the MLflow logger, the ``run_id`` resolution, ``_update_paths`` and the
    dry-run gate) read :attr:`AnemoiTrainer._run_identity` — resolved from the
    ``training.checkpoint.source`` RunSource — not these keys. This write exists solely so
    the flattened config hyperparameters carry ``config.training.run_id`` /
    ``config.training.fork_run_id``, which ``MlFlowSync`` indexes by literal key when
    rewriting offline->online runs. The write runs under a temporary struct unlock so the
    keys can be (re)created even when the config is struct-locked, restoring the prior
    struct flag afterwards.
    """
    prior_struct = OmegaConf.is_struct(config)
    OmegaConf.set_struct(config, False)
    try:
        config.training.run_id = run_id
        config.training.fork_run_id = fork_run_id
    finally:
        OmegaConf.set_struct(config, prior_struct)


class AnemoiTrainer(ABC):
    """Utility class for training the model."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize the Anemoi trainer.

        Parameters
        ----------
        config : DictConfig
            Config object from Hydra.

        """
        # Allow for lower internal precision of float32 matrix multiplications.
        # This can increase performance (and TensorCore usage, where available).
        torch.set_float32_matmul_precision("high")
        # Resolve the config to avoid shenanigans with lazy loading

        if config.config_validation:
            OmegaConf.resolve(config)
            self.config = BaseSchema(**config)

            LOGGER.info("Config validated.")
        else:
            config = OmegaConf.to_object(config)
            self.config = UnvalidatedBaseSchema(**DictConfig(config))

            LOGGER.info("Skipping config validation.")

        self.config = convert_to_omegaconf(self.config)

        # Optionally override the torch default BLAS backend.
        _blas_backend = self.config.training.get("preferred_blas_backend", None)
        if _blas_backend:
            if hasattr(torch.backends.cuda, "preferred_blas_library"):
                torch.backends.cuda.preferred_blas_library(_blas_backend)
                LOGGER.info("BLAS backend forced to %r (config.training.preferred_blas_backend)", _blas_backend)
            else:
                LOGGER.warning(
                    "config.training.preferred_blas_backend=%r ignored: API unavailable in this PyTorch version",
                    _blas_backend,
                )

        # A configured ``training.checkpoint.source`` is the single trigger. The
        # removed ``run_id`` / ``fork_run_id`` / ``system.input.warm_start`` keys
        # are now expressed as a ``RunSource`` (resume/fork) or ``LocalSource``
        # (explicit file) under that block.
        self.start_from_checkpoint = (
            OmegaConf.select(self.config, "training.checkpoint.source", default=None) is not None
        )
        LOGGER.info("Starting from checkpoint: %s", self.start_from_checkpoint)

        self.parent_uuid = None

        # Serialize the resolved run identity onto the config for the MLflow params /
        # ``MlFlowSync`` contract only: ``MlFlowSync`` indexes ``config.training.run_id`` /
        # ``config.training.fork_run_id`` by literal key when rewriting offline->online
        # runs. Internal consumers (the logger, output paths, dry-run gate and the
        # :attr:`run_id` resolution) read :attr:`_run_identity` — resolved from the
        # ``training.checkpoint.source`` RunSource — never these written-back keys.
        _write_run_identity(self.config, run_id=self.run_id, fork_run_id=self._run_identity[1])
        LOGGER.info("Run id: %s", self.config.training.run_id)

        # Get the server2server lineage
        self._get_server2server_lineage()

        # Update paths to contain the run ID
        self._update_paths()

        # Update dry_run attribute, check if checkpoint exists
        self._check_dry_run()

        # Check for dry run, i.e. run id without data
        self._log_information()

    @cached_property
    def task(self) -> BaseTask:
        """Task instance."""
        return instantiate(self.config.task)

    @cached_property
    def datamodule(self) -> Any:
        """DataModule instance and DataSets."""
        datamodule = AnemoiDatasetsDataModule(self.config, self.task)

        # Log information for each dataset
        for name, data in datamodule.ds_train.data.items():
            LOGGER.info("Dataset '%s' - Number of variables: %s", name, len(data.variables))
            LOGGER.info("Dataset '%s' - Variables: %s", name, str(data.variables))
        return datamodule

    @cached_property
    def data_indices(self) -> dict:
        """Returns a dictionary of data indices.

        This is used to slice the data.
        """
        return self.datamodule.data_indices

    @cached_property
    def base_seed(self) -> int:
        """Base seed shared by all ranks."""
        return get_base_seed()

    @cached_property
    def initial_seed(self) -> int:
        """Initial seed for the RNG.

        This sets the same initial seed for all ranks. Ranks are re-seeded in the
        strategy to account for model communication groups.
        """
        initial_seed = derive_seed(self.base_seed, SeedContext.TRAINER)
        rnd_seed = pl.seed_everything(initial_seed, workers=True)
        np_rng = np.random.default_rng(rnd_seed)
        (torch.rand(1), np_rng.random())
        LOGGER.info(
            "Initial seed: Rank %d, initial seed %d, running with random seed: %d",
            self.strategy.global_rank,
            initial_seed,
            rnd_seed,
        )
        return initial_seed

    @cached_property
    @abstractmethod
    def profiler(self) -> None:
        """Abstract method to be used for AnemoiProfiler."""
        return None

    @cached_property
    def _dataset_names(self) -> list[str]:
        """Dataset names derived from the dataloader training config."""
        return list(get_multiple_datasets_config(self.config.dataloader.training).keys())

    @cached_property
    def graph_data(self) -> HeteroData:
        """Graph data built or loaded for the current trainer config."""
        dataset_names = self._dataset_names
        graph_cfg = self.config.graph
        graph_path = self.config.system.input.graph
        save_path = Path(graph_path) if graph_path else None

        # Existing-graph mode: path given but no nodes/edges defined — load as-is.
        is_existing = (
            graph_path is not None
            and not graph_cfg.overwrite
            and not getattr(graph_cfg, "nodes", None)
            and not getattr(graph_cfg, "edges", None)
        )
        if is_existing:
            if not save_path.exists():
                msg = f"Existing graph file not found: {save_path}"
                raise FileNotFoundError(msg)
            graph = load_graph_from_file(save_path)
            fused = uses_fused_dataset_graph(graph, dataset_names)
            required = dataset_names if fused else [DEFAULT_DATASET_NAME]
            validate_loaded_graph(graph, required)
            return graph

        # Build mode: expand config and create graph via GraphCreator.
        graph_config = OmegaConf.create(OmegaConf.to_container(graph_cfg, resolve=False))

        if not uses_fused_dataset_graph(graph_cfg, dataset_names):
            if len(dataset_names) == 1:
                dataset_configs = get_multiple_datasets_config(self.config.dataloader.training)
                dataset_name = dataset_names[0]
                reader_cfg = dataset_configs[dataset_name].dataset_config
                dataset_path = reader_cfg["dataset"] if isinstance(reader_cfg, (DictConfig, dict)) else reader_cfg
                if dataset_path is None:
                    msg = f"Dataset source is None for dataset '{dataset_name}'."
                    raise ValueError(msg)
                data_node_cfg = graph_config.get("nodes", {}).get(DEFAULT_DATASET_NAME)
                if (
                    data_node_cfg is not None
                    and hasattr(data_node_cfg, "node_builder")
                    and hasattr(data_node_cfg.node_builder, "dataset")
                ):
                    # Build the dataset value for the graph node builder.
                    # Start with the bare dataset path, then merge any keys from
                    # dataset_config that are not part of DatasetConfigSchema
                    # (e.g. check_variables_compatibility).  Schema-managed keys
                    # are excluded to avoid forwarding complex validated objects
                    # or None defaults (e.g. select: None, frequency: Frequency(...)).
                    graph_dataset_value: str | dict = {"dataset": dataset_path}
                    if isinstance(reader_cfg, (DictConfig, dict)):
                        _schema_keys = set(DatasetConfigSchema.model_fields.keys())
                        graph_dataset_value.update(
                            {k: v for k, v in dict(reader_cfg).items() if k not in _schema_keys and v is not None},
                        )
                    data_node_cfg.node_builder.dataset = graph_dataset_value
            else:
                msg = (
                    "Multiple datasets require a fused graph config with one node group per dataset. "
                    f"Received datasets {dataset_names} but graph nodes "
                    f"{list(graph_cfg.nodes.keys())}."
                )
                raise ValueError(msg)

        # Try loading existing saved graph before rebuilding.
        overwrite = graph_cfg.get("overwrite", False)
        if save_path and save_path.exists() and not overwrite:
            fused = uses_fused_dataset_graph(graph_cfg, dataset_names)
            required = dataset_names if fused else [DEFAULT_DATASET_NAME]
            graph = load_graph_from_file(save_path)
            validate_loaded_graph(graph, required)
            return graph

        return GraphCreator(graph_config).create(save_path=save_path, overwrite=overwrite)

    def _validate_transfer_learning_datasets(
        self,
        model: pl.LightningModule,
    ) -> None:
        """Validate dataset compatibility between checkpoint and config for transfer learning.

        This method handles multiple transfer learning scenarios when loading a checkpoint:

        - **Scenario 1**: Exact match (checkpoint datasets == config datasets)
          All weights are loaded normally.

        - **Scenario 2**: Adding datasets (config has datasets not in checkpoint)
          Missing datasets will have their encoder & decoder weights randomly initialized.
          The shared processor weights are still transferred.

        - **Scenario 3**: Removing datasets (checkpoint has datasets not in config)
          Extra datasets in checkpoint are ignored (their weights are not loaded).

        - **Scenario 4**: Swapping datasets (combination of scenarios 2 and 3)
          Some datasets are added (randomly initialized), others are removed (ignored).

        -----
        - Logs warnings for datasets that are missing or ignored
        - Logs info summary of loaded and initialized datasets
        - The shared processor weights are always transferred
        """
        loaded_datasets = []
        initialized_datasets = []

        # Check if checkpoint has multi-dataset format
        if not isinstance(model._ckpt_model_name_to_index, dict):
            return

        # Opt-in: allow fine-tuning into a model with FEWER variables (the current data is a
        # strict subset of the checkpoint's variables, issue #838) instead of raising.
        allow_subset = bool(self.config.training.get("allow_variable_subset", False))

        # Validate each dataset in current config against checkpoint
        for dataset_name, data_indices in self.data_indices.items():
            if dataset_name in model._ckpt_model_name_to_index:
                # Dataset found in checkpoint - validate variables match
                ckpt_name_to_index = model._ckpt_model_name_to_index[dataset_name]
                data_indices.compare_variables(
                    ckpt_name_to_index,
                    data_indices.name_to_index,
                    allow_subset=allow_subset,
                )
                loaded_datasets.append(dataset_name)
            else:
                # Dataset not found in checkpoint - will be randomly initialized
                LOGGER.warning(
                    "Dataset '%s' NOT found in checkpoint. Encoder & decoder weights will be randomly initialized!",
                    dataset_name,
                )
                initialized_datasets.append(dataset_name)

        # Check for datasets in checkpoint but not in config
        ignored_datasets = [name for name in model._ckpt_model_name_to_index if name not in self.data_indices]
        if ignored_datasets:
            for ignored_dataset in ignored_datasets:
                LOGGER.warning(
                    "Dataset '%s' found in checkpoint but NOT in config. "
                    "Encoder & decoder weights for '%s' will be ignored.",
                    ignored_dataset,
                    ignored_dataset,
                )

        # Log summary of what was loaded
        if loaded_datasets:
            LOGGER.info("Successfully loaded weights for datasets: %s", loaded_datasets)
        if initialized_datasets:
            LOGGER.info("Randomly initialized weights for datasets: %s", initialized_datasets)

    def _validate_transfer_learning_units(
        self,
        model: pl.LightningModule,
    ) -> None:
        """Validate variable unit compatibility between checkpoint and current dataset.

        Compares the variables_metadata stored on the model (extracted from the checkpoint
        during loading) with the current dataset's variables_metadata. For shared datasets,
        the variables are assumed to match exactly.

        Raises
        ------
        ValueError
            If variables have incompatible units between checkpoint and dataset.

        Warns
        -----
        If variables_metadata is missing from either the checkpoint or the current dataset,
        a warning is logged and the check is skipped.
        """
        from anemoi.training.utils.variables_metadata import check_variables_metadata_compatibility

        ckpt_variables_metadata = getattr(model, "_ckpt_variables_metadata", None)
        compat_cfg = self.config.training.get("check_variables_compatibility", {})
        compat_options = (
            OmegaConf.to_container(compat_cfg, resolve=True) if OmegaConf.is_config(compat_cfg) else (compat_cfg or {})
        )
        # Opt-in fine-tuning into FEWER variables (issue #838): tolerate checkpoint-only
        # variables in the unit check, mirroring the dataset variable-order check.
        allow_subset = bool(self.config.training.get("allow_variable_subset", False))
        check_variables_metadata_compatibility(
            ckpt_variables_metadata,
            self.datamodule.metadata,
            allow_subset=allow_subset,
            **compat_options,
        )

    @cached_property
    def model(self) -> pl.LightningModule:
        """Provide the model instance."""
        kwargs = {
            "config": self.config,
            "task": self.task,
            "data_indices": self.data_indices,
            "graph_data": self.graph_data,
            "metadata": self.metadata,
            "statistics": self.datamodule.statistics,
            "statistics_tendencies": self.datamodule.statistics_tendencies,
            "supporting_arrays": self.supporting_arrays,
        }

        training_method_cfg = self.config.training.method
        model = instantiate_with_runtime_kwargs(training_method_cfg, **kwargs)  # Task -> pl.LightningModule

        # Declarative checkpoint pipeline (opt-in via ``training.checkpoint``): when
        # configured, the source -> loading -> modifier pipeline owns weight loading
        # and model modification. Without ``training.checkpoint`` this is a fresh run.
        # The legacy load_weights_only / transfer_learning / submodules_to_freeze keys
        # are rejected at config validation (schemas.base_schema._DEPRECATED_KEYS).
        if self._checkpoint_pipeline_configured():
            return self._load_via_checkpoint_pipeline(model)

        return model

    def _checkpoint_pipeline_configured(self) -> bool:
        """Return whether a declarative checkpoint pipeline is configured.

        ``True`` when ``training.checkpoint`` is present (a ``source``, ``loading``
        and/or ``modifiers`` block). When absent, :meth:`model` is a fresh run; the
        legacy ``load_weights_only`` / ``transfer_learning`` / ``submodules_to_freeze``
        keys are rejected at config validation.
        """
        return OmegaConf.select(self.config, "training.checkpoint", default=None) is not None

    def _load_via_checkpoint_pipeline(
        self,
        model: pl.LightningModule,
    ) -> pl.LightningModule:
        """Load weights and apply modifiers through the checkpoint pipeline.

        Runs the configured ``training.checkpoint`` ``source`` -> ``loading`` ->
        ``modifiers`` stages. This is the single checkpoint load path.
        :attr:`last_checkpoint` resolves the ckpt_path independently from the source
        config (via :meth:`RunSource.resolve_path`), so it is not cached here.

        Parameters
        ----------
        model : pl.LightningModule
            The freshly instantiated training module whose parameter slots the
            loading stage fills in place (no re-instantiation).

        Returns
        -------
        pl.LightningModule
            The same module, with checkpoint weights loaded and any modifier
            stages applied.
        """
        from anemoi.training.checkpoint import build_checkpoint_pipeline
        from anemoi.training.checkpoint.base import CheckpointContext

        has_loading = OmegaConf.select(self.config, "training.checkpoint.loading", default=None) is not None

        context = CheckpointContext(model=model, config=self.config)
        # Runtime, logger-derived server-to-server lineage cannot reach a RunSource
        # through Hydra; the builder injects it into the source config before
        # instantiation (a no-op for non-RunSource sources).
        pipeline = build_checkpoint_pipeline(
            self.config,
            parent_run_server2server=getattr(self, "parent_run_server2server", None),
            fork_run_server2server=getattr(self, "fork_run_server2server", None),
        )

        executed = asyncio.run(pipeline.execute(context))
        loaded_model = executed.model

        # Trainer-side parity until the dataset/units validators move into the
        # pipeline: when weights were loaded, keep the current config's data
        # indices and run the transfer-learning compatibility checks.
        if has_loading:
            loaded_model.data_indices = self.data_indices
            self._validate_transfer_learning_datasets(loaded_model)
            self._validate_transfer_learning_units(loaded_model)
        return loaded_model

    @cached_property
    def _run_identity(self) -> tuple[str | None, str | None]:
        """Resolved ``(run_id, fork_run_id)`` from the configured checkpoint source.

        The ``training.checkpoint.source`` RunSource is the single source of truth for
        run lineage (resume vs fork); this reads it via
        :func:`~anemoi.training.checkpoint.sources.run.run_identity_from_config`. Every
        run-identity consumer — the MLflow logger (:attr:`_logger_kwargs`), the
        :attr:`run_id` resolution, the output paths (:meth:`_update_paths`) and the
        dry-run gate — reads this instead of the config, so the trainer never mutates
        the config to communicate the identity (a ``LocalSource`` / no source yields
        ``(None, None)`` — a fresh run).
        """
        from anemoi.training.checkpoint.sources.run import run_identity_from_config

        return run_identity_from_config(self.config)

    @cached_property
    def run_id(self) -> str:
        """Unique identifier for the current run."""
        resume_run_id = self._run_identity[0]

        # Resume: reuse the run id being resumed.
        if resume_run_id:
            return resume_run_id

        # Fork or a fresh run: let MLflow mint a new id (the run is tagged as forked
        # when _run_identity[1] is set), otherwise fall back to a random uuid4.
        if self.logger and self.logger.logger_name == "mlflow":
            return self.mlflow_logger.run_id

        import uuid

        return str(uuid.uuid4())

    @cached_property
    def last_checkpoint(self) -> Path | None:
        """Path to the checkpoint to resume from, for Lightning's ``ckpt_path``.

        Resolved directly from the configured ``training.checkpoint.source`` — no need
        to build the model. A ``RunSource`` yields
        ``<checkpoints.root.parent>/<id>/last.ckpt`` via :meth:`RunSource.resolve_path`
        (shared with the source stage so the two cannot drift); a ``LocalSource`` yields
        its explicit file. Remote sources (S3/HTTP) record no local path (``None``),
        which is why warm start is restricted to Local/Run sources (enforced by
        :func:`~anemoi.training.checkpoint.builder.reject_unsupported_warm_start`).

        Returns ``None`` when there is nothing to resume. A configured-but-missing run
        checkpoint still surfaces from the source stage during model build
        (``RuntimeError`` on rank 0 / ``CheckpointNotFoundError`` for an explicit file);
        the rank-0 policy lives solely in the acquisition layer, not here.
        """
        if not self.start_from_checkpoint:
            return None

        source = OmegaConf.select(self.config, "training.checkpoint.source", default=None)
        target = OmegaConf.select(source, "_target_", default="") or ""

        if target.endswith("RunSource"):
            from anemoi.training.checkpoint.sources.run import RunSource

            run_id = OmegaConf.select(source, "run_id", default=None)
            if run_id is None:
                return None
            return RunSource.resolve_path(
                self.config,
                run_id,
                bool(OmegaConf.select(source, "fork", default=False)),
                getattr(self, "parent_run_server2server", None),
                getattr(self, "fork_run_server2server", None),
            )

        if target.endswith("LocalSource"):
            path = OmegaConf.select(source, "path", default=None)
            return Path(path) if path is not None else None

        # Remote sources (S3/HTTP) provide no local path to hand Lightning.
        return None

    @cached_property
    def callbacks(self) -> list[pl.callbacks.Callback]:
        callbacks_context = CallbacksContext(
            diagnostics=self.config.diagnostics,
            checkpoints_output=self.config.system.output.checkpoints,
            plots_output=self.config.system.output.plots,
            wandb_enabled=getattr(getattr(self.config.diagnostics.log, "wandb", None), "enabled", False),
            mlflow_enabled=getattr(getattr(self.config.diagnostics.log, "mlflow", None), "enabled", False),
            weight_averaging_config=getattr(self.config.training, "weight_averaging", None),
        )
        return get_callbacks(callbacks_context)

    @cached_property
    def metadata(self) -> dict:
        """Metadata and provenance information."""
        metadata_inference = {
            "seed": self.initial_seed,
            "base_seed": self.base_seed,
            "run_id": self.run_id,
            "dataset_names": None,  # will be populated in DataModule
            "task": None,  # will be populated in BaseTrainingModule
        }
        # Store metadata needed in inference in a separate dict "metadata_inference"
        # For each group, we add a dictionary with:
        # - data_indices, containing name_to_index mappings
        # - variable_types, specifyting forcing/diagnostics/prognostic/target splits
        # - shapes, specifying the shape of the input tensor (for dimensions where the size is fixed)
        # - timesteps, specifying the time steps used during training for input and output

        md_dict = {
            "version": "2.0",
            "config": self.config,
            "seed": self.initial_seed,
            "base_seed": self.base_seed,
            "run_id": self.run_id,
            "task": None,  # will be populated in Task
            "dataset": None,  # will be populated in DataModule
            "data_indices": None,  # will be populated in DataModule
            "provenance_training": gather_provenance_info(),
            "timestamp": datetime.datetime.now(tz=datetime.UTC),
            "metadata_inference": metadata_inference,
            "uuid": None,  # will be populated in checkpoint callback
        }
        self.datamodule.fill_metadata(md_dict)
        self.task.fill_metadata(md_dict)
        return map_config_to_primitives(md_dict)

    @cached_property
    def supporting_arrays(self) -> dict:
        return self.datamodule.supporting_arrays

    @cached_property
    def _logger_kwargs(self) -> dict:
        """Shared keyword arguments for all loggers."""
        run_id, fork_run_id = self._run_identity
        return {
            "run_id": run_id,
            "fork_run_id": fork_run_id,
            "paths": self.config.system.output,
            "logger_config": self.config.diagnostics.log,
        }

    @cached_property
    def mlflow_logger(self) -> None:
        """Lazily initialize and cache the MLflow logger."""
        LOGGER.info("Initializing MLflow logger lazily...")
        return get_mlflow_logger(**self._logger_kwargs)

    @cached_property
    def wandb_logger(self) -> None:
        """Lazily initialize and cache the W&B logger."""
        LOGGER.info("Initializing W&B logger lazily...")
        kwargs = self._logger_kwargs.update({"model": self.model})
        return get_wandb_logger(**kwargs)

    @cached_property
    def logger(self) -> Logger | None:
        """Lazily build all enabled logger."""
        diagnostics_log = self.config.diagnostics.log

        logger_types = ("wandb", "mlflow")

        for logger_type in logger_types:
            logger_cfg = getattr(diagnostics_log, logger_type, None)
            if getattr(logger_cfg, "enabled", False):
                LOGGER.info("%s logger enabled", logger_type.upper())
                return getattr(self, f"{logger_type}_logger")

        return False  # No logger enabled

    @cached_property
    def accelerator(self) -> str:
        assert self.config.system.hardware.accelerator in {
            "auto",
            "cpu",
            "gpu",
            "cuda",
            "tpu",
        }, f"Invalid accelerator ({self.config.system.hardware.accelerator}) in system.hardware config."

        if self.config.system.hardware.accelerator == "cpu":
            LOGGER.info("WARNING: Accelerator set to CPU, this should only be used for debugging.")
        # For GPU, check if 'cuda' is available
        # For historical reasons, on AMD GPUs the API is still called 'cuda' even though ROCm is used.
        if (
            self.config.system.hardware.accelerator == "gpu" or self.config.system.hardware.accelerator == "cuda"
        ) and not torch.cuda.is_available():
            msg = (
                "GPU accelerator requested but running on GPUs is not possible. "
                "Possible reasons include no GPUs being available on the node,"
                "or PyTorch not being installed with CUDA/ROCm support. "
                "*Note* on aarch64 systems, the default torch wheel does not have CUDA/ROCm support."
                "To install PyTorch with CUDA support on aarch64, you should pass the index-url argument to pip install"
                "e.g. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`"
                "Please check your setup and run "
                "`python -c 'import torch; print(torch.cuda.is_available())'` to confirm GPU support.",
            )
            raise RuntimeError(msg)
        return self.config.system.hardware.accelerator

    def _log_information(self) -> None:
        # Log number of variables (features) per dataset
        for dataset_name, data in self.datamodule.ds_train.data.items():
            num_forcing_features = len(self.data_indices[dataset_name].forcing)
            num_fc_features = len(data.variables) - num_forcing_features
            LOGGER.info("Dataset '%s' - Total number of prognostic variables: %d", dataset_name, num_fc_features)
            LOGGER.info(
                "Dataset '%s' - Total number of auxiliary variables: %d",
                dataset_name,
                num_forcing_features,
            )

        # Log learning rate multiplier when running single-node, multi-GPU and/or multi-node
        total_number_of_model_instances = (
            self.config.system.hardware.num_nodes
            * self.config.system.hardware.num_gpus_per_node
            / self.config.system.hardware.num_gpus_per_model
        )

        LOGGER.info(
            "Total GPU count / model group size: %d - NB: the learning rate will be scaled by this factor!",
            total_number_of_model_instances,
        )
        LOGGER.info(
            "Effective learning rate: %.3e",
            int(total_number_of_model_instances) * self.config.training.optimization.lr,
        )

        if self.config.training.max_epochs is not None and self.config.training.max_steps not in (None, -1):
            lr_scheduler_cfg = getattr(self.config.training.optimization, "lr_scheduler", None)
            LOGGER.info(
                "Training limits: max_epochs=%d, max_steps=%d. "
                "Training will stop when either limit is reached first. "
                "Learning rate scheduler: %s.",
                self.config.training.max_epochs,
                self.config.training.max_steps,
                lr_scheduler_cfg or "none",
            )

    def _get_server2server_lineage(self) -> None:
        """Get the server2server lineage."""
        self.parent_run_server2server = None
        self.fork_run_server2server = None
        if self.logger and self.logger.logger_name == "mlflow":
            self.parent_run_server2server = self.mlflow_logger._parent_run_server2server
            LOGGER.info("Parent run server2server: %s", self.parent_run_server2server)
            self.fork_run_server2server = self.mlflow_logger._fork_run_server2server
            LOGGER.info("Fork run server2server: %s", self.fork_run_server2server)

    def _update_paths(self) -> None:
        """Update the paths in the configuration."""
        self.lineage_run = None
        if self.run_id:  # when using mlflow only rank0 will have a run_id except when resuming runs
            # Multi-gpu new runs or forked runs - only rank 0
            # Multi-gpu resumed runs - all ranks
            self.lineage_run = self.parent_run_server2server or self.run_id
            self.config.system.output.checkpoints.root = Path(
                self.config.system.output.checkpoints.root,
                self.lineage_run,
            )
            self.config.system.output.plots = Path(self.config.system.output.plots, self.lineage_run)
        elif self._run_identity[1]:
            # WHEN USING MANY NODES/GPUS
            self.lineage_run = self.parent_run_server2server or self._run_identity[1]
            # Only rank non zero in the forked run will go here
            self.config.system.output.checkpoints.root = Path(
                self.config.system.output.checkpoints.root,
                self.lineage_run,
            )

        LOGGER.info("Checkpoints path: %s", self.config.system.output.checkpoints)
        LOGGER.info("Plots path: %s", self.config.system.output.plots)

    @rank_zero_only
    def _check_dry_run(self) -> None:
        """Check if the run ID is dry, e.g. without a checkpoint.

        If the run ID is dry, the training will not be started.
        This is used to check the run can be restarted from the checkpoint.
        """
        self.dry_run = False
        if self.logger and self.logger.logger_name == "mlflow":
            # Check if the run ID is dry - e.g. without a checkpoint
            self.dry_run = (
                self.mlflow_logger._parent_dry_run and not Path(self.config.system.output.checkpoints.root).is_dir()
            )
            is_fork = bool(self._run_identity[1])
            self.start_from_checkpoint = False if (self.dry_run and not is_fork) else self.start_from_checkpoint
            LOGGER.info("Dry run: %s", self.dry_run)

    @cached_property
    def strategy(self) -> Any:
        return instantiate(
            self.config.training.strategy,
            static_graph=not self.config.training.accum_grad_batches > 1,
        )

    def _skip_lightning_restore(self) -> bool:
        """Whether to skip Lightning's ``ckpt_path`` full-state restore.

        The checkpoint pipeline applies the checkpoint *weights* to the model at build
        (:meth:`model`), so Lightning must not redo that. ``ckpt_path`` additionally
        makes Lightning restore the optimizer, scheduler and loop/epoch state — which is
        wanted only for warm start. Concretely, keyed on the configured
        ``training.checkpoint.loading``:

        - **kept** (returns ``False``, Lightning restores full training state): a loader
          that declares
          :attr:`~anemoi.training.checkpoint.loading.base.LoadingStrategy.restores_training_state`
          — i.e. ``WarmStartLoader``.
        - **suppressed** (returns ``True``, ``ckpt_path=None``, weights-only): every other
          loader — ``WeightsOnlyLoader`` / ``ColdStartLoader`` / ``TransferLearningLoader``
          — because they intentionally start fresh training state. A run with no loading
          configured returns ``False`` (nothing to suppress).
        """
        loading = OmegaConf.select(self.config, "training.checkpoint.loading", default=None)
        if loading is None:
            return False
        target = OmegaConf.select(loading, "_target_", default="") or ""
        if not target:
            return False

        from hydra.utils import get_class

        try:
            loader_cls = get_class(target)
        except (ImportError, ValueError):
            # An unresolvable loader fails the pipeline build elsewhere; do not
            # suppress the resume here.
            return False
        return not getattr(loader_cls, "restores_training_state", False)

    @cached_property
    def fit_parameters(self) -> Any:
        """Options to be passed to trainer.fit().

        This builds up different arguments based on the version of pytorch lightning.
        From 2.6 onwards pytorch-lightning has now exposed the weights_only flag to be
        consistent with Pytorch's behaviour.
        Refer to https://docs.pytorch.org/docs/stable/generated/torch.load.html for more details.
        `weights_only` does not refer to loading the optimizer. Pytorch_lightning controls this
        via the checkpoint connector. If a ckpt_path is passed then all states are loaded. If no ckpt_path
        is passed and just the `load_from_checkpoint` interface is used - then optimizer states are skipped.
        """
        params = {}

        params["model"] = self.model
        params["datamodule"] = self.datamodule
        # ckpt_path drives Lightning's optimizer/scheduler/epoch restore. Only warm start
        # wants it (the pipeline already loaded the weights); weights-only / cold-start /
        # transfer-learning suppress it so training state starts fresh. See
        # :meth:`_skip_lightning_restore`.
        params["ckpt_path"] = None if self._skip_lightning_restore() else self.last_checkpoint

        if version.parse("2.6.0") <= PL_VERSION:
            params["weights_only"] = False
        return params

    def train(self) -> None:
        """Training entry point."""
        LOGGER.debug("Setting up trainer..")

        trainer = pl.Trainer(
            accelerator=self.accelerator,
            callbacks=self.callbacks,
            deterministic=self.config.training.deterministic,
            detect_anomaly=self.config.diagnostics.debug.anomaly_detection,
            strategy=self.strategy,
            devices=self.config.system.hardware.num_gpus_per_node,
            num_nodes=self.config.system.hardware.num_nodes,
            precision=self.config.training.precision,
            max_epochs=self.config.training.max_epochs,
            max_steps=self.config.training.max_steps or -1,
            logger=self.logger,
            profiler=self.profiler,
            log_every_n_steps=self.config.diagnostics.log.interval,
            # run a fixed no of batches per epoch (helpful when debugging)
            limit_train_batches=self.config.dataloader.limit_batches.training,
            limit_val_batches=self.config.dataloader.limit_batches.validation,
            num_sanity_val_steps=self.config.training.num_sanity_val_steps,
            accumulate_grad_batches=self.config.training.accum_grad_batches,
            gradient_clip_val=self.config.training.gradient_clip.val,
            gradient_clip_algorithm=self.config.training.gradient_clip.algorithm,
            # we have our own DDP-compliant sampler logic baked into the dataset
            use_distributed_sampler=False,
            enable_progress_bar=self.config.diagnostics.enable_progress_bar,
            check_val_every_n_epoch=getattr(self.config.diagnostics, "check_val_every_n_epoch", 1),
        )

        self.model = prepare_compilation(self.model, self.config.model, self.config.training)

        LOGGER.debug("Starting training..")

        trainer.fit(**self.fit_parameters)

        if self.config.diagnostics.print_memory_summary:
            LOGGER.info("memory summary: %s", torch.cuda.memory_summary(device=0))

        LOGGER.debug("---- DONE. ----")


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(config: DictConfig) -> None:
    AnemoiTrainer(config).train()


if __name__ == "__main__":
    main()

# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


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
from hydra.utils import get_class
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from scipy.sparse import load_npz
from torch_geometric.data import HeteroData

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
from anemoi.training.diagnostics.callbacks import get_callbacks
from anemoi.training.diagnostics.logger import get_mlflow_logger
from anemoi.training.diagnostics.logger import get_tensorboard_logger
from anemoi.training.diagnostics.logger import get_wandb_logger
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.schemas.base_schema import UnvalidatedBaseSchema
from anemoi.training.schemas.base_schema import convert_to_omegaconf
from anemoi.training.utils.checkpoint import freeze_submodule_by_name
from anemoi.training.utils.checkpoint import transfer_learning_loading
from anemoi.training.utils.jsonify import map_config_to_primitives
from anemoi.training.utils.seeding import get_base_seed
from anemoi.utils.provenance import gather_provenance_info

LOGGER = logging.getLogger(__name__)


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

        self.start_from_checkpoint = (
            bool(self.config.training.run_id)
            or bool(self.config.training.fork_run_id)
            or bool(self.config.hardware.files.warm_start)
        )
        LOGGER.info("Starting from checkpoint: %s", self.start_from_checkpoint)

        self.load_weights_only = self.config.training.load_weights_only
        self.parent_uuid = None

        self.config.training.run_id = self.run_id
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
    def datamodule(self) -> Any:
        """DataModule instance and DataSets."""
        datamodule = AnemoiDatasetsDataModule(convert_to_omegaconf(self.config), self.graph_data)
        self.config.data.num_features = len(datamodule.ds_train.data.variables)
        LOGGER.info("Number of data variables: %s", str(len(datamodule.ds_train.data.variables)))
        LOGGER.info("Variables: %s", str(datamodule.ds_train.data.variables))
        return datamodule

    @cached_property
    def data_indices(self) -> dict:
        """Returns a dictionary of data indices.

        This is used to slice the data.
        """
        return self.datamodule.data_indices

    @cached_property
    def initial_seed(self) -> int:
        """Initial seed for the RNG.

        This sets the same initial seed for all ranks. Ranks are re-seeded in the
        strategy to account for model communication groups.
        """
        initial_seed = get_base_seed()
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
    def graph_data(self) -> HeteroData:
        """Graph data.

        Creates the graph in all workers.
        """
        if self.config.hardware.files.graph is not None:
            graph_filename = Path(
                self.config.hardware.paths.graph,
                self.config.hardware.files.graph,
            )

            if graph_filename.exists() and not self.config.graph.overwrite:
                from anemoi.graphs.utils import get_distributed_device

                LOGGER.info("Loading graph data from %s", graph_filename)
                return torch.load(graph_filename, map_location=get_distributed_device(), weights_only=False)

        else:
            graph_filename = None

        from anemoi.graphs.create import GraphCreator

        graph_config = convert_to_omegaconf(self.config).graph
        return GraphCreator(config=graph_config).create(
            save_path=graph_filename,
            overwrite=self.config.graph.overwrite,
        )

    @cached_property
    @abstractmethod
    def profiler(self) -> None:
        """Abstract method to be used for AnemoiProfiler."""
        return None

    @cached_property
    def truncation_data(self) -> dict:
        """Truncation data.

        Loads truncation data.
        """
        truncation_data = {}
        if self.config.hardware.files.truncation is not None:
            truncation_data["down"] = load_npz(
                Path(self.config.hardware.paths.truncation, self.config.hardware.files.truncation),
            )
        if self.config.hardware.files.truncation_inv is not None:
            truncation_data["up"] = load_npz(
                Path(self.config.hardware.paths.truncation, self.config.hardware.files.truncation_inv),
            )

        return truncation_data

    @cached_property
    def model(self) -> pl.LightningModule:
        """Provide the model instance."""
        assert (
            not (
                "GLU" in self.config.model.processor.layer_kernels["Activation"]["_target_"]
                and ".Transformer" in self.config.model.processor.target_
            )
            and not (
                "GLU" in self.config.model.encoder.layer_kernels["Activation"]["_target_"]
                and ".Transformer" in self.config.model.encoder.target_
            )
            and not (
                "GLU" in self.config.model.decoder.layer_kernels["Activation"]["_target_"]
                and ".Transformer" in self.config.model.decoder.target_
            )
        ), "GLU activation function is not supported in Transformer models, due to fixed dimensions. "
        "Please use a different activation function."

        kwargs = {
            "config": self.config,
            "data_indices": self.data_indices,
            "graph_data": self.graph_data,
            "truncation_data": self.truncation_data,
            "metadata": self.metadata,
            "statistics": self.datamodule.statistics,
            "statistics_tendencies": self.datamodule.statistics_tendencies,
            "supporting_arrays": self.supporting_arrays,
        }

        model_task = get_class(self.config.training.model_task)
        model = model_task(**kwargs)  # GraphForecaster -> pl.LightningModule

        # Load the model weights
        if self.load_weights_only:
            # Sanify the checkpoint for transfer learning
            if self.config.training.transfer_learning:
                LOGGER.info("Loading weights with Transfer Learning from %s", self.last_checkpoint)
                model = transfer_learning_loading(model, self.last_checkpoint)
            else:
                LOGGER.info("Restoring only model weights from %s", self.last_checkpoint)
                # pop data_indices so that the data indices on the checkpoint do not get overwritten
                # by the data indices from the new config
                kwargs.pop("data_indices")
                model = model_task.load_from_checkpoint(self.last_checkpoint, **kwargs, strict=False)

            model.data_indices = self.data_indices
            # check data indices in original checkpoint and current data indices are the same
            self.data_indices.compare_variables(model._ckpt_model_name_to_index, self.data_indices.name_to_index)

        if hasattr(self.config.training, "submodules_to_freeze"):
            # Freeze the chosen model weights
            LOGGER.info("The following submodules will NOT be trained: %s", self.config.training.submodules_to_freeze)
            for submodule_name in self.config.training.submodules_to_freeze:
                freeze_submodule_by_name(model, submodule_name)
                LOGGER.info("%s frozen successfully.", submodule_name.upper())

        return model

    @rank_zero_only
    def _get_mlflow_run_id(self) -> str:
        run_id = self.mlflow_logger.run_id
        # for resumed runs or offline runs logging this can be useful
        LOGGER.info("Mlflow Run id: %s", run_id)
        return run_id

    @cached_property
    def run_id(self) -> str:
        """Unique identifier for the current run."""
        # When a run ID is provided
        if self.config.training.run_id and not self.config.training.fork_run_id:
            # Return the provided run ID - reuse run_id if resuming run
            return self.config.training.run_id

        # When a run ID has been created externally and we want to fork a run
        if self.config.training.run_id and self.config.training.fork_run_id:
            return self.config.training.run_id

        # When we rely on mlflow to create a new run ID
        if self.config.diagnostics.log.mlflow.enabled:
            # if using mlflow with a new run get the run_id from mlflow
            return self._get_mlflow_run_id()

        # When no run ID is provided a random one is generated
        import uuid

        return str(uuid.uuid4())

    @cached_property
    def wandb_logger(self) -> pl.loggers.WandbLogger:
        """WandB logger."""
        return get_wandb_logger(self.config, self.model)

    @cached_property
    def mlflow_logger(self) -> pl.loggers.MLFlowLogger:
        """Mlflow logger."""
        return get_mlflow_logger(self.config)

    @cached_property
    def tensorboard_logger(self) -> pl.loggers.TensorBoardLogger:
        """TensorBoard logger."""
        return get_tensorboard_logger(self.config)

    def _get_warm_start_checkpoint(self) -> Path | None:
        """Returns the warm start checkpoint path if specified."""
        warm_start_dir = getattr(self.config.hardware.paths, "warm_start", None)  # avoid breaking change
        warm_start_file = self.config.hardware.files.warm_start
        warm_start_path = None

        if warm_start_dir or warm_start_file:
            assert (
                warm_start_dir is not None
            ), f"Please configure config.hardware.paths.warm_start correctly, found: {warm_start_dir}"
            assert (
                warm_start_file is not None
            ), f"Please configure config.hardware.files.warm_start correctly, found: {warm_start_file}"
            warm_start_path = Path(warm_start_dir) / Path(warm_start_file)
            msg = "Warm start checkpoint not found: %s", warm_start_path
            assert Path.is_file(warm_start_path), msg
        return warm_start_path

    def _get_checkpoint_directory(self, fork_id: str) -> Path:
        """Returns the directory where checkpoints are stored."""
        return Path(self.config.hardware.paths.checkpoints.parent, fork_id or self.lineage_run) / "last.ckpt"

    @cached_property
    def last_checkpoint(self) -> Path | None:
        """Path to the last checkpoint."""
        if not self.start_from_checkpoint:
            return None

        fork_id = self.fork_run_server2server or self.config.training.fork_run_id
        checkpoint = self._get_warm_start_checkpoint() or self._get_checkpoint_directory(fork_id)

        # Check if the last checkpoint exists
        if checkpoint.exists():
            LOGGER.info("Resuming training from last checkpoint: %s", checkpoint)
            return checkpoint

        if rank_zero_only.rank == 0:
            msg = "Could not find last checkpoint: %s", checkpoint
            raise RuntimeError(msg)

        return None

    @cached_property
    def callbacks(self) -> list[pl.callbacks.Callback]:
        return get_callbacks(self.config.model_dump(by_alias=True))

    @cached_property
    def metadata(self) -> dict:
        """Metadata and provenance information."""
        return map_config_to_primitives(
            {
                "version": "1.0",
                "config": convert_to_omegaconf(self.config),
                "seed": self.initial_seed,
                "run_id": self.run_id,
                "dataset": self.datamodule.metadata,
                "data_indices": self.datamodule.data_indices,
                "provenance_training": gather_provenance_info(),
                "timestamp": datetime.datetime.now(tz=datetime.timezone.utc),
            },
        )

    @cached_property
    def supporting_arrays(self) -> dict:
        return self.datamodule.supporting_arrays

    @cached_property
    def loggers(self) -> list:
        loggers = []
        if self.config.diagnostics.log.wandb.enabled:
            LOGGER.info("W&B logger enabled")
            loggers.append(self.wandb_logger)
        if self.config.diagnostics.log.tensorboard.enabled:
            LOGGER.info("TensorBoard logger enabled")
            loggers.append(self.tensorboard_logger)
        if self.config.diagnostics.log.mlflow.enabled:
            LOGGER.info("MLFlow logger enabled")
            loggers.append(self.mlflow_logger)
        return loggers

    @cached_property
    def accelerator(self) -> str:
        assert self.config.hardware.accelerator in {
            "auto",
            "cpu",
            "gpu",
            "cuda",
            "tpu",
        }, f"Invalid accelerator ({self.config.hardware.accelerator}) in hardware config."

        if self.config.hardware.accelerator == "cpu":
            LOGGER.info("WARNING: Accelerator set to CPU, this should only be used for debugging.")
        return self.config.hardware.accelerator

    def _log_information(self) -> None:
        # Log number of variables (features)
        num_fc_features = len(self.datamodule.ds_train.data.variables) - len(self.config.data.forcing)
        LOGGER.info("Total number of prognostic variables: %d", num_fc_features)
        LOGGER.info("Total number of auxiliary variables: %d", len(self.config.data.forcing))

        # Log learning rate multiplier when running single-node, multi-GPU and/or multi-node
        total_number_of_model_instances = (
            self.config.hardware.num_nodes
            * self.config.hardware.num_gpus_per_node
            / self.config.hardware.num_gpus_per_model
        )

        LOGGER.info(
            "Total GPU count / model group size: %d - NB: the learning rate will be scaled by this factor!",
            total_number_of_model_instances,
        )
        LOGGER.info(
            "Effective learning rate: %.3e",
            int(total_number_of_model_instances) * self.config.training.lr.rate,
        )

        if self.config.training.max_epochs is not None and self.config.training.max_steps not in (None, -1):
            LOGGER.info(
                "Training limits: max_epochs=%d, max_steps=%d. "
                "Training will stop when either limit is reached first. "
                "Learning rate scheduler will run for %d steps.",
                self.config.training.max_epochs,
                self.config.training.max_steps,
                self.config.training.lr.iterations,
            )

    def _get_server2server_lineage(self) -> None:
        """Get the server2server lineage."""
        self.parent_run_server2server = None
        self.fork_run_server2server = None
        if self.config.diagnostics.log.mlflow.enabled:
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
            self.config.hardware.paths.checkpoints = Path(self.config.hardware.paths.checkpoints, self.lineage_run)
            self.config.hardware.paths.plots = Path(self.config.hardware.paths.plots, self.lineage_run)
        elif self.config.training.fork_run_id:
            # WHEN USING MANY NODES/GPUS
            self.lineage_run = self.parent_run_server2server or self.config.training.fork_run_id
            # Only rank non zero in the forked run will go here
            self.config.hardware.paths.checkpoints = Path(self.config.hardware.paths.checkpoints, self.lineage_run)

        LOGGER.info("Checkpoints path: %s", self.config.hardware.paths.checkpoints)
        LOGGER.info("Plots path: %s", self.config.hardware.paths.plots)

    @rank_zero_only
    def _check_dry_run(self) -> None:
        """Check if the run ID is dry, e.g. without a checkpoint.

        If the run ID is dry, the training will not be started.
        This is used to check the run can be restarted from the checkpoint.
        """
        self.dry_run = False
        if self.config.diagnostics.log.mlflow.enabled:
            # Check if the run ID is dry - e.g. without a checkpoint
            self.dry_run = (
                self.mlflow_logger._parent_dry_run and not Path(self.config.hardware.paths.checkpoints).is_dir()
            )
            self.start_from_checkpoint = (
                False if (self.dry_run and not bool(self.config.training.fork_run_id)) else self.start_from_checkpoint
            )
            LOGGER.info("Dry run: %s", self.dry_run)

    @cached_property
    def strategy(self) -> Any:
        return instantiate(
            convert_to_omegaconf(self.config).training.strategy,
            static_graph=not self.config.training.accum_grad_batches > 1,
        )

    def train(self) -> None:
        """Training entry point."""
        LOGGER.debug("Setting up trainer..")

        trainer = pl.Trainer(
            accelerator=self.accelerator,
            callbacks=self.callbacks,
            deterministic=self.config.training.deterministic,
            detect_anomaly=self.config.diagnostics.debug.anomaly_detection,
            strategy=self.strategy,
            devices=self.config.hardware.num_gpus_per_node,
            num_nodes=self.config.hardware.num_nodes,
            precision=self.config.training.precision,
            max_epochs=self.config.training.max_epochs,
            max_steps=self.config.training.max_steps or -1,
            logger=self.loggers,
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

        LOGGER.debug("Starting training..")

        trainer.fit(
            self.model,
            datamodule=self.datamodule,
            ckpt_path=None if (self.load_weights_only) else self.last_checkpoint,
        )

        if self.config.diagnostics.print_memory_summary:
            LOGGER.info("memory summary: %s", torch.cuda.memory_summary())

        LOGGER.debug("---- DONE. ----")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig) -> None:
    AnemoiTrainer(config).train()


if __name__ == "__main__":
    main()

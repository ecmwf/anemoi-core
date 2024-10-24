# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import datetime
import logging
import os
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from anemoi.utils.provenance import gather_provenance_info
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
from anemoi.training.diagnostics.callbacks import get_callbacks
from anemoi.training.diagnostics.logger import get_mlflow_logger
from anemoi.training.diagnostics.logger import get_tensorboard_logger
from anemoi.training.diagnostics.logger import get_wandb_logger
from anemoi.training.distributed.strategy import DDPGroupStrategy
from anemoi.training.train.forecaster import GraphForecaster
from anemoi.training.utils.jsonify import map_config_to_primitives
from anemoi.training.utils.seeding import get_base_seed

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)


class AnemoiTrainer:
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
        OmegaConf.resolve(config)
        self.config = config

        # Default to not warm-starting from a checkpoint
        self.start_from_checkpoint = bool(self.config.training.run_id) or bool(self.config.training.fork_run_id)
        self.load_weights_only = config.training.load_weights_only
        self.parent_uuid = None

        self.config.training.run_id = self.run_id
        LOGGER.info("Run id: %s", self.config.training.run_id)

        # Get the server2server lineage
        self._get_server2server_lineage()

        # Update paths to contain the run ID
        self._update_paths()

        self._log_information()

    @cached_property
    def datamodule(self) -> AnemoiDatasetsDataModule:
        """DataModule instance and DataSets."""
        datamodule = AnemoiDatasetsDataModule(self.config)
        self.config.data.num_features = len(datamodule.ds_train.data.variables)
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
        LOGGER.debug(
            "Initial seed: Rank %d, initial seed %d, running with random seed: %d",
            int(os.environ.get("SLURM_PROCID", "0")),
            initial_seed,
            rnd_seed,
        )
        return initial_seed

    @cached_property
    def graph_data(self) -> HeteroData:
        """Graph data.

        Creates the graph in all workers.
        """
        graph_filename = Path(
            self.config.hardware.paths.graph,
            self.config.hardware.files.graph,
        )

        if graph_filename.exists() and not self.config.graph.overwrite:
            LOGGER.info("Loading graph data from %s", graph_filename)
            return torch.load(graph_filename)

        from anemoi.graphs.create import GraphCreator

        return GraphCreator(config=self.config.graph).create(
            save_path=graph_filename,
            overwrite=self.config.graph.overwrite,
        )

    @cached_property
    def model(self) -> GraphForecaster:
        """Provide the model instance."""
        kwargs = {
            "config": self.config,
            "data_indices": self.data_indices,
            "graph_data": self.graph_data,
            "metadata": self.metadata,
            "statistics": self.datamodule.statistics,
        }
        if self.load_weights_only:
            LOGGER.info("Restoring only model weights from %s", self.last_checkpoint)
            return GraphForecaster.load_from_checkpoint(self.last_checkpoint, **kwargs)
        return GraphForecaster(**kwargs)

    @rank_zero_only
    def _get_mlflow_run_id(self) -> str:
        run_id = self.mlflow_logger.run_id
        # for resumed runs or offline runs logging this can be useful
        LOGGER.info("Mlflow Run id: %s", run_id)
        return run_id

    @cached_property
    def run_id(self) -> str:
        """Unique identifier for the current run."""
        if self.config.training.run_id and not self.config.training.fork_run_id:
            # Return the provided run ID - reuse run_id if resuming run
            return self.config.training.run_id

        if self.config.diagnostics.log.mlflow.enabled:
            # if using mlflow with a new run get the run_id from mlflow
            return self._get_mlflow_run_id()

        # Generate a random UUID
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

    @cached_property
    def last_checkpoint(self) -> str | None:
        """Path to the last checkpoint."""
        if not self.start_from_checkpoint:
            return None

        fork_id = self.fork_run_server2server or self.config.training.fork_run_id
        checkpoint = Path(
            self.config.hardware.paths.checkpoints.parent,
            fork_id or self.lineage_run,
            self.config.hardware.files.warm_start or "last.ckpt",
        )
        # Check if the last checkpoint exists
        if Path(checkpoint).exists():
            LOGGER.info("Resuming training from last checkpoint: %s", checkpoint)
            return checkpoint

        if rank_zero_only.rank == 0:
            msg = "Could not find last checkpoint: %s", checkpoint
            raise RuntimeError(msg)

        return None

    @cached_property
    def callbacks(self) -> list[pl.callbacks.Callback]:
        return get_callbacks(self.config)

    @cached_property
    def metadata(self) -> dict:
        """Metadata and provenance information."""
        return map_config_to_primitives(
            {
                "version": "1.0",
                "config": self.config,
                "seed": self.initial_seed,
                "run_id": self.run_id,
                "dataset": self.datamodule.metadata,
                "data_indices": self.datamodule.data_indices,
                "provenance_training": gather_provenance_info(),
                "timestamp": datetime.datetime.now(tz=datetime.timezone.utc),
            },
        )

    @cached_property
    def profiler(self) -> PyTorchProfiler | None:
        """Returns a pytorch profiler object, if profiling is enabled."""
        if self.config.diagnostics.profiler:
            assert (
                self.config.diagnostics.log.tensorboard.enabled
            ), "Tensorboard logging must be enabled when profiling! Check your job config."
            return PyTorchProfiler(
                dirpath=self.config.hardware.paths.logs.tensorboard,
                filename="anemoi-profiler",
                export_to_chrome=False,
                # profiler-specific keywords
                activities=[
                    # torch.profiler.ProfilerActivity.CPU,  # this is memory-hungry
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    dir_name=self.config.hardware.paths.logs.tensorboard,
                ),
                profile_memory=True,
                record_shapes=True,
                with_stack=True,
            )
        return None

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
        LOGGER.debug("Total number of prognostic variables: %d", num_fc_features)
        LOGGER.debug("Total number of auxiliary variables: %d", len(self.config.data.forcing))

        # Log learning rate multiplier when running single-node, multi-GPU and/or multi-node
        total_number_of_model_instances = (
            self.config.hardware.num_nodes
            * self.config.hardware.num_gpus_per_node
            / self.config.hardware.num_gpus_per_model
        )
        LOGGER.debug(
            "Total GPU count / model group size: %d - NB: the learning rate will be scaled by this factor!",
            total_number_of_model_instances,
        )
        LOGGER.debug("Effective learning rate: %.3e", total_number_of_model_instances * self.config.training.lr.rate)
        LOGGER.debug("Rollout window length: %d", self.config.training.rollout.start)

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

    @cached_property
    def strategy(self) -> DDPGroupStrategy:
        """Training strategy."""
        return DDPGroupStrategy(
            self.config.hardware.num_gpus_per_model,
            static_graph=not self.config.training.accum_grad_batches > 1,
        )

    def train(self) -> None:
        """Training entry point."""
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
            logger=self.loggers,
            log_every_n_steps=self.config.diagnostics.log.interval,
            # run a fixed no of batches per epoch (helpful when debugging)
            limit_train_batches=self.config.dataloader.limit_batches.training,
            limit_val_batches=self.config.dataloader.limit_batches.validation,
            num_sanity_val_steps=4,
            accumulate_grad_batches=self.config.training.accum_grad_batches,
            gradient_clip_val=self.config.training.gradient_clip.val,
            gradient_clip_algorithm=self.config.training.gradient_clip.algorithm,
            # we have our own DDP-compliant sampler logic baked into the dataset
            use_distributed_sampler=False,
            profiler=self.profiler,
            enable_progress_bar=self.config.diagnostics.enable_progress_bar,
        )

        trainer.fit(
            self.model,
            datamodule=self.datamodule,
            ckpt_path=None if self.load_weights_only else self.last_checkpoint,
        )

        if self.config.diagnostics.print_memory_summary:
            LOGGER.debug("memory summary: %s", torch.cuda.memory_summary())

        LOGGER.debug("---- DONE. ----")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig) -> None:
    AnemoiTrainer(config).train()


if __name__ == "__main__":
    main()

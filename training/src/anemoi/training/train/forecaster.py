# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections import defaultdict
from collections.abc import Generator
from collections.abc import Mapping
from typing import Optional
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.interface import AnemoiModelInterface
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from timm.scheduler import CosineLRScheduler
from torch.distributed.distributed_c10d import ProcessGroup
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.training.losses.utils import grad_scaler
from anemoi.training.losses.weightedloss import BaseWeightedLoss
from anemoi.training.utils.jsonify import map_config_to_primitives
from anemoi.training.utils.masks import Boolean1DMask
from anemoi.training.utils.masks import NoOutputMask

LOGGER = logging.getLogger(__name__)


class GraphForecaster(pl.LightningModule):
    """Graph neural network forecaster for PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network forecaster.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        graph_data : HeteroData
            Graph object
        statistics : dict
            Statistics of the training data
        data_indices : IndexCollection
            Indices of the training data,
        metadata : dict
            Provenance information
        supporting_arrays : dict
            Supporting NumPy arrays to store in the checkpoint

        """
        super().__init__()

        graph_data = graph_data.to(self.device)

        if config.model.get("output_mask", None) is not None:
            self.output_mask = Boolean1DMask(graph_data[config.graph.data][config.model.output_mask])
        else:
            self.output_mask = NoOutputMask()

        self.model = AnemoiModelInterface(
            statistics=statistics,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays | self.output_mask.supporting_arrays,
            graph_data=graph_data,
            config=DotDict(map_config_to_primitives(OmegaConf.to_container(config, resolve=True))),
        )
        self.config = config
        self.data_indices = data_indices

        self.save_hyperparameters()

        self.latlons_data = graph_data[config.graph.data].x
        self.node_weights = self.get_node_weights(config, graph_data)
        self.node_weights = self.output_mask.apply(self.node_weights, dim=0, fill_value=0.0)

        self.logger_enabled = config.diagnostics.log.wandb.enabled or config.diagnostics.log.mlflow.enabled

        variable_scaling = self.get_variable_scaling(config, data_indices)

        self.internal_metric_ranges, self.val_metric_ranges = self.get_val_metric_ranges(config, data_indices)

        # Check if the model is a stretched grid
        if graph_data["hidden"].node_type == "StretchedTriNodes":
            mask_name = config.graph.nodes.hidden.node_builder.mask_attr_name
            limited_area_mask = graph_data[config.graph.data][mask_name].squeeze().bool()
        else:
            limited_area_mask = torch.ones((1,))

        # Kwargs to pass to the loss function
        loss_kwargs = {"node_weights": self.node_weights}
        # Scalars to include in the loss function, must be of form (dim, scalar)
        # Use -1 for the variable dimension, -2 for the latlon dimension
        # Add mask multiplying NaN locations with zero. At this stage at [[1]].
        # Filled after first application of preprocessor. dimension=[-2, -1] (latlon, n_outputs).
        self.scalars = {
            "variable": (-1, variable_scaling),
            "loss_weights_mask": ((-2, -1), torch.ones((1, 1))),
            "limited_area_mask": (2, limited_area_mask),
        }
        self.updated_loss_mask = False

        self.loss = self.get_loss_function(config.training.training_loss, scalars=self.scalars, **loss_kwargs)

        assert isinstance(self.loss, BaseWeightedLoss) and not isinstance(
            self.loss,
            torch.nn.ModuleList,
        ), f"Loss function must be a `BaseWeightedLoss`, not a {type(self.loss).__name__!r}"

        self.metrics = self.get_loss_function(config.training.validation_metrics, scalars=self.scalars, **loss_kwargs)
        if not isinstance(self.metrics, torch.nn.ModuleList):
            self.metrics = torch.nn.ModuleList([self.metrics])

        if config.training.loss_gradient_scaling:
            self.loss.register_full_backward_hook(grad_scaler, prepend=False)

        self.multi_step = config.training.multistep_input
        self.lr = (
            config.hardware.num_nodes
            * config.hardware.num_gpus_per_node
            * config.training.lr.rate
            / config.hardware.num_gpus_per_model
        )
        self.warmup_t = getattr(config.training.lr, "warmup_t", 1000)
        self.lr_iterations = config.training.lr.iterations
        self.lr_min = config.training.lr.min
        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

        self.use_zero_optimizer = config.training.zero_optimizer

        self.model_comm_group = None
        self.reader_groups = None

        LOGGER.debug("Rollout window length: %d", self.rollout)
        LOGGER.debug("Rollout increase every : %d epochs", self.rollout_epoch_increment)
        LOGGER.debug("Rollout max : %d", self.rollout_max)
        LOGGER.debug("Multistep: %d", self.multi_step)

        # lazy init model and reader group info, will be set by the DDPGroupStrategy:
        self.model_comm_group_id = 0
        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1

        self.reader_group_id = 0
        self.reader_group_rank = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, self.model_comm_group)

    # Future import breaks other type hints TODO Harrison Cook
    @staticmethod
    def get_loss_function(
        config: DictConfig,
        scalars: Union[dict[str, tuple[Union[int, tuple[int, ...], torch.Tensor]]], None] = None,  # noqa: FA100
        **kwargs,
    ) -> Union[BaseWeightedLoss, torch.nn.ModuleList]:  # noqa: FA100
        """Get loss functions from config.

        Can be ModuleList if multiple losses are specified.

        Parameters
        ----------
        config : DictConfig
            Loss function configuration, should include `scalars` if scalars are to be added to the loss function.
        scalars : Union[dict[str, tuple[Union[int, tuple[int, ...], torch.Tensor]]], None], optional
            Scalars which can be added to the loss function. Defaults to None., by default None
            If a scalar is to be added to the loss, ensure it is in `scalars` in the loss config
            E.g.
                If `scalars: ['variable']` is set in the config, and `variable` in `scalars`
                `variable` will be added to the scalar of the loss function.
        kwargs : Any
            Additional arguments to pass to the loss function

        Returns
        -------
        Union[BaseWeightedLoss, torch.nn.ModuleList]
            Loss function, or list of metrics

        Raises
        ------
        TypeError
            If not a subclass of `BaseWeightedLoss`
        ValueError
            If scalar is not found in valid scalars
        """
        config_container = OmegaConf.to_container(config, resolve=False)
        if isinstance(config_container, list):
            return torch.nn.ModuleList(
                [
                    GraphForecaster.get_loss_function(
                        OmegaConf.create(loss_config),
                        scalars=scalars,
                        **kwargs,
                    )
                    for loss_config in config
                ],
            )

        loss_config = OmegaConf.to_container(config, resolve=True)
        scalars_to_include = loss_config.pop("scalars", [])

        # Instantiate the loss function with the loss_init_config
        loss_function = instantiate(loss_config, **kwargs)

        if not isinstance(loss_function, BaseWeightedLoss):
            error_msg = f"Loss must be a subclass of 'BaseWeightedLoss', not {type(loss_function)}"
            raise TypeError(error_msg)

        for key in scalars_to_include:
            if key not in scalars or []:
                error_msg = f"Scalar {key!r} not found in valid scalars: {list(scalars.keys())}"
                raise ValueError(error_msg)
            loss_function.add_scalar(*scalars[key], name=key)

        return loss_function

    def training_weights_for_imputed_variables(
        self,
        batch: torch.Tensor,
    ) -> None:
        """Update the loss weights mask for imputed variables."""
        if "loss_weights_mask" in self.loss.scalar:
            loss_weights_mask = torch.ones((1, 1), device=batch.device)
            found_loss_mask_training = False
            # iterate over all pre-processors and check if they have a loss_mask_training attribute
            for pre_processor in self.model.pre_processors.processors.values():
                if hasattr(pre_processor, "loss_mask_training"):
                    loss_weights_mask = loss_weights_mask * pre_processor.loss_mask_training
                    found_loss_mask_training = True
                # if transform_loss_mask function exists for preprocessor apply it
                if hasattr(pre_processor, "transform_loss_mask") and found_loss_mask_training:
                    loss_weights_mask = pre_processor.transform_loss_mask(loss_weights_mask)
            # update scaler with loss_weights_mask retrieved from preprocessors
            self.loss.update_scalar(scalar=loss_weights_mask.cpu(), name="loss_weights_mask")
            self.scalars["loss_weights_mask"] = ((-2, -1), loss_weights_mask.cpu())

        self.updated_loss_mask = True

    @staticmethod
    def get_val_metric_ranges(config: DictConfig, data_indices: IndexCollection) -> tuple[dict, dict]:

        metric_ranges = defaultdict(list)
        metric_ranges_validation = defaultdict(list)

        for key, idx in data_indices.internal_model.output.name_to_index.items():
            split = key.split("_")
            if len(split) > 1 and split[-1].isdigit():
                # Group metrics for pressure levels (e.g., Q, T, U, V, etc.)
                metric_ranges[f"pl_{split[0]}"].append(idx)
            else:
                metric_ranges[f"sfc_{key}"].append(idx)

            # Specific metrics from hydra to log in logger
            if key in config.training.metrics:
                metric_ranges[key] = [idx]

        # Add the full list of output indices
        metric_ranges["all"] = data_indices.internal_model.output.full.tolist()

        # metric for validation, after postprocessing
        for key, idx in data_indices.model.output.name_to_index.items():
            # Split pressure levels on "_" separator
            split = key.split("_")
            if len(split) > 1 and split[1].isdigit():
                # Create grouped metrics for pressure levels (e.g. Q, T, U, V, etc.) for logger
                metric_ranges_validation[f"pl_{split[0]}"].append(idx)
            else:
                metric_ranges_validation[f"sfc_{key}"].append(idx)
            # Create specific metrics from hydra to log in logger
            if key in config.training.metrics:
                metric_ranges_validation[key] = [idx]

        # Add the full list of output indices
        metric_ranges_validation["all"] = data_indices.model.output.full.tolist()

        return metric_ranges, metric_ranges_validation

    @staticmethod
    def get_variable_scaling(
        config: DictConfig,
        data_indices: IndexCollection,
    ) -> torch.Tensor:
        variable_loss_scaling = (
            np.ones((len(data_indices.internal_data.output.full),), dtype=np.float32)
            * config.training.variable_loss_scaling.default
        )
        pressure_level = instantiate(config.training.pressure_level_scaler)

        LOGGER.info(
            "Pressure level scaling: use scaler %s with slope %.4f and minimum %.2f",
            type(pressure_level).__name__,
            pressure_level.slope,
            pressure_level.minimum,
        )

        for key, idx in data_indices.internal_model.output.name_to_index.items():
            split = key.split("_")
            if len(split) > 1 and split[-1].isdigit():
                # Apply pressure level scaling
                if split[0] in config.training.variable_loss_scaling.pl:
                    variable_loss_scaling[idx] = config.training.variable_loss_scaling.pl[
                        split[0]
                    ] * pressure_level.scaler(
                        int(split[-1]),
                    )
                else:
                    LOGGER.debug("Parameter %s was not scaled.", key)
            else:
                # Apply surface variable scaling
                if key in config.training.variable_loss_scaling.sfc:
                    variable_loss_scaling[idx] = config.training.variable_loss_scaling.sfc[key]
                else:
                    LOGGER.debug("Parameter %s was not scaled.", key)

        return torch.from_numpy(variable_loss_scaling)

    @staticmethod
    def get_node_weights(config: DictConfig, graph_data: HeteroData) -> torch.Tensor:
        node_weighting = instantiate(config.training.node_loss_weights)
        return node_weighting.weights(graph_data)

    def set_model_comm_group(
        self,
        model_comm_group: ProcessGroup,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        model_comm_group_size: int,
    ) -> None:
        self.model_comm_group = model_comm_group
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.model_comm_group_size = model_comm_group_size

    def set_reader_groups(
        self,
        reader_groups: list[ProcessGroup],
        reader_group_id: int,
        reader_group_rank: int,
        reader_group_size: int,
    ) -> None:
        self.reader_groups = reader_groups
        self.reader_group_id = reader_group_id
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

    def advance_input(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        batch: torch.Tensor,
        rollout_step: int,
    ) -> torch.Tensor:
        x = x.roll(-1, dims=1)

        # Get prognostic variables
        x[:, -1, :, :, self.data_indices.internal_model.input.prognostic] = y_pred[
            ...,
            self.data_indices.internal_model.output.prognostic,
        ]

        x[:, -1] = self.output_mask.rollout_boundary(x[:, -1], batch[:, -1], self.data_indices)

        # get new "constants" needed for time-varying fields
        x[:, -1, :, :, self.data_indices.internal_model.input.forcing] = batch[
            :,
            self.multi_step + rollout_step,
            :,
            :,
            self.data_indices.internal_data.input.forcing,
        ]
        return x

    def rollout_step(
        self,
        batch: torch.Tensor,
        rollout: Optional[int] = None,  # noqa: FA100
        training_mode: bool = True,
        validation_mode: bool = False,
    ) -> Generator[tuple[Union[torch.Tensor, None], dict, list], None, None]:  # noqa: FA100
        """Rollout step for the forecaster.

        Will run pre_processors on batch, but not post_processors on predictions.

        Parameters
        ----------
        batch : torch.Tensor
            Batch to use for rollout
        rollout : Optional[int], optional
            Number of times to rollout for, by default None
            If None, will use self.rollout
        training_mode : bool, optional
            Whether in training mode and to calculate the loss, by default True
            If False, loss will be None
        validation_mode : bool, optional
            Whether in validation mode, and to calculate validation metrics, by default False
            If False, metrics will be empty

        Yields
        ------
        Generator[tuple[Union[torch.Tensor, None], dict, list], None, None]
            Loss value, metrics, and predictions (per step)

        Returns
        -------
        None
            None
        """
        # for validation not normalized in-place because remappers cannot be applied in-place
        batch = self.model.pre_processors(batch, in_place=not validation_mode)

        if not self.updated_loss_mask:
            # update loss scalar after first application and initialization of preprocessors
            self.training_weights_for_imputed_variables(batch)

        # start rollout of preprocessed batch
        x = batch[
            :,
            0 : self.multi_step,
            ...,
            self.data_indices.internal_data.input.full,
        ]  # (bs, multi_step, latlon, nvar)
        msg = (
            "Batch length not sufficient for requested multi_step length!"
            f", {batch.shape[1]} !>= {rollout + self.multi_step}"
        )
        assert batch.shape[1] >= rollout + self.multi_step, msg

        for rollout_step in range(rollout or self.rollout):
            # prediction at rollout step rollout_step, shape = (bs, latlon, nvar)
            y_pred = self(x)

            y = batch[:, self.multi_step + rollout_step, ..., self.data_indices.internal_data.output.full]
            # y includes the auxiliary variables, so we must leave those out when computing the loss
            loss = checkpoint(self.loss, y_pred, y, use_reentrant=False) if training_mode else None

            x = self.advance_input(x, y_pred, batch, rollout_step)

            metrics_next = {}
            if validation_mode:
                metrics_next = self.calculate_val_metrics(
                    y_pred,
                    y,
                    rollout_step,
                )
            yield loss, metrics_next, y_pred

    def _step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        del batch_idx
        batch = self.allgather_batch(batch)

        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        metrics = {}
        y_preds = []

        for loss_next, metrics_next, y_preds_next in self.rollout_step(
            batch,
            rollout=self.rollout,
            training_mode=True,
            validation_mode=validation_mode,
        ):
            loss += loss_next
            metrics.update(metrics_next)
            y_preds.extend(y_preds_next)

        loss *= 1.0 / self.rollout
        return loss, metrics, y_preds

    def allgather_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Allgather the batch-shards across the reader group.

        Parameters
        ----------
        batch : torch.Tensor
            Batch-shard of current reader rank

        Returns
        -------
        torch.Tensor
            Allgathered (full) batch
        """
        grid_size = len(self.latlons_data)  # number of points

        if grid_size == batch.shape[-2]:
            return batch  # already have the full grid

        grid_shard_size = grid_size // self.reader_group_size
        last_grid_shard_size = grid_size - (grid_shard_size * (self.reader_group_size - 1))

        # prepare tensor list with correct shapes for all_gather
        shard_shape = list(batch.shape)
        shard_shape[-2] = grid_shard_size
        last_shard_shape = list(batch.shape)
        last_shard_shape[-2] = last_grid_shard_size

        tensor_list = [torch.empty(tuple(shard_shape), device=self.device) for _ in range(self.reader_group_size - 1)]
        tensor_list.append(torch.empty(last_shard_shape, device=self.device))

        torch.distributed.all_gather(
            tensor_list,
            batch,
            group=self.reader_groups[self.reader_group_id],
        )

        return torch.cat(tensor_list, dim=-2)

    def calculate_val_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        rollout_step: int,
    ) -> tuple[dict, list[torch.Tensor]]:
        """Calculate metrics on the validation output.

        Parameters
        ----------
            y_pred: torch.Tensor
                Predicted ensemble
            y: torch.Tensor
                Ground truth (target).
            rollout_step: int
                Rollout step

        Returns
        -------
            val_metrics, preds:
                validation metrics and predictions
        """
        metrics = {}
        y_postprocessed = self.model.post_processors(y, in_place=False)
        y_pred_postprocessed = self.model.post_processors(y_pred, in_place=False)

        for metric in self.metrics:
            metric_name = getattr(metric, "name", metric.__class__.__name__.lower())

            if not isinstance(metric, BaseWeightedLoss):
                # If not a weighted loss, we cannot feature scale, so call normally
                metrics[f"{metric_name}/{rollout_step + 1}"] = metric(
                    y_pred_postprocessed,
                    y_postprocessed,
                )
                continue

            for mkey, indices in self.val_metric_ranges.items():
                if "scale_validation_metrics" in self.config.training and (
                    mkey in self.config.training.scale_validation_metrics.metrics
                    or "*" in self.config.training.scale_validation_metrics.metrics
                ):
                    with metric.scalar.freeze_state():
                        for key in self.config.training.scale_validation_metrics.scalars_to_apply:
                            metric.add_scalar(*self.scalars[key], name=key)

                        # Use internal model space indices
                        internal_model_indices = self.internal_metric_ranges[mkey]

                        metrics[f"{metric_name}/{mkey}/{rollout_step + 1}"] = metric(
                            y_pred,
                            y,
                            scalar_indices=[..., internal_model_indices],
                        )
                else:
                    if -1 in metric.scalar:
                        exception_msg = (
                            "Validation metrics cannot be scaled over the variable dimension"
                            " in the post processed space. Please specify them in the config"
                            " at `scale_validation_metrics`."
                        )
                        raise ValueError(exception_msg)

                    metrics[f"{metric_name}/{mkey}/{rollout_step + 1}"] = metric(
                        y_pred_postprocessed,
                        y_postprocessed,
                        scalar_indices=[..., indices],
                    )

        return metrics

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        train_loss, _, _ = self._step(batch, batch_idx)
        self.log(
            f"train_{getattr(self.loss, 'name', self.loss.__class__.__name__.lower())}",
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.shape[0],
            sync_dist=True,
        )
        self.log(
            "rollout",
            float(self.rollout),
            on_step=True,
            logger=self.logger_enabled,
            rank_zero_only=True,
            sync_dist=False,
        )
        return train_loss

    def lr_scheduler_step(self, scheduler: CosineLRScheduler, metric: None = None) -> None:
        """Step the learning rate scheduler by Pytorch Lightning.

        Parameters
        ----------
        scheduler : CosineLRScheduler
            Learning rate scheduler object.
        metric : Optional[Any]
            Metric object for e.g. ReduceLRonPlateau. Default is None.

        """
        del metric
        scheduler.step(epoch=self.trainer.global_step)

    def on_train_epoch_end(self) -> None:
        if self.rollout_epoch_increment > 0 and self.current_epoch % self.rollout_epoch_increment == 0:
            self.rollout += 1
            LOGGER.debug("Rollout window length: %d", self.rollout)
        self.rollout = min(self.rollout, self.rollout_max)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """
        Calculate the loss over a validation batch using the training loss function.

        Parameters
        ----------
        batch : torch.Tensor
            Validation batch
        batch_idx : int
            Batch inces

        Returns
        -------
        None
        """
        with torch.no_grad():
            val_loss, metrics, y_preds = self._step(batch, batch_idx, validation_mode=True)

        self.log(
            f"val_{getattr(self.loss, 'name', self.loss.__class__.__name__.lower())}",
            val_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.shape[0],
            sync_dist=True,
        )

        for mname, mvalue in metrics.items():
            self.log(
                "val_" + mname,
                mvalue,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=self.logger_enabled,
                batch_size=batch.shape[0],
                sync_dist=True,
            )

        return val_loss, y_preds

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict]]:
        if self.use_zero_optimizer:
            optimizer = ZeroRedundancyOptimizer(
                self.trainer.model.parameters(),
                optimizer_class=torch.optim.AdamW,
                betas=(0.9, 0.95),
                lr=self.lr,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.trainer.model.parameters(),
                betas=(0.9, 0.95),
                lr=self.lr,
            )  # , fused=True)

        scheduler = CosineLRScheduler(
            optimizer,
            lr_min=self.lr_min,
            t_initial=self.lr_iterations,
            warmup_t=self.warmup_t,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

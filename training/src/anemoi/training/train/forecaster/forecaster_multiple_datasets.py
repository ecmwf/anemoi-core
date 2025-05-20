# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch
from timm.scheduler import CosineLRScheduler
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.checkpoint import checkpoint

from anemoi.models.interface import AnemoiModelInterface
from anemoi.training.data.utils import DataHandlerName
from anemoi.training.data.utils import RecordProviderName
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.scaler_tensor import grad_scaler
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.schemas.base_schema import convert_to_omegaconf
from anemoi.training.utils.enums import TensorDim

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Mapping

    from torch.distributed.distributed_c10d import ProcessGroup
    from torch_geometric.data import HeteroData


LOGGER = logging.getLogger(__name__)


class GraphForecasterMultiDataset(pl.LightningModule):
    """Graph neural network forecaster for PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: BaseSchema,
        sample_provider: SampleProvider,
        graph_data: HeteroData,
        metadata: dict,
    ) -> None:
        """Initialize graph neural network forecaster.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        graph_data : HeteroData
            Graph object
        metadata : dict
            Provenance information

        """
        super().__init__()

        graph_data = graph_data.to(self.device)

        # TODO: Handle supporting arrays for multiple output masks (multiple outputs)
        # (It is handled in the loss function, but not the version here that is sent to model for supporting_arrays)
        # self.output_mask = instantiate(config.model_dump(by_alias=True).model.output_mask, graph_data=graph_data)

        self.model = AnemoiModelInterface(
            # data_indices=data_indices,
            metadata=metadata,
            sample_provider=sample_provider,
            graph_data=graph_data,
            config=convert_to_omegaconf(config),
        )
        self.config = config
        # self.model_data_indices = {self.datasets[0]: self.model.model.data_indices}  # TODO: generalize

        self.save_hyperparameters()

        self.latlons_data = graph_data[config.graph.data].x  # TODO: Generalize, link graph key to DataHandler key

        self.logger_enabled = config.diagnostics.log.wandb.enabled or config.diagnostics.log.mlflow.enabled

        # Instantiate all scalers with the training configuration
        # self.scalers, self.delayed_scaler_builders = create_scalers(
        #    config.model_dump(by_alias=True).training.scalers,
        #    group_config=config.model_dump(by_alias=True).training.variable_groups,
        # data_indices=data_indices,
        #    graph_data=graph_data,
        #    statistics=statistics,
        #    statistics_tendencies=statistics_tendencies,
        #    metadata_variables=metadata["dataset"].get("variables_metadata"),
        #    output_mask=self.output_mask,
        # )

        # self.internal_metric_ranges, self.val_metric_ranges = get_metric_ranges(
        #    config,
        #    data_indices,
        #    metadata["dataset"].get("variables_metadata"),
        # )

        self.loss = get_loss_function(
            config.model_dump(by_alias=True).training.training_loss,
            scalers={},  # self.scalers,
            #    data_indices=self.data_indices,
        )
        # print_variable_scaling(self.loss, data_indices)

        self.metrics = torch.nn.ModuleDict({})
        #    {
        #        metric_name: get_loss_function(val_metric_config, scalers=self.scalers, data_indices=self.data_indices)
        #        for metric_name, val_metric_config in config.model_dump(
        #            by_alias=True,
        #        ).training.validation_metrics.items()
        #    },
        # )
        if config.training.loss_gradient_scaling:
            self.loss.register_full_backward_hook(grad_scaler, prepend=False)

        self.is_first_step = True
        self.multi_step = config.training.multistep_input
        self.lr = (
            config.hardware.num_nodes
            * config.hardware.num_gpus_per_node
            * config.training.lr.rate
            / config.hardware.num_gpus_per_model
        )
        self.lr_iterations = config.training.lr.iterations
        self.lr_warmup = config.training.lr.warmup
        self.lr_min = config.training.lr.min
        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

        self.optimizer_settings = config.training.optimizer

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
        return self.model(x, model_comm_group=self.model_comm_group)

    def define_delayed_scalers(self) -> None:
        """Update delayed scalers such as the loss weights mask for imputed variables."""
        for name, scaler_builder in self.delayed_scaler_builders.items():
            self.scalers[name] = scaler_builder.get_delayed_scaling(model=self.model)
            self.loss.update_scaler(scaler=self.scalers[name][1], name=name)

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

        x[:, -1] = self.output_mask.rollout_boundary(
            x[:, -1],
            batch[:, self.multi_step + rollout_step],
            self.data_indices,
        )

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
        batch: dict[RecordProviderName, dict[DataHandlerName, torch.Tensor]],
        rollout: int | None = None,
        training_mode: bool = True,
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, list], None, None]:
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

        """
        # for validation not normalized in-place because remappers cannot be applied in-place
        batch["input"] = {"era5": self.model.input_pre_processors(batch["input"]["era5"], in_place=not validation_mode)}
        batch["target"] = {
            "era5": self.model.target_pre_processors(batch["target"]["era5"], in_place=not validation_mode),
        }

        # Delayed scalers need to be initialized after the pre-processors once
        if False:  # self.is_first_step:
            self.define_delayed_scalers()
            self.is_first_step = False

        assert rollout in [1, None], "Rollout NOT IMPLEMENTED yet"
        for rollout_step in range(rollout or self.rollout):
            # prediction at rollout step rollout_step, shape = (bs, latlon, nvar)
            y_pred = self(batch["input"])

            # y includes the auxiliary variables, so we must leave those out when computing the loss
            loss = (
                checkpoint(self.loss, y_pred["era5"], batch["target"]["era5"], use_reentrant=False)
                if training_mode
                else None
            )

            # x = self.advance_input(x, y_pred, batch, rollout_step)

            metrics_next = {}
            if validation_mode:
                metrics_next = self.calculate_val_metrics(y_pred, batch["target"], rollout_step)
            yield loss, metrics_next, y_pred

    def _step(
        self,
        batch: dict[RecordProviderName, dict[DataHandlerName, torch.Tensor]],
        batch_idx: int,
        validation_mode: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Mapping[str, torch.Tensor]], dict[str, torch.Tensor]]:
        del batch_idx
        # batch = self.allgather_batch(batch)

        loss = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=False)
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
        y_postprocessed = {"era5": self.model.target_post_processors(y["era5"], in_place=False)}
        y_pred_postprocessed = {"era5": self.model.target_post_processors(y_pred["era5"], in_place=False)}

        for metric_name, metric in self.metrics.items():
            if not isinstance(metric, BaseLoss):
                # If not a loss, we cannot feature scale, so call normally
                metrics[f"{metric_name}_metric/{rollout_step + 1}"] = metric(y_pred_postprocessed, y_postprocessed)
                continue

            for mkey, indices in self.val_metric_ranges.items():
                metric_step_name = f"{metric_name}_metric/{mkey}/{rollout_step + 1}"
                if len(metric.scaler.subset_by_dim(TensorDim.VARIABLE.value)):
                    exception_msg = (
                        "Validation metrics cannot be scaled over the variable dimension"
                        " in the post processed space."
                    )
                    raise ValueError(exception_msg)

                metrics[metric_step_name] = metric(y_pred_postprocessed, y_postprocessed, scaler_indices=[..., indices])

        return metrics

    def training_step(
        self,
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],  # C - G - S - B - T
        batch_idx: int,
    ) -> torch.Tensor:


        class Spec:
            def __init__(self, spec: dict) -> None:
                if spec is None:
                    return TensorSpec()
                self.spec = {k:Spec[k] for k in spec.keys()}

            def check(self, batch):
        
        class TensorSpec(Spec):
            pass


        # C-G-S-B-T
        spec = Spec(
            dict(
                input=dict(
                    _era5={
                        -1: {1: None},
                        0: {1: None},
                    }
                ),
                target=dict(
                    era5_={
                        1: {1: None},
                    },
                ),
            ),
        )
        spec.check(batch)


        spec = self.sample_provider.get_spec()
        data = self.sample_provider.build_structure(batch)
        indices = self.sample_provider.get_indices()

        new_input = data.advance_input(data, indices)

        for grp in groups:
            batch[0][grp][]


        train_loss, _, _ = self._step(batch, batch_idx)
        self.log(
            "train_" + self.loss.name + "_loss",
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            # batch_size=batch.shape[0],
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

    def validation_step(
        self,
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            val_loss, metrics, y_preds = self._step(batch, batch_idx, validation_mode=True)

        self.log(
            "val_" + self.loss.name + "_loss",
            val_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            # batch_size=batch.shape[0],
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
        """Configure the optimizers and learning rate scheduler.

        Returns
        -------
        tuple[list[torch.optim.Optimizer], list[dict]]
            List of optimizers and list of dictionaries containing the
            learning rate scheduler
        """
        if self.optimizer_settings.zero:
            optimizer = ZeroRedundancyOptimizer(
                self.trainer.model.parameters(),
                lr=self.lr,
                optimizer_class=torch.optim.AdamW,
                **self.optimizer_settings.kwargs,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.trainer.model.parameters(),
                lr=self.lr,
                **self.optimizer_settings.kwargs,
            )

        scheduler = CosineLRScheduler(
            optimizer,
            lr_min=self.lr_min,
            t_initial=self.lr_iterations,
            warmup_t=self.lr_warmup,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

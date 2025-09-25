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
from abc import ABC
from abc import abstractmethod

import pytorch_lightning as pl
import torch
from timm.scheduler import CosineLRScheduler
from torch.distributed.distributed_c10d import ProcessGroup
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch_geometric.data import HeteroData

from anemoi.models.data_structure.sample_provider import SampleProvider
from anemoi.models.preprocessing.normalisers import build_normaliser
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.dict import DictLoss
from anemoi.training.losses.scaler_tensor import grad_scaler
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.schemas.base_schema import convert_to_omegaconf
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


class BaseGraphPLModule(pl.LightningModule, ABC):
    """Abstract base class for Anemoi GNN forecasters using PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: BaseSchema,
        graph_data: HeteroData,
        sample_static_info: SampleProvider,
        metadata: dict,
    ) -> None:
        super().__init__()
        self.sample_static_info = sample_static_info

        self.normaliser = sample_static_info.map_expanded(build_normaliser).as_module_dict()

        self.graph_data = graph_data  # .to(self.device) # at init this will be on cpu

        # TODO: oandle supporting arrays for multiple output masks (multiple outputs)
        # (It is handled in the loss function, but not the version here that is sent to model for supporting_arrays)
        # self.output_mask = instantiate(config.model_dump(by_alias=True).model.output_mask, graph_data=graph_data)

        self.model = self.build_model(
            model_config=convert_to_omegaconf(config).model,
            sample_static_info=sample_static_info,
            metadata=metadata,
            # truncation_data=self.truncation_data,
        )

        self.config = config

        # self.save_hyperparameters() # needed for storing the checkpoints

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
            static_info=self.sample_static_info["target"],
        )
        # print_variable_scaling(self.loss, data_indices)

        self.metrics = torch.nn.ModuleDict({})

        if config.training.loss_gradient_scaling:
            print("registering_full_backward_hook for grad_scaler")
            self.loss.register_full_backward_hook(grad_scaler, prepend=False)

        self.is_first_step = True
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

        # lazy init model and reader group info, will be set by the DDPGroupStrategy:
        self.model_comm_group_id = 0
        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1

        self.reader_group_id = 0
        self.reader_group_rank = 0
        print("✅ BaseGraphPLModule initialized")

    def build_model(self, model_config, **kwargs) -> torch.nn.Module:
        if "_target_" not in model_config.model:
            raise ValueError("model_config must contain a '_target_' key for instantiation")

        # mimic the partial instantiate from hydra
        def find_class_by_name(target: str):
            components = target.split(".")
            module_path = ".".join(components[:-1])
            class_name = components[-1]
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)

        cls = find_class_by_name(model_config.model._target_)

        return cls(model_config=model_config, **kwargs)

    def apply_normaliser_to_batch(self, batch):
        # apply in-place normalisation : the non-normalised data is lost

        print(batch.to_str("⚠️batch before normalistation"))
        for k, v in batch.items():
            normaliser = self.normaliser[k]
            assert isinstance(normaliser, torch.nn.Module), type(normaliser)
            v["data"] = normaliser(v["data"])
        print(batch.to_str("⚠️batch after normalistation"))
        return batch
        # This could be done with but using .each is not recommended for now:
        # batch.each["data"] = self.normaliser.each(batch.each["data"])

    def log(self, *args, **kwargs):
        kwargs["logger"] = kwargs.get("logger", self.logger_enabled)
        return super().log(*args, **kwargs)

    def forward(self, x: dict[str, torch.Tensor], graph: HeteroData) -> dict[str, torch.Tensor]:
        return self.model(x, graph, model_comm_group=self.model_comm_group)

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

    def training_step(
        self,
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx  # unused
        loss, _, _ = self._step(batch)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        # self.log("rollout", float(self.rollout), on_step=True, rank_zero_only=True, sync_dist=False)
        return loss

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
        del batch_idx  # unused
        with torch.no_grad():
            loss, metrics, y_preds = self._step(batch, validation_mode=True)

        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)

        for mname, mvalue in metrics.items():
            self.log(
                "val_" + mname,
                mvalue,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                batch_size=batch.shape[0],
                sync_dist=True,
            )

        # TODO (latter): should return a dictionnary here, but the callbacks need to be adapted
        return loss, y_preds

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
        if len(self.metrics) == 0:
            return {}

        metrics = {}
        y_postprocessed = self.model.target_post_processors(y, in_place=False)
        y_pred_postprocessed = self.model.target_post_processors(y_pred, in_place=False)

        for metric_name, metric in self.metrics.items():

            if isinstance(metric, BaseLoss):
                assert isinstance(metric, DictLoss), type(metric)

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

    @abstractmethod
    def _step(
        self,
        batch: torch.Tensor,
        validation_mode: bool = False,
        apply_processors: bool = True,
    ):
        pass

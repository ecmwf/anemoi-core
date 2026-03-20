# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Mapping

import torch
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import open_dict
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.train.tasks.base import BaseGraphModule

LOGGER = logging.getLogger(__name__)


class GraphNowcaster(BaseGraphModule):
    """Interpolates between NWP states using surface observations."""

    task_type = "time-interpolator"

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        with open_dict(config.training):
            config.training.multistep_output = len(config.training.explicit_times.target)
        super().__init__(
            config=config,
            graph_data=graph_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )
        self.n_step_input = self.model.n_step_input

        configured_time_only = bool(config.training.get("decoder_context_time_only", True))
        if not configured_time_only:
            LOGGER.info("Ignoring decoder_context_time_only=False: nowcaster uses time-fraction-only conditioning.")
        self.decoder_context_time_only = True
        self.decoder_context_sources = {str(dataset_name): [str(dataset_name)] for dataset_name in self.dataset_names}

        self.input_times = list(range(self.n_step_input))

        explicit_input_times = [int(t) for t in config.training.explicit_times.input]
        if len(explicit_input_times) == 0:
            raise ValueError("`training.explicit_times.input` cannot be empty for nowcaster.")
        target_anchor = max(explicit_input_times)
        self.interp_times = [target_anchor + int(t) for t in config.training.explicit_times.target]
        sorted_indices = sorted(set(self.input_times).union(self.interp_times))
        self.imap = {data_index: batch_index for batch_index, data_index in enumerate(sorted_indices)}

        self.dataset_relative_time_indices, self.dataset_input_relative_time_indices = self._resolve_dataset_time_indices(
            metadata=metadata,
            fallback_relative_indices=sorted_indices,
            fallback_input_indices=self.input_times,
        )
        self.dataset_time_maps = {
            dataset_name: {int(time_idx): batch_idx for batch_idx, time_idx in enumerate(relative_indices)}
            for dataset_name, relative_indices in self.dataset_relative_time_indices.items()
        }

    @property
    def output_times(self) -> int:
        """Number of interpolation times (outer loop in plot callbacks; one forward, n_step_output steps)."""
        return len(self.interp_times)

    def get_init_step(self, rollout_step: int) -> int:
        return rollout_step

    def _resolve_dataset_time_indices(
        self,
        *,
        metadata: dict,
        fallback_relative_indices: list[int],
        fallback_input_indices: list[int],
    ) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
        metadata_inference = metadata.get("metadata_inference", {}) if isinstance(metadata, Mapping) else {}
        relative_by_dataset: Mapping | None = None
        input_by_dataset: Mapping | None = None
        for dataset_name in self.dataset_names:
            dataset_meta = metadata_inference.get(dataset_name, {}) if isinstance(metadata_inference, Mapping) else {}
            timesteps_meta = dataset_meta.get("timesteps", {}) if isinstance(dataset_meta, Mapping) else {}
            candidate_relative = timesteps_meta.get("relative_date_indices_training_by_dataset", None)
            candidate_input = timesteps_meta.get("relative_date_input_indices_training_by_dataset", None)
            if relative_by_dataset is None and isinstance(candidate_relative, Mapping):
                relative_by_dataset = candidate_relative
            if input_by_dataset is None and isinstance(candidate_input, Mapping):
                input_by_dataset = candidate_input
            if relative_by_dataset is not None and input_by_dataset is not None:
                break

        dataset_relative_indices: dict[str, list[int]] = {}
        dataset_input_indices: dict[str, list[int]] = {}
        for dataset_name in self.dataset_names:
            relative_indices = None
            if relative_by_dataset is not None:
                raw_indices = relative_by_dataset.get(dataset_name, None)
                if isinstance(raw_indices, (list, tuple, ListConfig)):
                    relative_indices = [int(v) for v in raw_indices]
            if not relative_indices:
                relative_indices = [int(v) for v in fallback_relative_indices]
            if len(relative_indices) != len(set(relative_indices)):
                raise ValueError(
                    f"Dataset {dataset_name} has duplicate relative time indices in metadata: {relative_indices}"
                )

            input_indices = None
            if input_by_dataset is not None:
                raw_indices = input_by_dataset.get(dataset_name, None)
                if isinstance(raw_indices, (list, tuple, ListConfig)):
                    input_indices = [int(v) for v in raw_indices]
            if not input_indices:
                input_indices = [int(v) for v in fallback_input_indices]
            if len(input_indices) == 0:
                raise ValueError(f"Dataset {dataset_name} has no input relative times configured.")
            relative_set = set(relative_indices)
            if any(int(value) not in relative_set for value in input_indices):
                raise ValueError(
                    f"Dataset {dataset_name} input relative times must be present in dataset relative times: "
                    f"input={input_indices}, relative={relative_indices}"
                )

            dataset_relative_indices[dataset_name] = relative_indices
            dataset_input_indices[dataset_name] = input_indices

        return dataset_relative_indices, dataset_input_indices

    def _lookup_required_time_indices(
        self,
        *,
        dataset_name: str,
        relative_times: list[int],
        usage: str,
    ) -> list[int]:
        time_map = self.dataset_time_maps.get(dataset_name, self.imap)
        batch_indices: list[int] = []
        for relative_time in relative_times:
            batch_idx = time_map.get(int(relative_time), None)
            if batch_idx is None:
                available = sorted(time_map.keys())
                raise ValueError(
                    f"Dataset {dataset_name} is missing exact {usage} time {relative_time}. "
                    f"Available times: {available}"
                )
            batch_indices.append(int(batch_idx))
        return batch_indices

    def _lookup_available_target_time_indices(self, *, dataset_name: str) -> tuple[list[int], list[int]]:
        time_map = self.dataset_time_maps.get(dataset_name, self.imap)
        pred_indices: list[int] = []
        batch_indices: list[int] = []
        for pred_idx, relative_time in enumerate(self.interp_times):
            batch_idx = time_map.get(int(relative_time), None)
            if batch_idx is None:
                continue
            pred_indices.append(int(pred_idx))
            batch_indices.append(int(batch_idx))
        return pred_indices, batch_indices

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> tuple[Tensor, Mapping[str, Tensor], Tensor]:
        decode_dataset_names = tuple(self.target_dataset_names)
        x, y = {}, {}
        processed_batch = {}
        for dataset_name, data_batch in batch.items():
            if data_batch.is_floating_point() and data_batch.dtype == torch.float64:
                data_batch = data_batch.float()
            processed_batch[dataset_name] = data_batch
            obs = {var.item() for var in self.data_indices[dataset_name].data.input.full}.difference(
                set(self.known_future_indices[dataset_name]),
            )
            if len(obs) == 0:
                assert (
                    len(self.known_future_indices[dataset_name]) > 0
                ), "If no observed variables, need known future variables to derive bounds."
                x_init = data_batch[:, itemgetter(*self.boundary_times)(self.imap)][
                    ...,
                    self.known_future_indices[dataset_name],
                ]  # bounds are derived from variables we know in the future
            else:
                assert (
                    len(self.known_future_indices[dataset_name]) == 0
                ), "Known future variables not supported for datasets with observed variables."
                x_init = data_batch[
                    :,
                    : self.n_step_input,
                    ...,
                    list(obs),
                ]  # here only past steps are used for observed vars
            if dataset_name in decode_dataset_names:
                y[dataset_name] = data_batch[:, itemgetter(*self.interp_times)(self.imap)]
            x[dataset_name] = x_init
        decoder_ctx = self._build_decoder_context(processed_batch, decode_dataset_names=decode_dataset_names)
        y_pred = self(
            x,
            decoder_context=decoder_ctx,
            decode_dataset_names=decode_dataset_names,
        )
        loss, metrics, y_pred = checkpoint(
            self.compute_loss_metrics,
            y_pred_for_loss,
            y,
            validation_mode=validation_mode,
            use_reentrant=False,
        )

        return loss, metrics, y_pred

    def compute_loss_metrics(
        self,
        y_pred: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        validation_mode: bool = False,
        **kwargs,
    ) -> tuple[Tensor | None, Mapping[str, Tensor], dict[str, torch.Tensor]]:
        """Compute loss/metrics for only the datasets present in this batch."""
        assert isinstance(y_pred, dict), "y_pred must be a dict keyed by dataset name"
        assert isinstance(y, dict), "y must be a dict keyed by dataset name"

        active_dataset_names = [name for name in self.target_dataset_names if name in y_pred and name in y]
        if len(active_dataset_names) == 0:
            raise ValueError("No active datasets with matching predictions and targets for loss computation.")

        total_loss, metrics_next, y_preds = None, {}, {}
        for dataset_name in active_dataset_names:
            dataset_loss, dataset_metrics, y_preds[dataset_name] = self.compute_dataset_loss_metrics(
                y_pred[dataset_name],
                y[dataset_name],
                validation_mode=validation_mode,
                dataset_name=dataset_name,
                **kwargs,
            )

            if dataset_loss is not None:
                dataset_loss_sum = dataset_loss.sum()
                total_loss = dataset_loss_sum if total_loss is None else total_loss + dataset_loss_sum

                if validation_mode:
                    loss_obj = self.loss[dataset_name]
                    loss_name = getattr(loss_obj, "name", loss_obj.__class__.__name__.lower())
                    metrics_next[f"{dataset_name}_{loss_name}_loss"] = dataset_loss

            for metric_name, metric_value in dataset_metrics.items():
                metrics_next[f"{dataset_name}_{metric_name}"] = metric_value

        return total_loss, metrics_next, y_preds

    def _build_decoder_context(
        self,
        batch: dict[str, torch.Tensor],
        decode_dataset_names: tuple[str, ...] | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Build decoder_context with key cond containing only time-fraction."""
        batch_size = next(iter(batch.values())).shape[0]
        ens_size = next(iter(batch.values())).shape[2]
        dtype = next(iter(batch.values())).dtype

        active_datasets = decode_dataset_names or tuple(self.dataset_names)
        num_output_times = len(self.interp_times)
        if num_output_times == 0:
            raise ValueError("Nowcaster has no interpolation times configured.")

        interp_values = torch.tensor(self.interp_times, device=self.device, dtype=dtype)
        interp_min = torch.min(interp_values)
        interp_max = torch.max(interp_values)
        interp_span = interp_max - interp_min
        if float(interp_span.item()) == 0.0:
            cond_fraction = torch.zeros_like(interp_values)
        else:
            cond_fraction = (interp_values - interp_min) / interp_span
        cond_fraction = cond_fraction.view(1, num_output_times, 1, 1, 1)

        ctx = {}
        for dataset_name in active_datasets:
            grid_size = batch[dataset_name].shape[3]
            ctx[dataset_name] = {
                "cond": cond_fraction.expand(batch_size, num_output_times, ens_size, grid_size, 1),
            }
        return ctx

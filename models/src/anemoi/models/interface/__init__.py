# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import uuid
from typing import Any
from typing import Mapping
from typing import Optional

import torch
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.preprocessing import Processors


class AnemoiModelInterface(torch.nn.Module):
    """An interface for Anemoi models.

    This class is a wrapper around the Anemoi model that includes pre-processing and post-processing steps.
    It inherits from the PyTorch Module class.

    Attributes
    ----------
    config : Mapping[str, Any]
        Configuration settings for the model.
    id : str
        A unique identifier for the model instance.
    multi_step : bool
        Whether the model uses multi-step input.
    graph_data : HeteroData
        Graph data for the model.
    statistics : dict
        Statistics for the data.
    metadata : dict
        Metadata for the model.
    statistics_tendencies : dict
        Statistics for the tendencies of the data.
    supporting_arrays : dict
        Numpy arraysto store in the checkpoint.
    data_indices : dict
        Indices for the data.
    pre_processors : Processors
        Pre-processing steps to apply to the data before passing it to the model.
    post_processors : Processors
        Post-processing steps to apply to the model's output.
    model : AnemoiModelEncProcDec
        The underlying Anemoi model.
    """

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        graph_data: HeteroData,
        statistics: dict,
        data_indices: dict,
        metadata: dict,
        statistics_tendencies: dict | None = None,
        supporting_arrays: dict | None = None,
        model: torch.nn.Module | None = None,
        pre_processors: Mapping[str, Processors] | Processors | None = None,
        post_processors: Mapping[str, Processors] | Processors | None = None,
        pre_processors_tendencies: Mapping[str, Processors] | Processors | None = None,
        post_processors_tendencies: Mapping[str, Processors] | Processors | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.id = str(uuid.uuid4())
        self.multi_step = self.config.training.multistep_input
        self.graph_data = graph_data
        self.statistics = statistics
        self.statistics_tendencies = statistics_tendencies
        self.metadata = metadata
        self.supporting_arrays = supporting_arrays if supporting_arrays is not None else {}
        self.data_indices = data_indices
        self._build_model(
            model=model,
            pre_processors=pre_processors,
            post_processors=post_processors,
            pre_processors_tendencies=pre_processors_tendencies,
            post_processors_tendencies=post_processors_tendencies,
        )
        self._update_metadata()

    def _ensure_processors_mapping(
        self,
        processors: Mapping[str, Processors] | Processors | None,
        dataset_names: list[str],
        label: str,
    ) -> Mapping[str, Processors] | None:
        if processors is None:
            return None
        if isinstance(processors, Processors):
            if len(dataset_names) != 1:
                msg = f"{label} must be a mapping for multi-dataset models."
                raise ValueError(msg)
            return {dataset_names[0]: processors}
        if isinstance(processors, torch.nn.ModuleDict):
            processors_map = {name: processors[name] for name in processors}
        elif isinstance(processors, Mapping):
            processors_map = dict(processors)
        else:
            msg = f"{label} must be a Processors instance or a mapping of dataset_name -> Processors."
            raise TypeError(msg)

        return processors_map

    def _build_model(
        self,
        *,
        model: torch.nn.Module | None = None,
        pre_processors: Mapping[str, Processors] | Processors | None = None,
        post_processors: Mapping[str, Processors] | Processors | None = None,
        pre_processors_tendencies: Mapping[str, Processors] | Processors | None = None,
        post_processors_tendencies: Mapping[str, Processors] | Processors | None = None,
    ) -> None:
        """Builds the model and pre- and post-processors."""
        dataset_names = list(self.statistics.keys())
        pre_map = self._ensure_processors_mapping(pre_processors, dataset_names, "pre_processors")
        post_map = self._ensure_processors_mapping(post_processors, dataset_names, "post_processors")
        pre_tend_map = self._ensure_processors_mapping(
            pre_processors_tendencies,
            dataset_names,
            "pre_processors_tendencies",
        )
        post_tend_map = self._ensure_processors_mapping(
            post_processors_tendencies,
            dataset_names,
            "post_processors_tendencies",
        )

        if model is None:
            msg = "model must be provided; construct it outside the interface."
            raise ValueError(msg)

        self.pre_processors = torch.nn.ModuleDict(dict(pre_map or {}))
        self.post_processors = torch.nn.ModuleDict(dict(post_map or {}))
        self.pre_processors_tendencies = torch.nn.ModuleDict(dict(pre_tend_map or {}))
        self.post_processors_tendencies = torch.nn.ModuleDict(dict(post_tend_map or {}))

        self.model = model

        # Use the forward method of the model directly
        self.forward = self.model.forward

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Prediction step for the model.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Input batched data.
        model_comm_group : Optional[ProcessGroup], optional
            model communication group, specifies which GPUs work together
        gather_out : bool, optional
            Whether to gather the output, by default True.

        Returns
        -------
        dict[str, torch.Tensor]
            Predicted data.
        """
        # Prepare kwargs for model's predict_step
        predict_kwargs = {
            "batch": batch,
            "pre_processors": self.pre_processors,
            "post_processors": self.post_processors,
            "multi_step": self.multi_step,
            "model_comm_group": model_comm_group,
        }

        # Add tendency processors if they exist
        if hasattr(self, "pre_processors_tendencies"):
            predict_kwargs["pre_processors_tendencies"] = self.pre_processors_tendencies
        if hasattr(self, "post_processors_tendencies"):
            predict_kwargs["post_processors_tendencies"] = self.post_processors_tendencies

        # Delegate to the model's predict_step implementation with processors
        return self.model.predict_step(**predict_kwargs, **kwargs)

    def _update_metadata(self) -> None:
        self.model.fill_metadata(self.metadata)

# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import uuid
from typing import Optional

import torch
from hydra.utils import instantiate
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.preprocessing import Processors
from anemoi.utils.config import DotDict


class AnemoiModelInterface(torch.nn.Module):
    """An interface for Anemoi models.

    This class is a wrapper around the Anemoi model that includes pre-processing and post-processing steps.
    It inherits from the PyTorch Module class.

    Attributes
    ----------
    config : DotDict
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
        config: DotDict,
        graph_data: HeteroData,
        statistics: dict,
        data_indices: dict,
        metadata: dict,
        statistics_tendencies: dict = None,
        supporting_arrays: dict = None,
        truncation_data: dict,
    ) -> None:
        super().__init__()
        self.config = config
        self.id = str(uuid.uuid4())
        self.multi_step = self.config.training.multistep_input
        self.graph_data = graph_data
        self.statistics = statistics
        self.statistics_tendencies = statistics_tendencies
        self.truncation_data = truncation_data
        self.metadata = metadata
        self.supporting_arrays = supporting_arrays if supporting_arrays is not None else {}
        self.data_indices = data_indices
        self._build_model()

    def _build_processors_for_dataset(
        self, dataset_name: str, statistics: dict, data_indices: dict, statistics_tendencies: dict = None
    ):
        """Build processors for a single dataset.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        statistics : dict
            Statistics for the dataset
        data_indices : dict
            Data indices for the dataset
        statistics_tendencies : dict, optional
            Tendencies statistics for the dataset

        Returns
        -------
        tuple
            (pre_processors, post_processors, pre_processors_tendencies, post_processors_tendencies)
        """
        from anemoi.training.utils.config_utils import get_dataset_data_config

        # Get dataset-specific data config
        dataset_data_config = get_dataset_data_config(self.config, dataset_name)

        # Build processors for the dataset
        processors = [
            [name, instantiate(processor, data_indices=data_indices, statistics=statistics)]
            for name, processor in dataset_data_config.processors.items()
        ]

        pre_processors = Processors(processors)
        post_processors = Processors(processors, inverse=True)

        # Build tendencies processors if provided
        pre_processors_tendencies = None
        post_processors_tendencies = None
        if statistics_tendencies is not None:
            processors_tendencies = [
                [name, instantiate(processor, data_indices=data_indices, statistics=statistics_tendencies)]
                for name, processor in dataset_data_config.processors.items()
            ]
            pre_processors_tendencies = Processors(processors_tendencies)
            post_processors_tendencies = Processors(processors_tendencies, inverse=True)

        return pre_processors, post_processors, pre_processors_tendencies, post_processors_tendencies

    def _build_model(self) -> None:
        """Builds the model and pre- and post-processors."""
        # Multi-dataset mode: create processors for each dataset
        self.pre_processors = torch.nn.ModuleDict()
        self.post_processors = torch.nn.ModuleDict()
        self.pre_processors_tendencies = torch.nn.ModuleDict()
        self.post_processors_tendencies = torch.nn.ModuleDict()

        for dataset_name in self.statistics.keys():
            # Build processors for each dataset
            pre, post, pre_tend, post_tend = self._build_processors_for_dataset(
                dataset_name,
                self.statistics[dataset_name],
                self.data_indices[dataset_name],
                self.statistics_tendencies[dataset_name] if self.statistics_tendencies is not None else None,
            )
            self.pre_processors[dataset_name] = pre
            self.post_processors[dataset_name] = post
            if pre_tend is not None:
                self.pre_processors_tendencies[dataset_name] = pre_tend
                self.post_processors_tendencies[dataset_name] = post_tend

        # Instantiate the model
        # Only pass _target_ and _convert_ from model config to avoid passing diffusion as kwarg
        model_instantiate_config = {
            "_target_": self.config.model.model._target_,
            "_convert_": getattr(self.config.model.model, "_convert_", "all"),
        }
        self.model = instantiate(
            model_instantiate_config,
            model_config=self.config,
            data_indices=self.data_indices,
            statistics=self.statistics,
            graph_data=self.graph_data,
            truncation_data=self.truncation_data,
            _recursive_=False,  # Disables recursive instantiation by Hydra
        )

        # Use the forward method of the model directly
        self.forward = self.model.forward

    def predict_step(
        self, batch: torch.Tensor, model_comm_group: Optional[ProcessGroup] = None, gather_out: bool = True, **kwargs
    ) -> torch.Tensor:
        """Prediction step for the model.

        Parameters
        ----------
        batch : torch.Tensor
            Input batched data.
        model_comm_group : Optional[ProcessGroup], optional
            model communication group, specifies which GPUs work together
        gather_out : bool, optional
            Whether to gather the output, by default True.

        Returns
        -------
        torch.Tensor
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

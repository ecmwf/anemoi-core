# (C) Copyright 2024-2026 Anemoi contributors.
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
from hydra.utils import get_class
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing import StepwiseProcessors
from anemoi.models.preprocessing.imputer import BaseImputer
from anemoi.models.preprocessing.normalizer import InputNormalizer
from anemoi.models.preprocessing.postprocessor import Postprocessor
from anemoi.models.preprocessing.spatial import SpatialPreprocessor
from anemoi.models.utils.config import get_multiple_datasets_config


class AnemoiModelInterface(torch.nn.Module):
    """An interface for Anemoi models.

    This class is a wrapper around the Anemoi model that includes pre-processing and post-processing steps.
    It inherits from the PyTorch Module class.

    Attributes
    ----------
    config : DictConfig
        Configuration settings for the model.
    id : str
        A unique identifier for the model instance.
    n_step_input : int
        Number of input timesteps provided to the model.
    graph_data : HeteroData
        Graph data for the model.
    statistics : dict
        Statistics for the data.
    metadata : dict
        Metadata for the model.
    statistics_tendencies : dict
        Statistics for the tendencies of the data.
    residual_statistics : dict
        Precomputed residual normalization statistics, keyed by dataset name.
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
        config: DictConfig,
        n_step_input: int,
        n_step_output: int,
        graph_data: HeteroData,
        statistics: dict,
        data_indices: dict,
        metadata: dict,
        statistics_tendencies: dict | None = None,
        residual_statistics: dict | None = None,
        supporting_arrays: dict | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.id = str(uuid.uuid4())
        self.n_step_input = n_step_input
        self.n_step_output = n_step_output
        self.graph_data = graph_data
        self.statistics = statistics
        self.statistics_tendencies = statistics_tendencies
        self.residual_statistics = residual_statistics
        self.metadata = metadata
        self.supporting_arrays = supporting_arrays if supporting_arrays is not None else {}
        self.data_indices = data_indices
        self._build_model()
        self._update_metadata()

    def _build_processors_for_dataset(
        self,
        processors_configs: dict,
        statistics: dict,
        data_indices: dict,
        statistics_tendencies: dict | None = None,
        *,
        residual_statistics: dict | None = None,
        dataset_name: str,
    ) -> tuple[
        Processors,
        Processors,
        Processors | StepwiseProcessors | None,
        Processors | StepwiseProcessors | None,
        Processors | None,
        Processors | None,
    ]:
        """Build processors for a single dataset.

        Parameters
        ----------
        processors_configs : dict
            Configuration for the processors.
        statistics : dict
            Statistics for the dataset.
        data_indices : dict
            Data indices for the dataset.
        statistics_tendencies : dict, optional
            Tendencies statistics for the dataset.
        residual_statistics : dict, optional
            Residual statistics for the dataset.
        dataset_name : str
            Name of the dataset, used in error messages.

        Returns
        -------
        tuple
            ``(pre_processors, post_processors, pre_processors_tendencies,
            post_processors_tendencies, pre_processors_residual,
            post_processors_residual)``.  Exactly one of the two extra pairs is
            populated (depending on whether ``residual_statistics`` is set);
            the other two entries are ``None``.
        """
        pre_processors, post_processors = self._build_processor_pair(
            processors_configs,
            data_indices,
            statistics,
        )
        pre_processors_tendencies: Processors | StepwiseProcessors | None = None
        post_processors_tendencies: Processors | StepwiseProcessors | None = None
        pre_processors_residual: Processors | None = None
        post_processors_residual: Processors | None = None
        if residual_statistics is not None:
            pre_processors_residual, post_processors_residual = self._build_residual_processors(
                processors_configs,
                data_indices,
                residual_statistics,
                dataset_name=dataset_name,
            )
        else:
            pre_processors_tendencies, post_processors_tendencies = self._build_tendency_processors(
                processors_configs,
                data_indices,
                statistics_tendencies,
            )
        return (
            pre_processors,
            post_processors,
            pre_processors_tendencies,
            post_processors_tendencies,
            pre_processors_residual,
            post_processors_residual,
        )

    @staticmethod
    def _build_processor_pair(
        processors_configs: dict,
        data_indices: dict,
        statistics: dict,
    ) -> tuple[Processors, Processors]:
        processors = [
            [name, instantiate(processor, data_indices=data_indices, statistics=statistics)]
            for name, processor in processors_configs.items()
        ]
        return Processors(processors), Processors(processors, inverse=True)

    def _build_tendency_processors(
        self,
        processors_configs: dict,
        data_indices: dict,
        statistics_tendencies: dict | None,
    ) -> tuple[Processors | StepwiseProcessors | None, Processors | StepwiseProcessors | None]:
        if statistics_tendencies is None:
            return None, None

        if "lead_times" not in statistics_tendencies:
            return self._build_processor_pair(processors_configs, data_indices, statistics_tendencies)

        lead_times = list(statistics_tendencies.get("lead_times") or [])
        if self.n_step_output == 1:
            step_stats = statistics_tendencies.get(lead_times[0]) if lead_times else None
            stats_for_tendencies = step_stats or statistics_tendencies
            return self._build_processor_pair(processors_configs, data_indices, stats_for_tendencies)

        pre_processors_tendencies = StepwiseProcessors(lead_times)
        post_processors_tendencies = StepwiseProcessors(lead_times)
        for lead_time in lead_times:
            step_stats = statistics_tendencies.get(lead_time)
            if step_stats is None:
                continue
            pre_step, post_step = self._build_processor_pair(processors_configs, data_indices, step_stats)
            pre_processors_tendencies.set(lead_time, pre_step)
            post_processors_tendencies.set(lead_time, post_step)
        return pre_processors_tendencies, post_processors_tendencies

    def _build_residual_processors(
        self,
        processors_configs: dict,
        data_indices: dict,
        residual_statistics: dict,
        *,
        dataset_name: str,
    ) -> tuple[Processors, Processors]:
        """Build the residual pre/post-processor pair from the dataset's normaliser alone.

        Imputers are left out: residuals are always computed with imputation skipped,
        and the state processors still impute the batch. Post-processors (e.g. clamping)
        are left out too: they act on the rebuilt state through the state processors.
        Any other processor (e.g. a remapper) would act on the residual itself, which
        can be negative, so it is rejected.
        """
        residual_configs = {}
        for name, processor_config in processors_configs.items():
            processor_class = get_class(processor_config["_target_"])
            if issubclass(processor_class, (BaseImputer, Postprocessor)):
                continue
            if not issubclass(processor_class, InputNormalizer):
                raise ValueError(
                    f"Residual target '{dataset_name}' configures processor '{name}' "
                    f"({processor_config['_target_']}). Residual processors support only the normaliser "
                    "(imputers and post-processors apply to the state only); any other processor would be "
                    f"applied to the residual itself. Remove it from data.datasets.{dataset_name}.processors."
                )
            residual_configs[name] = processor_config

        if not residual_configs:
            raise ValueError(
                f"Residual target '{dataset_name}' has residual statistics but no normaliser in "
                f"data.datasets.{dataset_name}.processors."
            )

        return self._build_processor_pair(residual_configs, data_indices, residual_statistics)

    def _build_model(self) -> None:
        """Builds the model and pre- and post-processors."""
        # Multi-dataset mode: create processors for each dataset
        self.pre_processors = torch.nn.ModuleDict()
        self.post_processors = torch.nn.ModuleDict()
        self.pre_processors_tendencies = torch.nn.ModuleDict()
        self.post_processors_tendencies = torch.nn.ModuleDict()
        self.pre_processors_residual = torch.nn.ModuleDict()
        self.post_processors_residual = torch.nn.ModuleDict()

        data_config = get_multiple_datasets_config(self.config.data)
        for dataset_name in self.statistics.keys():
            # Build processors for each dataset
            (
                pre,
                post,
                pre_tend,
                post_tend,
                pre_res,
                post_res,
            ) = self._build_processors_for_dataset(
                data_config[dataset_name].processors,
                self.statistics[dataset_name],
                self.data_indices[dataset_name],
                self.statistics_tendencies[dataset_name] if self.statistics_tendencies is not None else None,
                residual_statistics=(
                    self.residual_statistics.get(dataset_name) if self.residual_statistics is not None else None
                ),
                dataset_name=dataset_name,
            )
            self.pre_processors[dataset_name] = pre
            self.post_processors[dataset_name] = post
            if pre_tend is not None:
                self.pre_processors_tendencies[dataset_name] = pre_tend
                self.post_processors_tendencies[dataset_name] = post_tend
            if pre_res is not None:
                self.pre_processors_residual[dataset_name] = pre_res
                self.post_processors_residual[dataset_name] = post_res

        # Spatial preprocessors (e.g. CrossGridProjector for downscaling).
        # Keyed by dataset name; empty by default so existing models are unaffected.
        # Built from optional config.data.datasets.<dataset_name>.spatial_processor entries.
        self.spatial_pre_processors: torch.nn.ModuleDict = torch.nn.ModuleDict()
        for dataset_name, dataset_config in data_config.items():
            sp_config = getattr(dataset_config, "spatial_processor", None)
            if sp_config is None:
                continue
            projector = instantiate(sp_config, graph=self.graph_data, _recursive_=False)
            if not isinstance(projector, SpatialPreprocessor):
                raise TypeError(
                    f"datasets.{dataset_name}.spatial_processor must instantiate a SpatialPreprocessor, "
                    f"got {type(projector)}"
                )
            self.spatial_pre_processors[dataset_name] = projector

        # Instantiate the model
        # Only pass _target_ and _convert_ from model config to avoid passing nested model settings as kwargs.
        model_instantiate_config = {
            "_target_": self.config.model.model._target_,
            "_convert_": getattr(self.config.model.model, "_convert_", "none"),
        }
        self.model = instantiate(
            model_instantiate_config,
            model_config=self.config.model,
            data_indices=self.data_indices,
            statistics=self.statistics,
            graph_data=self.graph_data,
            n_step_input=self.n_step_input,
            n_step_output=self.n_step_output,
            _recursive_=False,  # Disables recursive instantiation by Hydra
        )

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
            Model communication group, specifies which GPUs work together.
        gather_out : bool, optional
            Whether to gather the output, by default True.
        **kwargs
            Additional prediction keyword arguments.

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
            "n_step_input": self.n_step_input,
            "model_comm_group": model_comm_group,
        }

        # Add tendency processors if they exist
        if hasattr(self, "pre_processors_tendencies"):
            predict_kwargs["pre_processors_tendencies"] = self.pre_processors_tendencies
        if hasattr(self, "post_processors_tendencies"):
            predict_kwargs["post_processors_tendencies"] = self.post_processors_tendencies

        if hasattr(self, "pre_processors_residual") and len(self.pre_processors_residual) > 0:
            predict_kwargs["pre_processors_residual"] = self.pre_processors_residual
        if hasattr(self, "post_processors_residual") and len(self.post_processors_residual) > 0:
            predict_kwargs["post_processors_residual"] = self.post_processors_residual
        if getattr(self, "spatial_pre_processors", None):
            predict_kwargs["spatial_pre_processors"] = self.spatial_pre_processors

        # Delegate to the model's predict_step implementation with processors
        return self.model.predict_step(**predict_kwargs, **kwargs)

    def _update_metadata(self) -> None:
        self.model.fill_metadata(self.metadata)

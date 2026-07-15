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
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing import StepwiseProcessors
from anemoi.models.preprocessing.normalizer import InputNormalizer
from anemoi.models.utils.config import get_multiple_datasets_config

# Denormalizing an interpolated normalized source is exact only for affine pre-processors.
# Residual downscaling requires interpolation and normalization to commute, so only affine
# normalizers are permitted on any source/target dataset feeding a residual pair. Extend this
# allowlist deliberately (only with processors whose transform is affine per channel).
AFFINE_PRE_PROCESSORS: tuple[type, ...] = (InputNormalizer,)


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
    statistics_residuals : dict
        First-class statistics for spatial residuals.
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
        statistics_residuals: dict | None = None,
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
        self.statistics_residuals = statistics_residuals
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
    ) -> tuple[
        Processors,
        Processors,
        Processors | StepwiseProcessors | None,
        Processors | StepwiseProcessors | None,
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

        Returns
        -------
        tuple
            (pre_processors, post_processors, pre_processors_tendencies, post_processors_tendencies).
        """
        pre_processors, post_processors = self._build_processor_pair(
            processors_configs,
            data_indices,
            statistics,
        )
        pre_processors_tendencies, post_processors_tendencies = self._build_tendency_processors(
            processors_configs,
            data_indices,
            statistics_tendencies,
        )
        return pre_processors, post_processors, pre_processors_tendencies, post_processors_tendencies

    def _build_residual_processors(
        self,
        processors_configs: dict,
        data_indices: dict,
        statistics_residuals: dict | None,
    ) -> tuple[Processors, Processors] | tuple[None, None]:
        """Build residual processors from explicit residual statistics.

        Residual normalization must never silently fall back to state statistics.
        """
        if statistics_residuals is None:
            return None, None
        return self._build_processor_pair(processors_configs, data_indices, statistics_residuals)

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

    def _build_model(self) -> None:
        """Builds the model and pre- and post-processors."""
        # Multi-dataset mode: create processors for each dataset
        self.pre_processors = torch.nn.ModuleDict()
        self.post_processors = torch.nn.ModuleDict()
        self.pre_processors_tendencies = torch.nn.ModuleDict()
        self.post_processors_tendencies = torch.nn.ModuleDict()
        self.pre_processors_residuals = torch.nn.ModuleDict()
        self.post_processors_residuals = torch.nn.ModuleDict()

        data_config = get_multiple_datasets_config(self.config.data)
        for dataset_name in self.statistics.keys():
            # Build processors for each dataset
            pre, post, pre_tend, post_tend = self._build_processors_for_dataset(
                data_config[dataset_name].processors,
                self.statistics[dataset_name],
                self.data_indices[dataset_name],
                self.statistics_tendencies[dataset_name] if self.statistics_tendencies is not None else None,
            )
            self.pre_processors[dataset_name] = pre
            self.post_processors[dataset_name] = post
            if pre_tend is not None:
                self.pre_processors_tendencies[dataset_name] = pre_tend
                self.post_processors_tendencies[dataset_name] = post_tend
            residual_stats = self.statistics_residuals.get(dataset_name) if self.statistics_residuals else None
            pre_residual, post_residual = self._build_residual_processors(
                data_config[dataset_name].processors,
                self.data_indices[dataset_name],
                residual_stats,
            )
            if pre_residual is not None:
                self.pre_processors_residuals[dataset_name] = pre_residual
                self.post_processors_residuals[dataset_name] = post_residual

        # Instantiate the model
        # Only pass _target_ and _convert_ from model config to avoid passing nested model settings as kwargs.
        model_instantiate_config = {
            "_target_": self.config.model.model._target_,
            "_convert_": getattr(self.config.model.model, "_convert_", "none"),
        }
        self.model = instantiate(
            model_instantiate_config,
            model_config=self.config,
            data_indices=self.data_indices,
            statistics=self.statistics,
            graph_data=self.graph_data,
            n_step_input=self.n_step_input,
            n_step_output=self.n_step_output,
            _recursive_=False,  # Disables recursive instantiation by Hydra
        )

        self._validate_residual_linearity()

        # Use the forward method of the model directly
        self.forward = self.model.forward

    def _validate_residual_linearity(self) -> None:
        """Reject non-affine pre-processors on datasets feeding a residual pair.

        Residual reconstruction denormalizes an interpolated *normalized* source; this equals the
        interpolation of the physical source only when normalization is affine (interpolation and
        normalization commute). A non-affine processor (imputer, remapper, non-linear transform)
        would silently break that identity, so we fail loudly at setup.
        """
        pairs = getattr(self.model, "_residual_pairs", {}) or {}
        if not pairs:
            return
        involved = set(pairs) | set(pairs.values())
        for dataset_name in sorted(involved):
            if dataset_name not in self.pre_processors:
                continue
            processors = self.pre_processors[dataset_name]
            for name, processor in processors.processors.items():
                if not isinstance(processor, AFFINE_PRE_PROCESSORS):
                    raise ValueError(
                        f"Residual downscaling requires affine pre-processors so interpolation and "
                        f"normalization commute, but dataset '{dataset_name}' uses non-affine processor "
                        f"'{name}' ({type(processor).__name__}). Allowed: "
                        f"{[cls.__name__ for cls in AFFINE_PRE_PROCESSORS]}. "
                        "Remove it from the source/target datasets feeding the residual pair, or extend "
                        "AFFINE_PRE_PROCESSORS deliberately if it is provably affine."
                    )

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
        if hasattr(self, "pre_processors_residuals"):
            predict_kwargs["pre_processors_residuals"] = self.pre_processors_residuals
        if hasattr(self, "post_processors_residuals"):
            predict_kwargs["post_processors_residuals"] = self.post_processors_residuals

        # Delegate to the model's predict_step implementation with processors
        return self.model.predict_step(**predict_kwargs, **kwargs)

    def _update_metadata(self) -> None:
        self.model.fill_metadata(self.metadata)

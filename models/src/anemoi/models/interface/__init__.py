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

from anemoi.models.data.batch import Batch
from anemoi.models.data.layout import TensorLayout
from anemoi.models.data.sources import TabularSource
from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing import StepwiseProcessors
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
    n_step_input : dict[str, int]
        Number of input timesteps provided to the model for each dataset and location.
    n_step_output : dict[str, int]
        Number of output timesteps predicted by the model for each dataset and location.
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
        config: DictConfig,
        n_step_input: dict[str, int],
        n_step_output: dict[str, int],
        statistics: dict,
        data_indices: dict,
        metadata: dict,
        data_readers: dict,
        statistics_tendencies: dict | None = None,
        supporting_arrays: dict | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.id = str(uuid.uuid4())
        self.n_step_input = n_step_input
        self.n_step_output = n_step_output
        self.statistics = statistics
        self.statistics_tendencies = statistics_tendencies
        self.metadata = metadata
        self.supporting_arrays = supporting_arrays if supporting_arrays is not None else {}
        self.data_indices = data_indices

        self.is_dataset_static = {key: val.is_static_grid for key, val in data_readers.items()}
        self.sample_types = {name: reader.sample_type for name, reader in data_readers.items()}

        self._build_model()
        self._update_metadata()

    def _build_processors_for_dataset(
        self,
        processors_configs: dict,
        statistics: dict,
        data_indices: dict,
        dataset_name: str,
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
            dataset_name,
        )
        return pre_processors, post_processors, pre_processors_tendencies, post_processors_tendencies

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
        dataset_name: str,
    ) -> tuple[Processors | StepwiseProcessors | None, Processors | StepwiseProcessors | None]:
        if statistics_tendencies is None:
            return None, None

        if "lead_times" not in statistics_tendencies:
            return self._build_processor_pair(processors_configs, data_indices, statistics_tendencies)

        lead_times = list(statistics_tendencies.get("lead_times") or [])
        if self.n_step_output[dataset_name] == 1:
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

        data_config = get_multiple_datasets_config(self.config.data)
        for dataset_name in self.statistics.keys():
            # Build processors for each dataset
            pre, post, pre_tend, post_tend = self._build_processors_for_dataset(
                data_config[dataset_name].processors,
                self.statistics[dataset_name],
                self.data_indices[dataset_name],
                dataset_name,
                self.statistics_tendencies[dataset_name] if self.statistics_tendencies is not None else None,
            )
            self.pre_processors[dataset_name] = pre
            self.post_processors[dataset_name] = post
            if pre_tend is not None:
                self.pre_processors_tendencies[dataset_name] = pre_tend
                self.post_processors_tendencies[dataset_name] = post_tend

        # Instantiate the model
        # Only pass _target_ and _convert_ from model config to avoid passing nested model settings as kwargs.
        model_instantiate_config = {
            "_target_": self.config.model.model._target_,
            "_convert_": getattr(self.config.model.model, "_convert_", "none"),
        }
        self.model = instantiate(
            model_instantiate_config,
            model_config=self.config,
            model_graph_config=self.config.graph,
            data_indices=self.data_indices,
            statistics=self.statistics,
            is_dataset_static=self.is_dataset_static,
            data_layouts=self.data_layouts,
            n_step_input=self.n_step_input,
            n_step_output=self.n_step_output,
            _recursive_=False,  # Disables recursive instantiation by Hydra
        )
        # The model builds the graph from the graph config; spatial projectors read their edges from it.
        self.graph_data = self.model._graph_data

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

        # Use the forward method of the model directly
        self.forward = self.model.forward

    def apply_spatial_pre_processors(self, batch: Batch, model_comm_group: Optional[ProcessGroup] = None) -> Batch:
        """Project each dataset that has a spatial preprocessor onto its graph node set.

        Applied to raw (un-normalized) data. The projected source carries the coordinates
        of the dataset's graph nodes, which is the grid the encoder runs on.
        """
        for dataset_name, projector in self.spatial_pre_processors.items():
            if dataset_name in batch:
                projected = projector.project_source(
                    batch[dataset_name],
                    self.graph_data[dataset_name].x,
                    model_comm_group=model_comm_group,
                )
                batch = batch.replace(dataset_name, projected)
        return batch

    @staticmethod
    def _as_payload(ds_data: torch.Tensor | dict) -> dict:
        """Normalise one dataset entry to the payload dict of the inference boundary.
        A bare tensor is accepted as a data-only payload.
        """
        return ds_data if isinstance(ds_data, dict) else {"data": ds_data}

    def _statistics_for(self, dataset_name: str, variables: list[str]) -> dict:
        """Slice the checkpoint's data-space statistics down to the variables.
        Same alignment that is done for the model outputs in AnemoiModelEncProcDec._assemble_output.
        """
        name_to_index = self.data_indices[dataset_name].name_to_index
        positions = [name_to_index[name] for name in variables]
        return {name: values[positions] for name, values in self.statistics[dataset_name].items()}

    def _target_forcing_names(self, dataset_name: str) -> list[str]:
        """Returns the names of the output-time forcing variables that condition this dataset's decoder.
        Mirrors the BaseTask.get_forcings method.
        """
        data_input = self.data_indices[dataset_name].data.input
        return [data_input.full_index_to_name[int(index)] for index in data_input.forcing]

    def _source_sample(self, dataset_name: str, payload: dict) -> SourceSample:
        """Build one dataset's SourceSample from a plain inference payload.

        The payload carries ``data``, ``latitudes`` / ``longitudes`` (degrees), ``layout``
        (per-sample axis names, no batch axis) and ``variables``; tabular datasets also carry
        ``timedeltas`` and ``boundaries`` (``(start, stop)`` pairs, one per time window).
        Statistics, ``time_in_grid`` and whether the grid is static come from the checkpoint.
        The payload is not modified.
        """
        for key in ("data", "latitudes", "longitudes", "layout", "variables"):
            if payload.get(key) is None:
                raise ValueError(f"Dataset {dataset_name!r}: missing {key!r} in the sample.")

        data = payload["data"]
        latitudes = torch.as_tensor(payload["latitudes"], dtype=torch.float32).reshape(-1)
        longitudes = torch.as_tensor(payload["longitudes"], dtype=torch.float32).reshape(-1)
        if latitudes.shape != longitudes.shape:
            raise ValueError(
                f"Dataset {dataset_name!r}: latitudes {tuple(latitudes.shape)} and longitudes "
                f"{tuple(longitudes.shape)} must describe the same nodes."
            )
        coordinates = torch.deg2rad(torch.stack([latitudes, longitudes], dim=-1)).to(device=data.device)

        layout_names = tuple(payload["layout"])
        if "batch" in layout_names:
            raise ValueError(f"Dataset {dataset_name!r}: the sample layout {layout_names} must not have a batch axis.")
        if data.ndim != len(layout_names):
            raise ValueError(
                f"Dataset {dataset_name!r}: data of shape {tuple(data.shape)} does not match the layout {layout_names}."
            )
        # time_in_grid cannot be read off the axis names, so it comes from the dataset kind.
        is_tabular = self.data_layouts[dataset_name].time_in_grid
        layout = TensorLayout.from_tuple(*layout_names, time_in_grid=is_tabular)

        variables = list(payload["variables"])
        if data.shape[layout.axis("variables", ndim=data.ndim)] != len(variables):
            raise ValueError(
                f"Dataset {dataset_name!r}: data carries {data.shape[layout.axis('variables', ndim=data.ndim)]} "
                f"variables but {len(variables)} names were given."
            )

        timedeltas = payload.get("timedeltas")
        if timedeltas is not None:
            timedeltas = torch.as_tensor(timedeltas, dtype=torch.float32).reshape(-1).to(device=data.device)

        boundaries = payload.get("boundaries")
        if boundaries is not None:
            boundaries = tuple(slice(int(start), int(stop)) for start, stop in boundaries)

        return SourceSample(
            data=data,
            variables=variables,
            layout=layout,
            statistics=self._statistics_for(dataset_name, variables),
            grid_size=None if is_tabular else coordinates.shape[0],
            coordinates_are_static=self.is_dataset_static[dataset_name] and not is_tabular,
            coordinates=coordinates,
            timedeltas=timedeltas,
            boundaries=boundaries,
        )

    def get_batch(self, data: dict[str, dict]) -> Batch:
        """Collate the per-dataset payloads into a single-sample Batch."""
        return Batch.collate(
            {dataset_name: self._source_sample(dataset_name, payload) for dataset_name, payload in data.items()},
        )

    def unwrap_batch(self, batch: Batch) -> dict[str, dict]:
        """Convert a model output Batch back to plain per-dataset payload dicts.
        The coordinates are converted from radians to degrees, and the batch axis of one is dropped.
        """
        unwrapped = {}
        for dataset_name, sample in batch.items():
            data, coordinates, layout = sample.data, sample.coordinates, sample.layout
            if layout.time_in_grid:
                # Sparse payloads keep the batch as the outer list; unwrap the one sample.
                data = data[0]
                coordinates = coordinates[0]
                tabular_payload = {
                    "timedeltas": sample.timedeltas[0],
                    "boundaries": [(int(s.start), int(s.stop)) for s in sample.boundaries[0]],
                }
            else:
                batch_axis = sample.layout.axis("batch", ndim=data.ndim) if sample.layout.batch is not None else None
                if batch_axis is not None:
                    assert data.shape[batch_axis] == 1, (
                        f"Dataset {dataset_name!r}: expected a single sample to unwrap, got "
                        f"{data.shape[batch_axis]}."
                    )
                    data = data.squeeze(batch_axis)
                layout = layout.without_batch_dim()

            coords_deg = torch.rad2deg(coordinates)

            payload = {
                "data": data,
                "latitudes": coords_deg[:, 0],
                "longitudes": coords_deg[:, 1],
                "variables": sample.variables,
                "layout": layout.axis_names,
                **tabular_payload,
            }
            unwrapped[dataset_name] = payload

        return unwrapped

    def predict_step(
        self,
        x: dict[str, dict],
        target_forcing: dict[str, dict] = None,
        target_template: dict[str, dict] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Prediction step for the model.

        Parameters
        ----------
        x : dict[str, dict]
            Input data, one payload per dataset. A payload carries the data in
            model-input space and, optionally, the latlons (in degrees)
            and layout. If the coords are omitted, we fall back to the graph.
        target_forcing : dict[str, dict]
            Decoder conditioning, one payload per decoded dataset, holding the forcing
            variables at the output valid times.
        model_comm_group : Optional[ProcessGroup], optional
            Model communication group, specifies which GPUs work together.
        gather_out : bool, optional
            Whether to gather the output, by default True.
        **kwargs
            Additional prediction keyword arguments. Transport models require
            ``target_template`` here so sampling knows the output geometry.

        Returns
        -------
        dict[str, torch.Tensor]
            Predicted data.
        """
        assert target_template is not None, "target_template must be provided for prediction."

        # Convert to batch
        x = self.get_batch(x)
        target = self.get_batch(target_template)

        # Prepare kwargs for model's predict_step
        predict_kwargs = {
            "x": x,
            "target": target,
            "pre_processors": self.pre_processors,
            "post_processors": self.post_processors,
            "n_step_input": self.n_step_input,
            "model_comm_group": model_comm_group,
            "gather_out": gather_out,
        }

        # Add tendency processors if they exist
        if hasattr(self, "pre_processors_tendencies"):
            predict_kwargs["pre_processors_tendencies"] = self.pre_processors_tendencies
        if hasattr(self, "post_processors_tendencies"):
            predict_kwargs["post_processors_tendencies"] = self.post_processors_tendencies
        if getattr(self, "spatial_pre_processors", None):
            predict_kwargs["spatial_pre_processors"] = self.spatial_pre_processors

        pred_batch = self.model.predict_step(**predict_kwargs, **kwargs)
        return self.unwrap_batch(pred_batch)

    def _update_metadata(self) -> None:
        self.model.fill_metadata(self.metadata)

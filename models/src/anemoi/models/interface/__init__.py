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

from anemoi.models.data.batch import BOUNDARIES_META_KEY
from anemoi.models.data.batch import Batch
from anemoi.models.data.tensor_layout import TensorLayout
from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing import StepwiseProcessors
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
        n_step_input: int,
        n_step_output: int,
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
        self.data_layouts = {name: reader.layout.with_batch_dim() for name, reader in data_readers.items()}
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

        # Use the forward method of the model directly
        self.forward = self.model.forward

    @staticmethod
    def _as_payload(ds_data: torch.Tensor | dict) -> dict:
        """Normalise one dataset entry to the payload dict of the inference boundary.
        A bare tensor is accepted as a data-only payload.
        """
        return ds_data if isinstance(ds_data, dict) else {"data": ds_data}

    def _coordinates_in_radians(self, payload: dict, dataset_name: str) -> torch.Tensor:
        """Return one dataset's latlon tensor, in radians. Convenience method to liaise with anemoi-inference."""
        latitudes = payload.get("latitudes")
        longitudes = payload.get("longitudes")

        if latitudes is None or longitudes is None:
            return self.model._graph_data[dataset_name].x

        latitudes = torch.as_tensor(latitudes, dtype=torch.float32).reshape(-1)
        longitudes = torch.as_tensor(longitudes, dtype=torch.float32).reshape(-1)
        assert latitudes.shape == longitudes.shape, (
            f"Dataset {dataset_name!r}: latitudes {tuple(latitudes.shape)} and longitudes "
            f"{tuple(longitudes.shape)} must describe the same nodes."
        )
        return torch.deg2rad(torch.stack([latitudes, longitudes], dim=-1))

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

    def _is_tabular(self, dataset_name: str) -> bool:
        return self.data_layouts[dataset_name].time_in_grid

    def _prepare_data(self, payload: dict, dataset_name: str, variables: list[str]) -> dict:
        """Prepare the input data for the model.

        Parameters
        ----------
        payload : dict
            data, plus, optionally, latitudes / longitudes (in degrees), layout, and for tabular
            datasets: timedeltas (seconds) and boundaries ((start, stop) pairs, one per time slot).
        dataset_name : str
            The name of the dataset.
        variables : list[str]
            Names of the variables along the data tensor's variable axis, in order.

        Returns
        -------
        dict
            One sample payload.
        """
        data = payload.get("data")
        coordinates = self._coordinates_in_radians(payload, dataset_name)
        if data is not None:
            coordinates = coordinates.to(device=data.device)

        is_tabular = self._is_tabular(dataset_name)
        default_layout = ("grid", "variables") if is_tabular else ("time", "ensemble", "grid", "variables")
        layout_names = tuple(payload.get("layout") or default_layout)

        if data is not None:
            assert data.ndim == len(layout_names), (
                f"Dataset {dataset_name!r}: data tensor of shape {tuple(data.shape)} does not match "
                f"the specified layout {layout_names}."
            )
            var_axis = layout_names.index("variables")
            assert data.shape[var_axis] == len(variables), (
                f"Dataset {dataset_name!r}: data tensor carries {data.shape[var_axis]} variables "
                f"but {len(variables)} were expected ({variables})."
            )

        sample = {
            "data": data,
            "coordinates": coordinates,
            "variables": variables,
            "statistics": self._statistics_for(dataset_name, variables),
            # `time_in_grid` cannot be read off the axis names, so it comes from the dataset kind
            "layout": TensorLayout.from_tuple(*layout_names, time_in_grid=is_tabular),
            "grid_size": coordinates.shape[0],
            "metadata": {},
        }

        # Tabular payload
        timedeltas = payload.get("timedeltas")
        if timedeltas is not None:
            sample["timedeltas"] = torch.as_tensor(timedeltas, dtype=torch.float32).reshape(-1)
            if data is not None:
                sample["timedeltas"] = sample["timedeltas"].to(device=data.device)

        boundaries = payload.get("boundaries")
        if boundaries is not None:
            sample["metadata"][BOUNDARIES_META_KEY] = self._as_boundary_slices(boundaries, dataset_name)
        elif is_tabular:
            raise ValueError(f"Tabular dataset {dataset_name!r} needs boundaries!")

        shard_sizes = payload.get("shard_sizes")
        if shard_sizes is not None:
            sample["shard_sizes" if is_tabular else "grid_shard_sizes"] = shard_sizes

        return sample

    @staticmethod
    def _as_boundary_slices(boundaries, dataset_name: str) -> list[slice]:
        """Convert the boundary contract's ``(start, stop)`` pairs to the slices views use."""
        result = []
        for entry in boundaries:
            if isinstance(entry, slice):
                result.append(entry)
                continue
            try:
                start, stop = entry
            except (TypeError, ValueError) as err:
                msg = (
                    f"Dataset {dataset_name!r}: each boundary must be a (start, stop) pair or a slice, "
                    f"got {entry!r}."
                )
                raise ValueError(msg) from err
            result.append(slice(int(start), int(stop)))
        return result

    def prepare_input_spec(self, data: dict[str, torch.Tensor | dict]) -> dict[str, dict]:
        """Build the input specs. The caller supplies model-input-space variables per dataset."""
        return {
            dataset_name: self._prepare_data(
                self._as_payload(ds_data),
                dataset_name,
                self.data_indices[dataset_name].model.input.ordered_names,
            )
            for dataset_name, ds_data in data.items()
        }

    def prepare_target_spec(self, target: dict[str, torch.Tensor | dict]) -> dict[str, dict]:
        """Build the decoder conditioning specs -- the forcing variables at the output times."""
        assert target is not None, "predict_step requires a valid target argument"

        missing = [dataset_name for dataset_name in self.model.target_datasets if dataset_name not in target]
        if missing:
            msg = f"No target provided for decoded dataset(s) {missing}; got targets for {list(target)}."
            raise ValueError(msg)

        spec = {}
        for dataset_name in self.model.target_datasets:
            payload = self._as_payload(target[dataset_name])
            forcing_names = self._target_forcing_names(dataset_name)
            if payload.get("data") is None:
                raise ValueError(f"Target payload for dataset {dataset_name!r} carries no data.")
            spec[dataset_name] = self._prepare_data(payload, dataset_name, forcing_names)

        return spec

    def get_batch(self, data: dict[str, dict]) -> Batch:
        """Collate the per-dataset sample payloads into a single-sample Batch."""
        static_coord_datasets = frozenset(dataset_name for dataset_name in data if self.is_dataset_static[dataset_name])
        return Batch.collate(data, static_coord_datasets=static_coord_datasets)

    def unwrap_batch(self, batch: Batch) -> dict[str, dict]:
        """Convert a model output Batch back to plain per-dataset payload dicts.
        The coordinates are converted from radians to degrees, and the batch axis of one is dropped.
        """
        unwrapped = {}
        for dataset_name in batch.dataset_names:
            view = batch[dataset_name]
            is_tabular = self._is_tabular(dataset_name)

            data, coordinates, timedeltas = view.data, view.coordinates, view.timedeltas
            if is_tabular:
                # Sparse payloads keep the batch as the outer list; unwrap the one sample.
                data = data[0]
                coordinates = None if coordinates is None else coordinates[0]
                timedeltas = None if timedeltas is None else timedeltas[0]
                layout = view.layout
            else:
                batch_axis = view.layout.axis("batch", ndim=data.ndim) if view.layout.batch is not None else None
                if batch_axis is not None:
                    assert data.shape[batch_axis] == 1, (
                        f"Dataset {dataset_name!r}: expected a single sample to unwrap, got "
                        f"{data.shape[batch_axis]}."
                    )
                    data = data.squeeze(batch_axis)
                layout = view.layout.without_batch_dim()

            payload = {
                "data": data,
                "variables": view.variables,
                "layout": layout.axis_names,
            }
            if coordinates is not None:
                degrees = torch.rad2deg(coordinates)
                payload["latitudes"] = degrees[:, 0]
                payload["longitudes"] = degrees[:, 1]
            if timedeltas is not None:
                payload["timedeltas"] = timedeltas
            if view.boundaries is not None:
                sample_bounds = view.boundaries[0]
                payload["boundaries"] = [(int(s.start), int(s.stop)) for s in sample_bounds]

            unwrapped[dataset_name] = payload

        return unwrapped

    def predict_step(
        self,
        batch: dict[str, dict],
        target: dict[str, dict],
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Prediction step for the model.

        Parameters
        ----------
        batch : dict[str, dict]
            Input data, one payload per dataset. A payload carries the data in
            model-input space and, optionally, the latlons (in degrees)
            and layout. If the coords are omitted, we fall back to the graph.
        target : dict[str, dict]
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
        x = self.prepare_input_spec(batch)  # TODO: move to anemoi-inference
        target_spec = self.prepare_target_spec(target)  # TODO: move to anemoi-inference

        # Convert to batch
        x = self.get_batch(x)
        target = self.get_batch(target_spec)

        for dataset_name in x.dataset_names:
            view = x[dataset_name]
            if view.layout.batch is None and not isinstance(view.data, list):
                msg = (
                    f"Dataset {dataset_name!r} has neither a batch axis in its layout "
                    f"({view.layout!r}) nor a per-sample list payload, so the batch dimension is "
                    "missing."
                )
                raise ValueError(msg)

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

        return self.unwrap_batch(self.model.predict_step(**predict_kwargs, **kwargs))

    def _update_metadata(self) -> None:
        self.model.fill_metadata(self.metadata)

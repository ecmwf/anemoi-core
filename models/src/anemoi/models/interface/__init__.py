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

from anemoi.models.data.batch import Batch
from anemoi.models.data.sample import create_batched_struct
from anemoi.models.data.sources import BaseTemplate
from anemoi.models.data.sources import TabularSource
from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing.spatial import SpatialPreprocessor
from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.models.utils.graphs import build_graph
from anemoi.utils.config import DotDict


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
    data_types : dict[str, str]
        Kind of data of each dataset, as registered in the sample registry (``"gridded"``, ``"tabular"``).
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
        data_types: dict[str, str],
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
        self.data_types = data_types

        self._build_model()
        self._update_metadata()

    @staticmethod
    def _build_processors(
        processors_configs: dict,
        data_indices: dict,
    ) -> tuple[Processors, Processors]:
        """Build the (pre, post) processor pair for one dataset.

        Processors are stateless: they normalise with the statistics carried by each
        :class:`Source`. Tendencies go through the same pair, as sources carrying
        tendency statistics (see AnemoiTransportTendModelEncProcDec.compute_tendency).
        """
        processors = [
            [name, instantiate(processor, data_indices=data_indices)] for name, processor in processors_configs.items()
        ]
        return Processors(processors), Processors(processors, inverse=True)

    def _build_model(self) -> None:
        """Builds the model and pre- and post-processors."""
        # Multi-dataset mode: create processors for each dataset
        self.pre_processors = torch.nn.ModuleDict()
        self.post_processors = torch.nn.ModuleDict()

        data_config = get_multiple_datasets_config(self.config.data)
        for dataset_name in self.statistics.keys():
            pre, post = self._build_processors(
                data_config[dataset_name].processors,
                self.data_indices[dataset_name],
            )
            self.pre_processors[dataset_name] = pre
            self.post_processors[dataset_name] = post

        # Instantiate the model. The class is called directly rather than through hydra's instantiate,
        # which would convert the dynamic graph config (keyed by edge-type tuples) to an OmegaConf node.
        self.graph_data, dynamic_graph_config = self._build_graph()
        model_class = get_class(self.config.model.model._target_)
        self.model = model_class(
            model_config=self.config.model,
            static_graph_data=self.graph_data,
            dynamic_graph_config=dynamic_graph_config,
            data_indices=self.data_indices,
            statistics=self.statistics,
            n_step_input=self.n_step_input,
            n_step_output=self.n_step_output,
        )

        self._share_statistics_remap(self.model, self.pre_processors)

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

    @staticmethod
    def _share_statistics_remap(model: torch.nn.Module, pre_processors: torch.nn.ModuleDict) -> None:
        """Give the output boundings that normalise thresholds their dataset's normaliser remap.

        The boundings are built after the processors, so, as when the normaliser rewrote the shared
        statistics in place, they see the remap of every normaliser of their dataset; see
        :attr:`~anemoi.models.preprocessing.BasePreprocessor.shares_statistics_remap`.
        """
        for dataset_name, boundings in getattr(model, "boundings", {}).items():
            remap = {}
            if dataset_name in pre_processors:
                remap = dict(getattr(pre_processors[dataset_name], "statistics_remap", {}))
            for bounding in boundings.modules():
                if getattr(bounding, "uses_statistics_remap", False):
                    bounding.statistics_remap = dict(remap)

    def _build_graph(self) -> tuple[HeteroData, DotDict]:
        """Build the static graph and the dynamic graph configuration from ``config.graph``.

        The graph stored at ``system.input.graph`` is loaded unless ``graph.overwrite`` is set; the
        static nodes and edges of the config missing from it are then built. The dynamic part of
        the config (nodes and edges touching tabular datasets) is built by the model per batch.
        """
        system_input = (self.config.get("system") or {}).get("input") or {}
        return build_graph(
            DotDict(self.config.graph),
            is_dataset_static={name: data_type == "gridded" for name, data_type in self.data_types.items()},
            hidden_nodes_name=self.config.model.model.hidden_nodes_name,
            graph_path=system_input.get("graph"),
        )

    def apply_spatial_pre_processors(self, batch: Batch, model_comm_group: Optional[ProcessGroup] = None) -> Batch:
        """Project each dataset that has a spatial preprocessor onto its graph node set.

        Applied to raw (un-normalized) data. The projected source carries the coordinates
        of the dataset's graph nodes, which is the grid the encoder runs on.
        """
        for dataset_name, projector in self.spatial_pre_processors.items():
            if dataset_name in batch:
                projected = projector.project_source(
                    batch[dataset_name],
                    self.model.graph_geometry.coordinates(dataset_name),
                    model_comm_group=model_comm_group,
                )
                batch = batch.replace(dataset_name, projected)
        return batch

    def _statistics_for(self, dataset_name: str, variables: list[str]) -> dict:
        """Slice the checkpoint's data-space statistics down to the variables.
        Same alignment that is done for the model outputs in AnemoiModelEncProcDec._assemble_output.
        """
        name_to_index = self.data_indices[dataset_name].name_to_index
        positions = [name_to_index[name] for name in variables]
        return {name: values[positions] for name, values in self.statistics[dataset_name].items()}

    def _get_batch(self, payloads: dict[str, dict]) -> Batch:
        """Utility function to build a :class:`Batch` from raw per-dataset payload dicts."""
        return Batch.collate(
            {
                name: create_batched_struct(statistics=self._statistics_for(name, payload["variables"]), **payload)
                for name, payload in payloads.items()
            }
        )

    def _get_template(
        self,
        target_template: dict[str, dict],
        ensemble_size: int = 1,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> dict[str, BaseTemplate]:
        """Build a template batch with the given ensemble size, batch size, and number of target timesteps.

        A template has no data to take a device from, so its geometry (coordinates and, for tabular
        templates, timedeltas) is placed on ``device``, the device of the input batch.
        """
        geometry_keys = ("latitudes", "longitudes", "timedeltas")
        return {
            name: create_batched_struct(
                name=name,
                statistics=self._statistics_for(name, payload["variables"]),
                **{
                    key: (
                        torch.as_tensor(value, device=device)
                        if device is not None and key in geometry_keys and value is not None
                        else value
                    )
                    for key, value in payload.items()
                },
                ensemble_size=ensemble_size,
                batch_size=batch_size,
                time_size=self.n_step_output[name],
            )
            for name, payload in target_template.items()
        }

    def unwrap_batch(self, batch: Batch) -> dict[str, dict]:
        """Convert a model output Batch back to plain per-dataset payload dicts.
        The coordinates are converted from radians to degrees, and the batch axis of one is dropped.
        """
        unwrapped = {}
        for dataset_name, sample in batch.items():
            data, coordinates, layout = sample.data, sample.coordinates, sample.layout
            tabular_payload = {}
            if isinstance(sample, TabularSource):
                # Tabular payloads keep the batch as the outer list; unwrap the one sample.
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
        target_template: Optional[dict[str, dict]] = None,
        target_forcing: Optional[dict[str, dict]] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
        **kwargs,
    ) -> dict[str, dict]:
        """Prediction step for the model.

        Parameters
        ----------
        x : dict[str, dict]
            Input data in physical units, one payload per input dataset, as taken by
            :func:`~anemoi.models.data.sample.create_batched_struct`: the ``data`` (with
            the model's input variables, in model-input order), its ``data_type``,
            ``variables`` and ``layout``, and the ``latitudes`` and ``longitudes`` in
            degrees (required; there is no fallback to the graph). Tabular payloads also
            carry ``timedeltas`` and ``boundaries``.
        target_template : Optional[dict[str, dict]]
            Output geometry, one payload per dataset to decode; required despite the
            default. Like an input payload without ``data``, naming the variables to decode.
            Only the datasets given here are decoded.
        target_forcing : Optional[dict[str, dict]]
            Decoder conditioning, one payload per decoded dataset, holding the forcing
            variables at the output valid times. ``None`` for decoders that take no forcings.
        model_comm_group : Optional[ProcessGroup], optional
            Model communication group, specifies which GPUs work together.
        gather_out : bool, optional
            Whether to gather the output, by default True.
        **kwargs
            Additional keyword arguments for the model's ``predict_step``.

        Returns
        -------
        dict[str, dict]
            The prediction, one payload per decoded dataset.
        """
        assert target_template is not None, "target_template must be provided for prediction."
        if target_forcing is None:
            target_forcing = {}

        # Convert input data to batch
        x = self._get_batch(x)
        target_forcing = self._get_batch(target_forcing)

        target = self._get_template(
            target_template, ensemble_size=x.ensemble_size, batch_size=x.batch_size, device=x.device
        )

        # Prepare kwargs for model's predict_step
        predict_kwargs = {
            "x": x,
            "target_template": target,
            "target_forcing": target_forcing,
            "pre_processors": self.pre_processors,
            "post_processors": self.post_processors,
            "model_comm_group": model_comm_group,
            "gather_out": gather_out,
        }

        # Tendency models convert sampled tendencies back to states with these statistics.
        if self.statistics_tendencies is not None:
            predict_kwargs["statistics_tendencies"] = self.statistics_tendencies
        if getattr(self, "spatial_pre_processors", None):
            predict_kwargs["spatial_pre_processors"] = self.spatial_pre_processors

        pred_batch = self.model.predict_step(**predict_kwargs, **kwargs)
        return self.unwrap_batch(pred_batch)

    def _update_metadata(self) -> None:
        self.model.fill_metadata(self.metadata)

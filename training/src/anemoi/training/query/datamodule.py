# (C) Copyright 2026 Anemoi contributors.

"""Lightning data module for query-first forecasting."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from anemoi.graphs.create import GraphCreator
from anemoi.graphs.create import load_graph_from_file
from anemoi.graphs.create import validate_loaded_graph
from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.training.data.data_reader import NativeGridDataset
from anemoi.training.data.data_reader import create_dataset
from anemoi.training.query.catalogue import QueryCatalogue
from anemoi.training.query.dataset import QueryDataset
from anemoi.training.query.dataset import collate_query_batch
from anemoi.training.utils.worker_init import worker_init_func


class QueryDataModule(pl.LightningDataModule):
    """Open enabled archives independently and expose query-selected examples."""

    def __init__(self, config: Any, task: Any) -> None:
        super().__init__()
        self.config = config
        self.task = task
        self.epoch = 0
        if any(config.dataloader.batch_size[stage] != 1 for stage in ("training", "validation")):
            msg = "Query-based forecasting currently requires training and validation batch_size=1."
            raise ValueError(msg)
        stretched = config.model.query.stretched_grid
        if stretched.enabled:
            configured = get_multiple_datasets_config(config.dataloader.training)
            if stretched.context_source not in configured or not configured[stretched.context_source].enabled:
                msg = f"Stretched query geometry requires enabled global context source {stretched.context_source!r}."
                raise ValueError(
                    msg,
                )

    def _reader_config(self, stage: str) -> tuple[dict[str, Any], dict[str, float]]:
        configured = get_multiple_datasets_config(
            getattr(self.config.dataloader, stage),
        )
        readers = {}
        weights = {}
        for name, value in configured.items():
            if not value.get("enabled", False):
                continue
            source = value.dataset_config
            path = source.get("dataset") if hasattr(source, "get") else source
            if path == "":
                msg = f"Dataset {name!r} is enabled for {stage} but dataset_config.dataset is empty."
                raise ValueError(msg)
            reader_value = {key: item for key, item in value.items() if key not in {"enabled", "sampling_weight"}}
            readers[name] = reader_value
            weights[name] = float(value.get("sampling_weight", 1.0))
        if not readers:
            msg = f"No datasets are enabled for query-based {stage} data."
            raise ValueError(msg)
        return readers, weights

    @cached_property
    def train_readers(self) -> dict[str, NativeGridDataset]:
        config, _ = self._reader_config("training")
        readers = {name: create_dataset(value, task=self.task) for name, value in config.items()}
        unsupported = [name for name, reader in readers.items() if not isinstance(reader, NativeGridDataset)]
        if unsupported:
            msg = (
                "Query-first sampling currently supports native time-series datasets only; "
                f"trajectory readers need explicit initialisation/step handling: {unsupported}."
            )
            raise ValueError(
                msg,
            )
        return readers

    @cached_property
    def valid_readers(self) -> dict[str, NativeGridDataset]:
        config, _ = self._reader_config("validation")
        readers = {name: create_dataset(value, task=self.task) for name, value in config.items()}
        unsupported = [name for name, reader in readers.items() if not isinstance(reader, NativeGridDataset)]
        if unsupported:
            msg = f"Validation trajectory readers are not yet supported: {unsupported}."
            raise ValueError(msg)
        return readers

    @cached_property
    def catalogue(self) -> QueryCatalogue:
        catalogue = QueryCatalogue(self.train_readers, aliases=self.task.aliases)
        if self.task.reference_provenance not in catalogue.provenances:
            msg = f"reference_provenance {self.task.reference_provenance!r} is not an enabled dataset provenance."
            raise ValueError(
                msg,
            )
        return catalogue

    @cached_property
    def ds_train(self) -> QueryDataset:
        _, weights = self._reader_config("training")
        return QueryDataset(
            self.train_readers,
            self.catalogue,
            self.task,
            weights,
            self.task.samples_per_epoch,
            self.task.seed,
        )

    @cached_property
    def ds_valid(self) -> QueryDataset:
        _, weights = self._reader_config("validation")
        if list(self.valid_readers) != self.dataset_names:
            msg = "Validation must enable the same source geometries as training; use start/end to select its dates."
            raise ValueError(
                msg,
            )
        for name in self.dataset_names:
            training_reader = self.train_readers[name]
            validation_reader = self.valid_readers[name]
            same_coordinates = np.array_equal(
                training_reader.data.latitudes,
                validation_reader.data.latitudes,
            ) and np.array_equal(
                training_reader.data.longitudes,
                validation_reader.data.longitudes,
            )
            if training_reader.variables != validation_reader.variables or not same_coordinates:
                msg = f"Validation dataset {name!r} must have the same fields, order, and grid as training."
                raise ValueError(msg)
        return QueryDataset(
            self.valid_readers,
            self.catalogue,
            self.task,
            weights,
            self.task.validation_samples,
            self.task.validation_seed,
        )

    @property
    def dataset_names(self) -> list[str]:
        return list(self.train_readers)

    @property
    def data_indices(self) -> dict:
        return dict.fromkeys(self.dataset_names)

    @property
    def statistics(self) -> dict:
        return {name: reader.statistics for name, reader in self.train_readers.items()}

    @property
    def statistics_tendencies(self) -> None:
        return None

    @property
    def metadata(self) -> dict:
        return {name: reader.metadata for name, reader in self.train_readers.items()}

    @property
    def supporting_arrays(self) -> dict:
        return {name: reader.supporting_arrays for name, reader in self.train_readers.items()}

    @cached_property
    def graph_data(self) -> Any:
        graph_path = self.config.system.input.graph
        save_path = Path(graph_path) if graph_path else None
        stretched = self.config.model.query.stretched_grid
        expected_geometry = {
            "datasets": self.dataset_names,
            "source_geometries": {
                name: {
                    "grid_size": self.catalogue.datasets[name]["grid_size"],
                    "latitude_bounds_degrees": self.catalogue.datasets[name]["latitude_bounds_degrees"],
                    "longitude_bounds_degrees": self.catalogue.datasets[name]["longitude_bounds_degrees"],
                    "coordinate_sha256": self.catalogue.datasets[name]["coordinate_sha256"],
                }
                for name in self.dataset_names
            },
            "stretched": bool(stretched.enabled),
            "context_source": stretched.context_source,
            "area": list(stretched.area),
            "global_resolution": stretched.global_resolution,
            "local_resolution": stretched.local_resolution,
            "margin_radius_km": stretched.margin_radius_km,
            "input_context_margin_degrees": self.task.input_context_margin_degrees,
            "global_context_sources": list(self.task.global_context_sources),
        }
        if save_path and save_path.exists() and not self.config.graph.overwrite:
            graph = load_graph_from_file(save_path)
            validate_loaded_graph(graph, [*self.dataset_names, "hidden"])
            if getattr(graph, "query_geometry", None) != expected_geometry:
                msg = (
                    f"Cached graph {save_path} has different query geometry. "
                    "Set graph.overwrite=true or use a new path."
                )
                raise ValueError(
                    msg,
                )
            return graph

        nodes = {}
        edges = []
        attributes = {
            "nodes": {},
            "edges": {
                "edge_length": {
                    "_target_": "anemoi.graphs.edges.attributes.EdgeLength",
                    "norm": "unit-max",
                },
            },
        }
        hidden_attributes = {}
        source_coverage_attributes = {}
        for index, name in enumerate(self.dataset_names):
            if name in self.task.global_context_sources:
                continue
            bounds = self.catalogue.datasets[name]
            south, north = bounds["latitude_bounds_degrees"]
            west, east = bounds["longitude_bounds_degrees"]
            margin = self.task.input_context_margin_degrees
            attribute_name = f"query_source_coverage_{index}"
            source_coverage_attributes[name] = attribute_name
            hidden_attributes[attribute_name] = {
                "_target_": "anemoi.graphs.nodes.attributes.GeographicAreaMask",
                "area": [west - margin, south - margin, east + margin, north + margin],
            }
        for name, reader_config in self._reader_config("training")[0].items():
            dataset_config = {
                key: value
                for key, value in OmegaConf.to_container(
                    reader_config["dataset_config"],
                    resolve=True,
                ).items()
                if value is not None
            }
            node_attributes = {}
            if stretched.enabled and name == stretched.context_source:
                node_attributes["query_area_mask"] = {
                    "_target_": "anemoi.graphs.nodes.attributes.GeographicAreaMask",
                    "area": list(stretched.area),
                }
            nodes[name] = {
                "node_builder": {
                    "_target_": "anemoi.graphs.nodes.AnemoiDatasetNodes",
                    "dataset": dataset_config,
                },
                "attributes": node_attributes,
            }
            for source, target, neighbours in (
                (name, "hidden", self.config.model.query.encoder_neighbours),
                ("hidden", name, self.config.model.query.decoder_neighbours),
            ):
                edges.append(
                    {
                        "source_name": source,
                        "target_name": target,
                        "edge_builders": [
                            {
                                "_target_": "anemoi.graphs.edges.KNNEdges",
                                "num_nearest_neighbours": neighbours,
                                "source_mask_attr_name": None,
                                "target_mask_attr_name": (
                                    source_coverage_attributes[name]
                                    if target == "hidden" and name in source_coverage_attributes
                                    else None
                                ),
                            },
                        ],
                        "attributes": attributes["edges"],
                    },
                )
        if stretched.enabled:
            hidden_builder = {
                "_target_": "anemoi.graphs.nodes.StretchedTriNodes",
                "global_resolution": stretched.global_resolution,
                "lam_resolution": stretched.local_resolution,
                "reference_node_name": stretched.context_source,
                "mask_attr_name": "query_area_mask",
                "margin_radius_km": stretched.margin_radius_km,
            }
            processor_resolution = stretched.local_resolution
        else:
            hidden_builder = {
                "_target_": "anemoi.graphs.nodes.TriNodes",
                "resolution": self.config.model.query.processor_mesh_resolution,
            }
            processor_resolution = self.config.model.query.processor_mesh_resolution
        nodes["hidden"] = {
            "node_builder": hidden_builder,
            "attributes": hidden_attributes,
        }
        edges.append(
            {
                "source_name": "hidden",
                "target_name": "hidden",
                "edge_builders": [
                    {
                        "_target_": "anemoi.graphs.edges.MultiScaleEdges",
                        "x_hops": 1,
                        "scale_resolutions": processor_resolution,
                        "source_mask_attr_name": None,
                        "target_mask_attr_name": None,
                    },
                ],
                "attributes": attributes["edges"],
            },
        )
        graph_config = OmegaConf.create(
            {"nodes": nodes, "edges": edges, "post_processors": []},
        )
        creator = GraphCreator(graph_config)
        graph = creator.create()
        graph.query_geometry = expected_geometry
        if save_path:
            creator.save(graph, save_path, overwrite=self.config.graph.overwrite)
        return graph

    def _dataloader(self, dataset: QueryDataset, stage: str) -> DataLoader:
        workers = self.config.dataloader.num_workers[stage]
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=workers,
            pin_memory=self.config.dataloader.pin_memory,
            worker_init_fn=worker_init_func,
            persistent_workers=False,
            prefetch_factor=self.config.dataloader.prefetch_factor if workers else None,
            collate_fn=collate_query_batch,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.ds_train, "training")

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.ds_valid, "validation")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        if "ds_train" in self.__dict__:
            self.ds_train.set_epoch(epoch)

    def sync_dataset_state(self) -> None:
        self.set_epoch(self.epoch)

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.set_epoch(state_dict["epoch"])

    def fill_metadata(self, metadata: dict) -> None:
        metadata["dataset"] = self.metadata
        metadata["data_indices"] = None
        metadata["query_catalogue"] = self.catalogue.snapshot()
        metadata["metadata_inference"]["dataset_names"] = self.dataset_names
        metadata["metadata_inference"]["query_catalogue"] = self.catalogue.snapshot()

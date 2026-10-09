# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.interface import AnemoiModelInterface
from anemoi.models.preprocessing.imputer import ConstantImputer
from anemoi.models.preprocessing.normalizer import InputNormalizer
from anemoi.models.preprocessing.postprocessor import Postprocessor

NORMALIZER_CONFIG = {
    "_target_": "anemoi.models.preprocessing.normalizer.InputNormalizer",
    "config": {"default": "mean-std"},
}
IMPUTER_CONFIG = {
    "_target_": "anemoi.models.preprocessing.imputer.ConstantImputer",
    "config": {"default": 0.0},
}
POSTPROCESSOR_CONFIG = {
    "_target_": "anemoi.models.preprocessing.postprocessor.Postprocessor",
    "config": {"default": "none", "relu": ["a"]},
}


def _bare_interface(config, graph: HeteroData) -> AnemoiModelInterface:
    model_interface = AnemoiModelInterface.__new__(AnemoiModelInterface)
    torch.nn.Module.__init__(model_interface)
    model_interface.config = config
    model_interface.graph_data = graph
    model_interface.statistics = {}
    model_interface.statistics_tendencies = None
    model_interface.residual_statistics = None
    model_interface.data_indices = {}
    model_interface.n_step_input = 1
    model_interface.n_step_output = 1
    return model_interface


def test_interface_passes_complete_graph_to_spatial_preprocessor() -> None:
    graph = HeteroData()
    graph["source"].num_nodes = 2
    graph["projected"].num_nodes = 1
    graph["source", "to", "projected"].edge_index = torch.tensor([[0, 1], [0, 0]])

    config = OmegaConf.create(
        {
            "data": {
                "datasets": {
                    "projected": {
                        "processors": {},
                        "spatial_processor": {
                            "_target_": "anemoi.models.preprocessing.cross_grid_projector.CrossGridProjector",
                            "edges_name": ["source", "to", "projected"],
                        },
                    }
                },
            },
            "model": {"model": {"_target_": "torch.nn.Identity"}},
        }
    )
    model_interface = _bare_interface(config, graph)

    model_interface._build_model()

    projector = model_interface.spatial_pre_processors["projected"]
    projected, grid_shard_sizes = projector(torch.tensor([[[[[1.0], [3.0]]]]]))

    assert grid_shard_sizes is None
    torch.testing.assert_close(projected, torch.tensor([[[[[2.0]]]]]))


def _data_indices_and_statistics() -> tuple[IndexCollection, dict]:
    data_config = OmegaConf.create({"forcing": [], "diagnostic": []})
    data_indices = IndexCollection(data_config=data_config, name_to_index={"a": 0, "b": 1})
    statistics = {
        "mean": np.array([1.0, 2.0]),
        "stdev": np.array([0.5, 2.0]),
        "minimum": np.array([0.0, 0.0]),
        "maximum": np.array([3.0, 4.0]),
    }
    return data_indices, statistics


def _processor_types(processors) -> list[type]:
    return [type(processor) for processor in processors.processors.values()]


def test_residual_processors_keep_only_the_normalizer() -> None:
    model_interface = _bare_interface(OmegaConf.create({}), HeteroData())
    data_indices, statistics = _data_indices_and_statistics()
    processors_configs = OmegaConf.create(
        {"imputer": IMPUTER_CONFIG, "normalizer": NORMALIZER_CONFIG, "postprocessor": POSTPROCESSOR_CONFIG}
    )

    pre, post, _, _, pre_residual, post_residual = model_interface._build_processors_for_dataset(
        processors_configs,
        statistics,
        data_indices,
        residual_statistics=statistics,
        dataset_name="out_hres",
    )

    assert _processor_types(pre) == [ConstantImputer, InputNormalizer, Postprocessor]
    assert _processor_types(post) == [Postprocessor, InputNormalizer, ConstantImputer]
    assert _processor_types(pre_residual) == [InputNormalizer]
    assert _processor_types(post_residual) == [InputNormalizer]


def test_residual_processors_reject_processors_other_than_normalizer_imputer_and_postprocessor() -> None:
    model_interface = _bare_interface(OmegaConf.create({}), HeteroData())
    data_indices, statistics = _data_indices_and_statistics()
    processors_configs = OmegaConf.create(
        {
            "normalizer": NORMALIZER_CONFIG,
            "other": {"_target_": "anemoi.models.preprocessing.remapper.Remapper", "config": {}},
        }
    )

    with pytest.raises(ValueError, match="out_hres.*'other'"):
        model_interface._build_residual_processors(
            processors_configs, data_indices, statistics, dataset_name="out_hres"
        )


def test_residual_processors_require_a_normalizer() -> None:
    model_interface = _bare_interface(OmegaConf.create({}), HeteroData())
    data_indices, statistics = _data_indices_and_statistics()
    processors_configs = OmegaConf.create({"imputer": IMPUTER_CONFIG})

    with pytest.raises(ValueError, match="out_hres.*normaliser"):
        model_interface._build_residual_processors(
            processors_configs, data_indices, statistics, dataset_name="out_hres"
        )


def test_tendency_processors_keep_every_configured_processor() -> None:
    model_interface = _bare_interface(OmegaConf.create({}), HeteroData())
    data_indices, statistics = _data_indices_and_statistics()
    processors_configs = OmegaConf.create({"imputer": IMPUTER_CONFIG, "normalizer": NORMALIZER_CONFIG})

    _, _, pre_tendencies, post_tendencies, pre_residual, post_residual = model_interface._build_processors_for_dataset(
        processors_configs,
        statistics,
        data_indices,
        statistics_tendencies=statistics,
        dataset_name="data",
    )

    assert _processor_types(pre_tendencies) == [ConstantImputer, InputNormalizer]
    assert _processor_types(post_tendencies) == [InputNormalizer, ConstantImputer]
    assert pre_residual is None and post_residual is None

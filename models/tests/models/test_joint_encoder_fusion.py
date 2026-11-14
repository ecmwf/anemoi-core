# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Encoders with ``dataset_fusing_strategy: joint``.

The spatial downscaler setup of ``test_transport_tendency_reference``: ``lres`` and ``hres`` on the
same grid, ``hres`` predicted as residuals against ``lres``, here with one joint encoder for both.
"""

import pytest
import torch

from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.models.encoder_processor_decoder import EncoderSource
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportTendModelEncProcDec
from anemoi.models.utils.graphs import build_graph
from anemoi.utils.config import DotDict
from tests.models.test_transport_input_only_datasets import _points_on_sphere
from tests.models.test_transport_tendency_reference import NUM_GRID_POINTS
from tests.models.test_transport_tendency_reference import STATISTICS
from tests.models.test_transport_tendency_reference import TENDENCY_STATISTICS
from tests.models.test_transport_tendency_reference import _data_indices
from tests.models.test_transport_tendency_reference import _graph_config
from tests.models.test_transport_tendency_reference import _model_config
from tests.models.test_transport_tendency_reference import _processors
from tests.models.test_transport_tendency_reference import _synthetic_batches


@pytest.fixture(autouse=True)
def _graphs_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build the graphs on the CPU, where the model and the data are, also on a GPU node."""
    monkeypatch.setenv("ANEMOI_GRAPHS_FORCE_CPU", "1")


def _joint_model(shift_hres_nodes: float = 0.0, hres_first: bool = False) -> AnemoiTransportTendModelEncProcDec:
    """The downscaler model with one joint encoder over ``lres`` and ``hres``.

    ``shift_hres_nodes`` moves the ``hres`` nodes away from the ``lres`` ones (in radians).
    ``hres_first`` lists ``hres`` before ``lres`` in the data indices.
    """
    latitudes, longitudes = _points_on_sphere(NUM_GRID_POINTS, torch.Generator().manual_seed(0))
    torch.manual_seed(1)
    static_graph_data, dynamic_graph_config = build_graph(
        DotDict(_graph_config(latitudes, longitudes)),
        is_dataset_static={"lres": True, "hres": True},
        hidden_nodes_name="hidden",
    )
    static_graph_data["hres"].x = static_graph_data["hres"].x + shift_hres_nodes
    config = _model_config()
    mapper = config.encoders.lres.mapper
    config.encoders = {
        "joint": {"source_datasets": ["lres", "hres"], "dataset_fusing_strategy": "joint", "mapper": mapper}
    }
    model = AnemoiTransportTendModelEncProcDec(
        model_config=config,
        static_graph_data=static_graph_data,
        dynamic_graph_config=dynamic_graph_config,
        data_indices=dict(reversed(_data_indices().items())) if hres_first else _data_indices(),
        statistics=STATISTICS,
        n_step_input={"lres": 1, "hres": 1},
        n_step_output={"lres": 1, "hres": 1},
    )
    model.test_grid_coordinates = static_graph_data["hres"].x.float()
    return model


@pytest.mark.parametrize("hres_first", [False, True], ids=["lres_first", "hres_first"])
def test_joint_encoder_has_one_latent_on_the_edges_of_its_first_dataset(hres_first: bool) -> None:
    model = _joint_model(hres_first=hres_first)

    assert list(model.encoder_graph_provider) == ["lres"]
    assert model.latent_aggregator.source_names == ("lres",)
    assert model.encoded_input_dim("hres") == model.input_dim["lres"] + model.input_dim["hres"]
    assert model.encoded_input_dim("lres") == model.encoded_input_dim("hres")


def test_joint_encoder_joins_the_assembled_inputs_node_by_node() -> None:
    model = _joint_model()
    batches = _synthetic_batches(model)
    pre_processors, post_processors = _processors()
    assembled, mapper_inputs = {}, []
    encoder_source_from_rows = model._encoder_source_from_rows

    def record_rows(dataset_name, rows, **kwargs):
        source = encoder_source_from_rows(dataset_name, rows, **kwargs)
        assembled[dataset_name] = source.x_data_latent.detach().clone()
        return source

    model._encoder_source_from_rows = record_rows
    model.encoder["joint"].register_forward_pre_hook(lambda _module, args: mapper_inputs.append(args[0][0].detach()))

    out = model.predict_step(
        batches["x"],
        target_template={"hres": batches["state"]["hres"].template()},
        pre_processors=pre_processors,
        post_processors=post_processors,
        target_forcing=batches["target_forcing"],
        statistics_tendencies=TENDENCY_STATISTICS,
    )

    assert torch.isfinite(out["hres"].data).all()
    assert list(assembled) == ["lres", "hres"]
    # One mapper pass per sampling evaluation, each on the lres rows followed by the hres rows.
    assert mapper_inputs
    expected_width = model.input_dim["lres"] + model.input_dim["hres"]
    assert all(rows.shape[-1] == expected_width for rows in mapper_inputs)
    torch.testing.assert_close(mapper_inputs[-1], torch.cat([assembled["lres"], assembled["hres"]], dim=-1))


def test_joint_encoder_needs_the_same_nodes() -> None:
    with pytest.raises(ValueError, match="the nodes of 'hres' .* differ from those of 'lres'"):
        _joint_model(shift_hres_nodes=1e-3)


def _source(dataset_name: str, rows: int, shard_sizes: list[int] | None = None) -> EncoderSource:
    return EncoderSource(
        dataset_name=dataset_name,
        x_data_latent=torch.zeros(rows, 3),
        x_skip=None,
        edge_attr=None,
        edge_index=None,
        shard_info=BipartiteGraphShardInfo(src_nodes=shard_sizes),
    )


@pytest.mark.parametrize(
    ("sources", "message"),
    [
        ([("lres", 20, None)], r"needs all its source datasets \['lres', 'hres'\] in every batch"),
        ([("lres", 20, None), ("hres", 19, None)], "must have the same nodes"),
        # The rows of one rank under model sharding, split differently across the ranks.
        ([("lres", 20, [20, 20]), ("hres", 20, [20, 21])], "must have the same nodes"),
    ],
)
def test_joint_encoder_rejects_batches_it_cannot_join(
    sources: list[tuple[str, int, list[int] | None]], message: str
) -> None:
    model = _joint_model()

    with pytest.raises(ValueError, match=message):
        model._encode_sources(
            "joint",
            [_source(*source) for source in sources],
            x_hidden_latent=torch.zeros(1, 3),
            x_data_latent_dict={},
            batch_size=1,
        )

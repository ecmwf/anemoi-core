# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import inspect
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import HeteroData

import anemoi.models.models.transport_encoder_processor_decoder as transport_model_module
from anemoi.graphs.nodes.attributes import Timedeltas
from anemoi.models.data import Batch
from anemoi.models.data.layout import TensorLayout
from anemoi.models.data.sources import BaseTemplate
from anemoi.models.data.sources import GriddedSource
from anemoi.models.data.sources import GriddedTemplate
from anemoi.models.data.sources import TabularSource
from anemoi.models.layers.aggregator import SumAggregator
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportModelEncProcDec
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportTendModelEncProcDec
from anemoi.models.samplers import transport_samplers
from anemoi.models.transport import EDMDiffusionModelObjective
from anemoi.models.transport import EdmSettings
from anemoi.models.transport import StochasticInterpolantModelObjective
from anemoi.models.transport import TransportSourceBuilder
from anemoi.models.transport import TransportSourceRequest
from anemoi.models.transport import TransportSourceSettings
from anemoi.models.transport import schedules
from tests.batch_builders import build_batch


class IdentityProcessor(torch.nn.Module):
    def forward(self, x: torch.Tensor, in_place: bool = True, inverse: bool = False, **kwargs):
        del inverse, kwargs
        if not in_place:
            x = x.clone()
        return x


def _transport_model_stub() -> AnemoiTransportModelEncProcDec:
    model = AnemoiTransportModelEncProcDec.__new__(AnemoiTransportModelEncProcDec)
    torch.nn.Module.__init__(model)
    model.transport_source = TransportSourceBuilder()
    model.dynamic_node_attributes = {}
    return model


class _EmptyNodeAttributes:
    num_nodes = {"hidden": 5}

    def __contains__(self, _dataset_name: str) -> bool:
        return False

    def __call__(self, _dataset_name: str, batch_size: int) -> torch.Tensor | None:
        del batch_size
        return None


class _GraphProvider:
    def get_edges(self, **_kwargs):
        return torch.zeros(1, 1), torch.zeros(2, 1, dtype=torch.long), None


def _data_indices(input_names: tuple[str, ...], output_names: tuple[str, ...]) -> SimpleNamespace:
    input_positions = {name: idx for idx, name in enumerate(input_names)}
    all_names = tuple(dict.fromkeys((*input_names, *output_names)))

    def positions_for_names(names: tuple[str, ...]) -> list[int]:
        try:
            return [input_positions[name] for name in names]
        except KeyError as exc:
            raise ValueError(f"missing variables: {names}") from exc

    return SimpleNamespace(
        name_to_index={name: idx for idx, name in enumerate(all_names)},
        model=SimpleNamespace(
            input=SimpleNamespace(ordered_names=input_names, positions_for_names=positions_for_names),
            output=SimpleNamespace(ordered_names=output_names),
        ),
    )


def _configure_sampling_model(
    model: AnemoiTransportModelEncProcDec,
    specs: dict[str, tuple[int, int, int]],
) -> None:
    """Attach the metadata needed by ``_sampling_template`` to a lightweight model stub.

    ``specs`` maps dataset name to ``(num_input_channels, num_output_channels, grid_size)``.
    """
    model.data_indices = {}
    model.statistics = {}
    model.is_dataset_static = {}
    model.target_datasets = list(specs)
    model._graph_data = HeteroData()
    for dataset_name, (num_inputs, num_outputs, grid_size) in specs.items():
        input_names = tuple(f"in_{idx}" for idx in range(num_inputs))
        output_names = tuple(f"out_{idx}" for idx in range(num_outputs))
        model.data_indices[dataset_name] = _data_indices(input_names, output_names)
        model.statistics[dataset_name] = {
            "mean": torch.zeros(num_inputs + num_outputs),
            "stdev": torch.ones(num_inputs + num_outputs),
        }
        model.is_dataset_static[dataset_name] = True
        model._graph_data[dataset_name].x = torch.zeros(grid_size, 2)


GRIDDED_LAYOUT = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)


def _model_batch(
    model: AnemoiTransportModelEncProcDec,
    data: dict[str, torch.Tensor],
    variable_space: str,
) -> Batch:
    """Gridded batch on the graph's grid nodes, described with the model's variables of ``variable_space``."""
    return Batch(
        {
            name: GriddedSource(
                name=name,
                variables=model._sampling_variables(name, variable_space),
                layout=GRIDDED_LAYOUT,
                statistics=model._sampling_statistics(name, variable_space),
                data=tensor,
                coordinates=model._graph_data[name].x,
            )
            for name, tensor in data.items()
        },
    )


def _sampling_batch(model: AnemoiTransportModelEncProcDec, data: dict[str, torch.Tensor]) -> Batch:
    return _model_batch(model, data, "input")


def _templates(batch: Batch) -> dict[str, BaseTemplate]:
    """The batch's sources without their data, as the interface and ``get_targets`` provide them."""
    return {name: source.template() for name, source in batch.items()}


def _target_template(model: AnemoiTransportModelEncProcDec, data: dict[str, torch.Tensor]) -> dict[str, BaseTemplate]:
    template_data = {
        name: torch.empty(
            sample.shape[0],
            model.n_step_output[name],
            sample.shape[2],
            sample.shape[-2],
            len(model.data_indices[name].model.output.ordered_names),
            device=sample.device,
            dtype=sample.dtype,
        )
        for name, sample in data.items()
    }
    return _templates(_model_batch(model, template_data, "output"))


def _sparse_batch(
    *,
    name: str,
    data_shapes: list[tuple[int, int]],
    variables: list[str],
) -> Batch:
    data = [torch.zeros(shape, dtype=torch.float32) for shape in data_shapes]
    coordinates = [torch.full((shape[0], 2), float(index)) for index, shape in enumerate(data_shapes)]
    return build_batch(
        data={name: data},
        coordinates={name: coordinates},
        timedeltas={name: [torch.zeros(shape[0]) for shape in data_shapes]},
        boundaries={name: [(slice(0, shape[0]),) for shape in data_shapes]},
        layouts={name: TensorLayout(grid=0, variables=1)},
        variables={name: variables},
        statistics={name: {}},
    )


def _sparse_target_template(
    *,
    name: str,
    node_counts: list[int],
    variables: list[str],
) -> dict[str, BaseTemplate]:
    return _templates(
        _sparse_batch(
            name=name,
            data_shapes=[(node_count, len(variables)) for node_count in node_counts],
            variables=variables,
        ),
    )


def test_transport_conditioning_embedding_uses_compact_condition_width() -> None:
    model = _transport_model_stub()
    model._graph_name_hidden = "hidden"
    model._graph_data = {"data": SimpleNamespace(num_nodes=4), "hidden": SimpleNamespace(num_nodes=5)}
    cond_dim = 8
    model._embed_noise_conditioning = lambda sigma: torch.ones(
        (*sigma.shape[:-1], cond_dim),
        device=sigma.device,
        dtype=sigma.dtype,
    )

    x = build_batch(
        data={"data": torch.empty(2, 2, 3, 4, 1)},
        coordinates={"data": torch.zeros(4, 2)},
        metadata={"static_coords": frozenset({"data"})},
        layouts={"data": TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)},
        variables={"data": ["a"]},
        statistics={"data": {}},
    )
    condition = {"data": torch.zeros(2, 1, 3, 1, 1)}

    fwd_mapper_kwargs, processor_kwargs, bwd_mapper_kwargs = model._build_conditioning_kwargs(x, x, condition)

    data_cond, hidden_cond = fwd_mapper_kwargs["data"]["cond"]
    hidden_back_cond, data_back_cond = bwd_mapper_kwargs["data"]["cond"]
    assert data_cond.shape == data_back_cond.shape == (2 * 3 * 4, cond_dim)
    assert hidden_cond.shape == hidden_back_cond.shape == processor_kwargs["cond"].shape == (2 * 3 * 5, cond_dim)


@pytest.mark.parametrize(
    ("local_grid", "shard_sizes"),
    [
        pytest.param(4, None, id="whole_view"),
        pytest.param(2, [2, 2], id="sharded_view"),
    ],
)
def test_transport_conditioning_is_split_like_the_nodes_it_conditions(
    monkeypatch: pytest.MonkeyPatch,
    local_grid: int,
    shard_sizes: list[int] | None,
) -> None:
    """Both mappers receive conditioning in the same layout as their input node features."""
    model = _transport_model_stub()
    model._graph_name_hidden = "hidden"
    model._graph_data = {"data": SimpleNamespace(num_nodes=4), "hidden": SimpleNamespace(num_nodes=6)}
    model._embed_noise_conditioning = lambda sigma: torch.ones((*sigma.shape[:-1], 2), dtype=sigma.dtype)
    # A model group of two ranks, seen from rank 0: every sharded tensor keeps its first shard.
    sharded_row_counts = []

    def shard_first(tensor, dim, sizes, model_comm_group):
        sharded_row_counts.append(tensor.shape[dim])
        return tensor.narrow(dim, 0, sizes[0])

    monkeypatch.setattr(
        transport_model_module,
        "get_shard_sizes",
        lambda tensor, dim, model_comm_group=None: [tensor.shape[dim] // 2] * 2,
    )
    monkeypatch.setattr(transport_model_module, "shard_tensor", shard_first)

    x = build_batch(
        data={"data": torch.empty(1, 1, 1, local_grid, 1)},
        coordinates={"data": torch.zeros(local_grid, 2)},
        metadata={"static_coords": frozenset({"data"})},
        shard_sizes={"data": shard_sizes},
        layouts={"data": TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)},
        variables={"data": ["a"]},
        statistics={"data": {}},
    )
    condition = {"data": torch.zeros(1, 1, 1, 1, 1)}

    fwd_mapper_kwargs, processor_kwargs, bwd_mapper_kwargs = model._build_conditioning_kwargs(
        x,
        x,
        condition,
        model_comm_group=object(),
    )

    encoder_data_cond, encoder_hidden_cond = fwd_mapper_kwargs["data"]["cond"]
    decoder_hidden_cond, decoder_data_cond = bwd_mapper_kwargs["data"]["cond"]
    assert encoder_data_cond.shape[0] == decoder_data_cond.shape[0] == local_grid
    assert encoder_hidden_cond.shape[0] == decoder_hidden_cond.shape[0] == processor_kwargs["cond"].shape[0] == 3
    assert set(sharded_row_counts) == ({6} if shard_sizes is None else {6, 4})


def _obs_source(node_counts: list[int], n_variables: int, value: float, coordinate_offset: float) -> TabularSource:
    """A one-window tabular source with ``node_counts[s]`` points in sample ``s``, all set to ``value``."""
    return TabularSource(
        name="obs",
        data=[torch.full((count, n_variables), value) for count in node_counts],
        coordinates=[
            coordinate_offset + 0.01 * torch.arange(count, dtype=torch.float32)[:, None].expand(count, 2)
            for count in node_counts
        ],
        variables=[f"v{i}" for i in range(n_variables)],
        statistics={},
        layout=TensorLayout(grid=0, variables=1),
        boundaries=[(slice(0, count),) for count in node_counts],
        timedeltas=[torch.full((count,), coordinate_offset) for count in node_counts],
    )


def test_transport_conditioning_covers_history_and_target_nodes_of_sparse_data() -> None:
    """The encoder is conditioned on each sample's history and target nodes; the decoder on the target nodes."""
    model = _transport_model_stub()
    model._graph_name_hidden = "hidden"
    model._graph_data = {"hidden": SimpleNamespace(num_nodes=5)}
    cond_dim = 6
    model._embed_noise_conditioning = (
        lambda sigma: torch.arange(sigma.shape[0], dtype=sigma.dtype).view(-1, 1, 1).expand(*sigma.shape[:-1], cond_dim)
    )
    x = Batch({"obs": _obs_source([3, 1], n_variables=1, value=0.0, coordinate_offset=-1.0)})
    target = Batch({"obs": _obs_source([2, 4], n_variables=1, value=0.0, coordinate_offset=1.0)})
    condition = {"obs": torch.zeros(2, 1, 1, 1, 1)}

    fwd_mapper_kwargs, processor_kwargs, bwd_mapper_kwargs = model._build_conditioning_kwargs(x, target, condition)

    encoder_cond, hidden_cond = fwd_mapper_kwargs["obs"]["cond"]
    hidden_back_cond, decoder_cond = bwd_mapper_kwargs["obs"]["cond"]
    # Sample 0: 3 history and 2 target nodes; sample 1: 1 history and 4 target nodes.
    torch.testing.assert_close(encoder_cond[:, 0], torch.tensor([0.0] * 5 + [1.0] * 5))
    torch.testing.assert_close(decoder_cond[:, 0], torch.tensor([0.0] * 2 + [1.0] * 4))
    assert hidden_cond.shape == hidden_back_cond.shape == processor_kwargs["cond"].shape == (2 * 5, cond_dim)


def test_transport_assemble_input_stacks_history_and_noisy_target_rows_of_sparse_data() -> None:
    """Each sample's history rows come first, then its noisy-target rows, with a flag on the target rows."""
    model = _transport_model_stub()
    model.node_attributes = _EmptyNodeAttributes()
    x = _obs_source([2, 1], n_variables=2, value=1.0, coordinate_offset=-1.0)
    y_noised = _obs_source([3, 2], n_variables=1, value=5.0, coordinate_offset=1.0)

    data_coords, x_data_latent, x_skip, shard_sizes, batch_sizes, timedeltas = model._assemble_transport_input(
        x, y_noised, batch_size=2, dataset_name="obs"
    )

    history_row = torch.tensor([1.0, 1.0, 0.0])
    target_row = torch.tensor([0.0, 0.0, 5.0])
    expected_values = torch.stack([history_row] * 2 + [target_row] * 3 + [history_row] + [target_row] * 2)
    expected_flag = torch.tensor([0.0] * 2 + [1.0] * 3 + [0.0] + [1.0] * 2)
    expected_coords = torch.cat(
        [x.coordinates[0], y_noised.coordinates[0], x.coordinates[1], y_noised.coordinates[1]],
    )
    assert x_skip is None
    assert shard_sizes is None
    assert batch_sizes == (5, 3)
    assert x_data_latent.shape == (8, 2 + 1 + 4 + 1)
    torch.testing.assert_close(x_data_latent[:, :3], expected_values)
    torch.testing.assert_close(x_data_latent[:, -1], expected_flag)
    torch.testing.assert_close(data_coords, expected_coords)
    torch.testing.assert_close(timedeltas, torch.tensor([-1.0] * 2 + [1.0] * 3 + [-1.0] + [1.0] * 2))


def test_transport_skips_a_sparse_dataset_without_points_in_the_batch() -> None:
    """As in the deterministic model, a dataset with no history and no target points is not encoded."""
    model = _transport_model_stub()
    model.node_attributes = _EmptyNodeAttributes()
    x = _obs_source([0], n_variables=2, value=1.0, coordinate_offset=-1.0)
    y_noised = _obs_source([0], n_variables=1, value=5.0, coordinate_offset=1.0)

    assembled = model._assemble_transport_input(x, y_noised, batch_size=1, dataset_name="obs")
    source = model._encoder_source_from_rows(
        "obs",
        assembled,
        batch_size=1,
        hidden_coordinates=torch.zeros(5, 2),
        hidden_coordinates_batched=torch.zeros(5, 2),
        hidden_batch_sizes=(5,),
        shard_sizes_hidden=None,
    )

    assert source is None


def test_transport_assemble_input_adds_timedelta_features_to_sparse_rows() -> None:
    """Configured timedelta node attributes are encoded for the history and the noisy-target rows."""
    model = _transport_model_stub()
    model.node_attributes = _EmptyNodeAttributes()
    model.dynamic_node_attributes = {"obs": {"timedeltas": Timedeltas(scale_seconds=1.0)}}
    x = _obs_source([2], n_variables=1, value=1.0, coordinate_offset=-3.0)
    y_noised = _obs_source([1], n_variables=1, value=5.0, coordinate_offset=4.0)

    _, x_data_latent, _, _, _, _ = model._assemble_transport_input(x, y_noised, batch_size=1, dataset_name="obs")

    # [history | noisy target | sin/cos coordinates | scaled timedelta | flag]
    assert x_data_latent.shape == (3, 1 + 1 + 4 + 1 + 1)
    torch.testing.assert_close(x_data_latent[:, -2], torch.tensor([-3.0, -3.0, 4.0]))


def test_transport_input_width_counts_the_flag_of_sparse_data() -> None:
    model = _transport_model_stub()
    model.is_dataset_static = {"grid": True, "obs": False}

    widths = {}
    for name in ("grid", "obs"):
        model.n_step_input = {name: 2}
        model.n_step_output = {name: 1}
        model.num_input_channels = {name: 3}
        model.num_output_channels = {name: 2}
        model.node_attributes = SimpleNamespace(num_trainable_parameters={})
        model.dynamic_node_attribute_dims = {}
        widths[name] = model._calculate_input_dim(name)

    coords_dim = 4
    assert widths["grid"] == 2 * 3 + coords_dim + 1 * 2
    assert widths["obs"] == 1 * 3 + coords_dim + 1 * 2 + 1


def test_second_blocks_recovers_the_rows_joined_second() -> None:
    first = torch.arange(4.0)[:, None]
    second = torch.arange(10.0, 15.0)[:, None]

    joined = transport_model_module._join_blocks(first, [3, 1], second, [2, 3])

    torch.testing.assert_close(joined[:, 0], torch.tensor([0.0, 1.0, 2.0, 10.0, 11.0, 3.0, 12.0, 13.0, 14.0]))
    torch.testing.assert_close(transport_model_module._second_blocks(joined, [3, 1], [2, 3]), second)


def test_tendency_transport_assemble_input_uses_dense_source_views_with_residual_conditioning() -> None:
    model = AnemoiTransportTendModelEncProcDec.__new__(AnemoiTransportTendModelEncProcDec)
    model.node_attributes = _EmptyNodeAttributes()
    model.dynamic_node_attributes = {}
    model.condition_on_residual = True
    model.n_step_output = {"data": 1}
    model._internal_input_idx = {"data": torch.tensor([0, 2])}

    class _Residual:
        def __init__(self) -> None:
            self.grid_shard_sizes = object()

        def __call__(self, x, grid_shard_sizes, model_comm_group, n_step_output):
            del model_comm_group
            self.grid_shard_sizes = grid_shard_sizes
            assert n_step_output == 1
            return x

    residual = _Residual()
    model.residual = {"data": residual}

    layout = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
    coordinates = torch.zeros(3, 2)
    x_data = torch.arange(1 * 2 * 1 * 3 * 4, dtype=torch.float32).reshape(1, 2, 1, 3, 4)
    y_noised_data = torch.full((1, 1, 1, 3, 2), 100.0)
    x = GriddedSource(
        name="data",
        data=x_data,
        coordinates=coordinates,
        variables=["a", "b", "c", "d"],
        statistics={},
        layout=layout,
    )
    y_noised = GriddedSource(
        name="data",
        data=y_noised_data,
        coordinates=coordinates,
        variables=["a", "c"],
        statistics={},
        layout=layout,
    )

    data_coords, x_data_latent, x_skip, shard_sizes, batch_sizes, timedeltas = model._assemble_transport_input(
        x,
        y_noised,
        batch_size=1,
        dataset_name="data",
    )

    expected_x_features = x_data.permute(0, 2, 3, 1, 4).reshape(3, 8)
    expected_target_features = y_noised_data.permute(0, 2, 3, 1, 4).reshape(3, 2)
    expected_residual_features = x_data[..., [0, 2]].permute(0, 2, 3, 1, 4).reshape(3, 4)

    assert shard_sizes is None
    assert residual.grid_shard_sizes is None
    torch.testing.assert_close(data_coords, coordinates)
    assert x_skip.shape == (1, 3, 4)
    assert x_data_latent.shape == (3, 8 + 2 + 4 + 4)
    torch.testing.assert_close(x_data_latent[:, :8], expected_x_features)
    torch.testing.assert_close(x_data_latent[:, 8:10], expected_target_features)
    torch.testing.assert_close(x_data_latent[:, -4:], expected_residual_features)


def test_tendency_transport_assemble_input_rejects_sparse_obs() -> None:
    model = AnemoiTransportTendModelEncProcDec.__new__(AnemoiTransportTendModelEncProcDec)
    model.node_attributes = _EmptyNodeAttributes()
    model.condition_on_residual = False

    layout = TensorLayout(grid=0, variables=1)
    sparse_view = TabularSource(
        name="obs",
        data=[torch.ones(2, 1)],
        coordinates=[torch.zeros(2, 2)],
        variables=["a"],
        statistics={},
        layout=layout,
        boundaries=[(slice(0, 2),)],
        timedeltas=[torch.zeros(2)],
    )

    with pytest.raises(NotImplementedError, match="Tendency transport.*sparse"):
        model._assemble_transport_input(sparse_view, sparse_view, batch_size=1, dataset_name="obs")


def test_tendency_transport_forward_network_uses_dense_source_view_override() -> None:
    """The forward pass encodes through the base model's steps, with the transport inputs and conditioning.

    The encoder rows hold the history and the noisy target, and each mapper receives its own conditioning.
    """
    model = AnemoiTransportTendModelEncProcDec.__new__(AnemoiTransportTendModelEncProcDec)
    torch.nn.Module.__init__(model)
    model._graph_name_hidden = "hidden"
    model.node_attributes = _EmptyNodeAttributes()
    model.dynamic_node_attributes = {}
    model.condition_on_residual = True
    model.n_step_output = {"data": 1}
    model._internal_input_idx = {"data": torch.tensor([0, 1])}
    model.latent_skip = False
    model.input_datasets = ["data"]
    model.target_datasets = ["data"]
    model.dataset2encoder = {"data": "data"}
    model.encoder2datasets = {"data": ["data"]}
    model.encoder_fusing_strategy = {"data": "none"}
    model.encoder_src_projection = torch.nn.ModuleDict()
    model.dataset2decoder = {"data": "data"}
    model.decoders_target_input = {
        "data": SimpleNamespace(features=[SimpleNamespace(name="encoded_data")]),
    }
    model._hidden_coordinates = lambda: torch.zeros(5, 2)
    encoder_cond, processor_cond, decoder_cond = object(), object(), object()
    model._build_conditioning_kwargs = lambda *_args, **_kwargs: (
        {"data": {"cond": encoder_cond}},
        {"cond": processor_cond},
        {"data": {"cond": decoder_cond}},
    )
    model._assemble_target = lambda _input, encoded, target, _target_template, **_kwargs: (
        target.flatten().coordinates,
        encoded,
        None,
        None,
        None,
    )

    class _Residual:
        def __init__(self) -> None:
            self.called = False

        def __call__(self, x, grid_shard_sizes, model_comm_group, n_step_output):
            del grid_shard_sizes, model_comm_group
            self.called = True
            assert n_step_output == 1
            return x

    residual = _Residual()
    model.residual = {"data": residual}

    received = {}

    class _Encoder:
        def __init__(self) -> None:
            self.source_width = None

        def __call__(self, x, **kwargs):
            self.source_width = x[0].shape[-1]
            received["encoder_rows"] = x[0]
            received["encoder"] = kwargs.get("cond")
            return x[0], torch.ones_like(x[1])

    class _Processor:
        def __call__(self, x, **kwargs):
            received["processor"] = kwargs.get("cond")
            return x

    class _Decoder:
        def __call__(self, x, **kwargs):
            received["decoder"] = kwargs.get("cond")
            target_features = x[1]
            return torch.zeros(target_features.shape[0], 2)

    encoder = _Encoder()
    model.encoder = {"data": encoder}
    model.processor = _Processor()
    model.latent_aggregator = SumAggregator(input_channels=4, source_channels={"data": 4})
    model.decoder = {"data": _Decoder()}
    model.encoder_graph_provider = {"data": _GraphProvider()}
    model.processor_graph_provider = _GraphProvider()
    model.decoder_graph_provider = {"data": _GraphProvider()}

    layout = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
    batch = build_batch(
        data={"data": torch.randn(1, 2, 1, 3, 2)},
        coordinates={"data": torch.zeros(3, 2)},
        metadata={"static_coords": frozenset({"data"})},
        layouts={"data": layout},
        variables={"data": ["a", "b"]},
        statistics={"data": {}},
    )
    conditioned_target = batch.with_data({"data": torch.randn(1, 1, 1, 3, 2)})

    out = model._forward_transport_network(
        batch,
        conditioned_target,
        {"data": torch.zeros(1, 1, 1, 1, 1)},
    )

    assert residual.called
    assert encoder.source_width == 14
    # Rows: [history (2 steps x 2 variables) | noisy target (2) | coordinates (4) | residual (4)].
    history = batch["data"].data[0, :, 0].permute(1, 0, 2).reshape(3, 4)
    torch.testing.assert_close(received["encoder_rows"][:, :4], history)
    torch.testing.assert_close(received["encoder_rows"][:, 4:6], conditioned_target["data"].data[0, 0, 0])
    assert received["encoder"] is encoder_cond
    assert received["processor"] is processor_cond
    assert received["decoder"] is decoder_cond
    assert isinstance(out, Batch)
    assert out["data"].data.shape == (1, 1, 1, 3, 2)


def test_transport_target_dim_combines_corrupted_target_and_decoding_forcings() -> None:
    """The obs transport decoder consumes outputs + decoding forcings + coordinates."""
    model = _transport_model_stub()
    model.dataset2decoder = {"obs": "obs_decoder"}
    model.decoders_target_input = {
        "obs_decoder": SimpleNamespace(
            dim=16,
            features=[SimpleNamespace(name="coordinates"), SimpleNamespace(name="target_forcings")],
        ),
    }
    model.n_step_output = {"obs": 1}
    model.num_output_channels = {"obs": 3}

    coords_dim = 4
    assert model._calculate_target_dim("obs") == 3 + 12 + coords_dim


@pytest.mark.parametrize("shared_encoder", [False, True])
def test_transport_decoder_combines_corrupted_target_with_explicit_target_features(shared_encoder: bool) -> None:
    model = _transport_model_stub()
    model._graph_name_hidden = "hidden"
    model.node_attributes = _EmptyNodeAttributes()
    model.latent_skip = False
    model.input_datasets = ["obs"]
    model.target_datasets = ["obs"]
    model.dataset2encoder = {"obs": "obs"}
    if shared_encoder:
        # An encoder shared with another dataset projects each dataset's rows to a common width.
        model.encoder2datasets = {"obs": ["obs", "grid"]}
        model.encoder_fusing_strategy = {"obs": "sequential"}
        model.encoder_src_projection = torch.nn.ModuleDict(
            {"obs": torch.nn.ModuleDict({"obs": torch.nn.Linear(22, 4), "grid": torch.nn.Linear(8, 4)})},
        )
    else:
        model.encoder2datasets = {"obs": ["obs"]}
        model.encoder_fusing_strategy = {"obs": "none"}
        model.encoder_src_projection = torch.nn.ModuleDict()
    model.dataset2decoder = {"obs": "obs"}
    model.decoders_target_input = {
        "obs": SimpleNamespace(
            features=[SimpleNamespace(name="coordinates"), SimpleNamespace(name="target_forcings")],
        ),
    }
    model._get_consistent_dim = lambda _batch, dim: 1
    model._resolve_in_out_sharded = lambda batch: {dataset_name: False for dataset_name in batch.keys()}
    model._assert_valid_sharding = lambda *_args, **_kwargs: None
    model._build_conditioning_kwargs = lambda *_args, **_kwargs: ({"obs": {}}, {}, {"obs": {}})
    model._hidden_coordinates = lambda: torch.zeros(5, 2)
    # The encoder rows are the 3 history nodes followed by the 3 noisy-target nodes.
    model._assemble_transport_input = lambda *_args, **_kwargs: (
        torch.zeros(6, 2),
        torch.zeros(6, 22),
        None,
        None,
        (6,),
        None,
    )

    expected_target_features = torch.zeros(3, 4)
    assembled_views = {}

    def _assemble_target_stub(_input_view, _encoded_data, target_view, _target_template, **_kwargs):
        assembled_views["obs"] = target_view
        return (
            torch.zeros(3, 2),
            expected_target_features,
            None,
            None,
            None,
        )

    model._assemble_target = _assemble_target_stub
    model._assemble_output = lambda _x_out, _x_skip, target, _dtype, _dataset_name: target.clone(
        data=[torch.zeros(3, 1)],
    )

    class _Encoder:
        def __init__(self) -> None:
            self.source_width = None

        def __call__(self, x, **_kwargs):
            self.source_width = x[0].shape[-1]
            return torch.zeros(6, x[0].shape[-1]), torch.ones_like(x[1])

    class _Processor:
        def __call__(self, x, **_kwargs):
            return x

    class _Decoder:
        def __init__(self) -> None:
            self.destination_features = None

        def __call__(self, x, **_kwargs):
            self.destination_features = x[1]
            return torch.zeros(3, 1)

    decoder = _Decoder()
    encoder = _Encoder()
    model.encoder = {"obs": encoder}
    model.processor = _Processor()
    model.latent_aggregator = SumAggregator(input_channels=4, source_channels={"obs": 4})
    model.decoder = {"obs": decoder}
    model.encoder_graph_provider = {"obs": _GraphProvider()}
    model.processor_graph_provider = _GraphProvider()
    model.decoder_graph_provider = {"obs": _GraphProvider()}

    layout = TensorLayout(grid=0, variables=1)
    batch = build_batch(
        data={"obs": [torch.ones(3, 1)]},
        coordinates={"obs": [torch.zeros(3, 2)]},
        timedeltas={"obs": [torch.zeros(3)]},
        boundaries={"obs": [(slice(0, 3),)]},
        layouts={"obs": layout},
        variables={"obs": ["a"]},
        statistics={"obs": {}},
    )

    target_forcing = batch.replace(
        "obs",
        batch["obs"].clone(
            data=[torch.full((3, 2), 5.0)],
            variables=["forcing_a", "forcing_b"],
        ),
    )
    model._forward_transport_network(
        batch,
        batch,
        {"obs": torch.zeros(1, 1, 1, 1, 1)},
        target_forcing=target_forcing,
    )

    assert encoder.source_width == (4 if shared_encoder else 22)
    assert decoder.destination_features.shape == (3, 5)
    torch.testing.assert_close(decoder.destination_features[:, 0], torch.ones(3))
    # Explicit target features receive the forcing view; the corrupted target is prepended separately.
    assembled_data = assembled_views["obs"].data[0]
    assert assembled_data.shape == (3, 2)
    torch.testing.assert_close(assembled_data, torch.full((3, 2), 5.0))


def test_transport_conditioning_rejects_expanded_condition_shape() -> None:
    model = _transport_model_stub()
    expanded_condition = {"data": torch.zeros(2, 4, 3, 7, 1)}

    with pytest.raises(AssertionError, match="Expected condition to have shape"):
        model._assert_condition_shapes(expanded_condition)


class AddOneProcessor(torch.nn.Module):
    """Stands in for a processor: adds one to every value of the source it is given."""

    def forward(self, view, in_place: bool = True, **_kwargs):
        assert in_place is False
        return view.clone(data=view.data + 1)


def test_predict_step_samples_onto_the_target_and_postprocesses() -> None:
    """predict_step samples onto the target templates, conditioned on the normalised forcings, and post-processes."""
    model = _transport_model_stub()
    _configure_sampling_model(model, {"ds_a": (2, 3, 4)})
    model.n_step_output = {"ds_a": 1}

    x = _sampling_batch(model, {"ds_a": torch.zeros(1, 2, 1, 4, 2)})
    target_template = _target_template(model, {"ds_a": torch.zeros(1, 2, 1, 4, 2)})
    target_forcing = _sampling_batch(model, {"ds_a": torch.zeros(1, 1, 1, 4, 2)})
    captured = {}

    def _sample(x_sampled, **kwargs):
        captured["x"] = x_sampled
        captured.update(kwargs)
        return Batch({"ds_a": kwargs["target_template"]["ds_a"].unflatten(torch.full((4, 3), 5.0))})

    model.sample = _sample

    out = model.predict_step(
        x=x,
        target_template=target_template,
        target_forcing=target_forcing,
        pre_processors=torch.nn.ModuleDict({"ds_a": AddOneProcessor()}),
        post_processors=torch.nn.ModuleDict({"ds_a": AddOneProcessor()}),
        n_step_input={"ds_a": 2},
        model_comm_group=None,
        gather_out=True,
        sampler_params={"sampler": "heun"},
    )

    # Inputs and target forcings are normalised before sampling.
    torch.testing.assert_close(captured["x"]["ds_a"].data, torch.ones(1, 2, 1, 4, 2))
    torch.testing.assert_close(captured["target_forcing"]["ds_a"].data, torch.ones(1, 1, 1, 4, 2))
    # The templates give the nodes to sample onto.
    assert captured["target_template"] is target_template
    assert captured["sampler_params"] == {"sampler": "heun"}
    # The sample is post-processed and returned as a Batch on the target nodes.
    assert isinstance(out, Batch)
    torch.testing.assert_close(out["ds_a"].data, torch.full((1, 1, 1, 4, 3), 6.0))
    assert out["ds_a"].variables == list(model.data_indices["ds_a"].model.output.ordered_names)


def test_predict_step_runs_without_target_forcings() -> None:
    """Without forcings, predict_step samples with an empty decoder conditioning."""
    model = _transport_model_stub()
    _configure_sampling_model(model, {"ds_a": (2, 3, 4)})
    model.n_step_output = {"ds_a": 1}
    captured = {}

    def _sample(_x, **kwargs):
        captured.update(kwargs)
        return Batch({"ds_a": kwargs["target_template"]["ds_a"].unflatten(torch.zeros(4, 3))})

    model.sample = _sample

    model.predict_step(
        x=_sampling_batch(model, {"ds_a": torch.zeros(1, 2, 1, 4, 2)}),
        target_template=_target_template(model, {"ds_a": torch.zeros(1, 2, 1, 4, 2)}),
        pre_processors=torch.nn.ModuleDict({"ds_a": AddOneProcessor()}),
        post_processors=torch.nn.ModuleDict({"ds_a": AddOneProcessor()}),
        n_step_input={"ds_a": 2},
    )

    assert list(captured["target_forcing"].keys()) == []


def test_transport_models_reject_configured_boundings() -> None:
    model = _transport_model_stub()
    model.boundings = torch.nn.ModuleDict(
        {"grid": torch.nn.Sequential(), "obs": torch.nn.Sequential(torch.nn.ReLU())},
    )

    with pytest.raises(ValueError, match=r"not supported for transport models.*'obs'"):
        model._assert_no_boundings()


def test_transport_models_accept_empty_boundings() -> None:
    model = _transport_model_stub()
    model.boundings = torch.nn.ModuleDict({"grid": torch.nn.Sequential()})

    model._assert_no_boundings()


def test_predict_step_accepts_the_interface_arguments() -> None:
    """Every argument the model interface passes to predict_step has a matching parameter."""
    parameters = inspect.signature(AnemoiTransportModelEncProcDec.predict_step).parameters
    interface_arguments = {
        "x",
        "target_template",
        "target_forcing",
        "pre_processors",
        "post_processors",
        "n_step_input",
        "model_comm_group",
        "gather_out",
        "statistics_tendencies",
        "spatial_pre_processors",
    }
    assert interface_arguments <= set(parameters)


def test_sample_passes_zero_terminated_schedule_to_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummySchedule(schedules.SigmaSchedule):
        def __init__(self, sigma_max: float, sigma_min: float, num_steps: int):
            super().__init__(sigma_max=sigma_max, sigma_min=sigma_min, num_steps=num_steps)

        def _build_schedule(self, device=None, dtype_compute: torch.dtype = torch.float64):
            return torch.linspace(1.0, 0.1, self.num_steps, device=device, dtype=dtype_compute)

    class DummySampler:
        def __init__(self, dtype: torch.dtype = torch.float64, **kwargs):
            del kwargs
            self.dtype = dtype

        def sample(
            self,
            x: Batch,
            y: Batch,
            sigmas: torch.Tensor,
            denoising_fn,
            model_comm_group=None,
            **kwargs,
        ):
            del denoising_fn, model_comm_group, kwargs
            assert isinstance(sigmas, torch.Tensor)
            assert sigmas.shape == (5,)
            assert sigmas[-1] == 0.0
            for dataset_name, y_data in ((n, s.data) for n, s in y.items()):
                assert y_data.dtype == sigmas.dtype
                assert y_data.shape[:4] == (
                    x[dataset_name].data.shape[0],
                    2,
                    x[dataset_name].data.shape[2],
                    x[dataset_name].data.shape[-2],
                )
            return y

    model = _transport_model_stub()
    model.inference_defaults = {
        "sampling_schedule": {
            "schedule_type": "dummy",
            "sigma_max": 1.0,
            "sigma_min": 0.1,
            "num_steps": 4,
        },
        "sampler": {"sampler": "dummy"},
    }
    model.n_step_output = {"ds_a": 2, "ds_b": 2}
    model.num_output_channels = {"ds_a": 3, "ds_b": 4}
    model.transport_model_objective = EDMDiffusionModelObjective()
    model.edm = EdmSettings(sigma_data=1.0)
    model._forward_transport_network = lambda *_args, **_kwargs: None
    _configure_sampling_model(model, {"ds_a": (6, 3, 5), "ds_b": (5, 4, 7)})

    monkeypatch.setitem(schedules.SIGMA_SCHEDULES, "dummy", DummySchedule)
    monkeypatch.setitem(transport_samplers.DIFFUSION_SAMPLERS, "dummy", DummySampler)

    x = {
        "ds_a": torch.randn(1, 3, 1, 5, 6, dtype=torch.float32),
        "ds_b": torch.randn(1, 3, 1, 7, 5, dtype=torch.float32),
    }

    out = model.sample(_sampling_batch(model, x), target_template=_target_template(model, x))
    assert set(out.keys()) == {"ds_a", "ds_b"}


def test_edm_sparse_sampling_uses_target_template_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    class SpySampler:
        def __init__(self, dtype: torch.dtype = torch.float64, **kwargs):
            del kwargs
            self.dtype = dtype

        def sample(
            self,
            x: Batch,
            y: Batch,
            sigmas: torch.Tensor,
            denoising_fn,
            model_comm_group=None,
            **kwargs,
        ):
            del x, sigmas, denoising_fn, model_comm_group, kwargs
            assert [tuple(sample.shape) for sample in y["obs"].data] == [(3, 1), (1, 1)]
            assert [tuple(coords.shape) for coords in y["obs"].coordinates] == [(3, 2), (1, 2)]
            return y

    model = _transport_model_stub()
    model.inference_defaults = {
        "sampling_schedule": {"schedule_type": "linear", "sigma_max": 1.0, "sigma_min": 0.1, "num_steps": 2},
        "sampler": {"sampler": "spy"},
    }
    model.n_step_output = {"obs": 1}
    model.num_output_channels = {"obs": 1}
    model.transport_model_objective = EDMDiffusionModelObjective()
    model.edm = EdmSettings(sigma_data=1.0)
    _configure_sampling_model(model, {"obs": (2, 1, 1)})

    monkeypatch.setitem(transport_samplers.DIFFUSION_SAMPLERS, "spy", SpySampler)

    x = _sparse_batch(name="obs", data_shapes=[(2, 2), (4, 2)], variables=["in_0", "in_1"])
    target_template = _sparse_target_template(name="obs", node_counts=[3, 1], variables=["out_0"])

    out = model.sample(x, target_template=target_template)

    assert [tuple(sample.shape) for sample in out["obs"].data] == [(3, 1), (1, 1)]


def test_stochastic_interpolant_sparse_sampling_uses_target_template_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SpyVectorFieldSampler:
        def __init__(self, dtype: torch.dtype = torch.float64, **kwargs):
            del kwargs
            self.dtype = dtype

        def sample(
            self,
            x: Batch,
            y: Batch,
            times: torch.Tensor,
            vector_field_fn,
            model_comm_group=None,
            **kwargs,
        ):
            del x, times, vector_field_fn, model_comm_group, kwargs
            assert [tuple(sample.shape) for sample in y["obs"].data] == [(5, 1), (2, 1)]
            assert [tuple(coords.shape) for coords in y["obs"].coordinates] == [(5, 2), (2, 2)]
            return y

    model = _transport_model_stub()
    model.transport_model_objective = StochasticInterpolantModelObjective()
    model.inference_defaults = {
        "sampling_schedule": {"schedule_type": "unit_time", "num_steps": 2},
        "sampler": {"sampler": "spy_vector"},
    }
    model.n_step_output = {"obs": 1}
    model.num_output_channels = {"obs": 1}
    _configure_sampling_model(model, {"obs": (2, 1, 1)})

    monkeypatch.setitem(transport_samplers.VECTOR_FIELD_SAMPLERS, "spy_vector", SpyVectorFieldSampler)

    x = _sparse_batch(name="obs", data_shapes=[(2, 2), (4, 2)], variables=["in_0", "in_1"])
    target_template = _sparse_target_template(name="obs", node_counts=[5, 2], variables=["out_0"])

    out = model.sample(x, target_template=target_template)

    assert [tuple(sample.shape) for sample in out["obs"].data] == [(5, 1), (2, 1)]


def test_transport_sampling_requires_target_template() -> None:
    model = _transport_model_stub()
    model.inference_defaults = {
        "sampling_schedule": {"schedule_type": "linear", "sigma_max": 1.0, "sigma_min": 0.1, "num_steps": 2},
        "sampler": {"sampler": "heun"},
    }
    model.n_step_output = {"data": 1}
    model.num_output_channels = {"data": 1}
    model.transport_model_objective = EDMDiffusionModelObjective()
    model.edm = EdmSettings(sigma_data=1.0)
    _configure_sampling_model(model, {"data": (1, 1, 2)})

    x = _sampling_batch(model, {"data": torch.randn(1, 1, 1, 2, 1)})

    with pytest.raises(TypeError, match="target_template"):
        model.sample(x)


def test_tendency_sparse_sampling_rejects_sparse_obs(monkeypatch: pytest.MonkeyPatch) -> None:
    class CallingSampler:
        def __init__(self, dtype: torch.dtype = torch.float64, **kwargs):
            del kwargs
            self.dtype = dtype

        def sample(
            self,
            x: Batch,
            y: Batch,
            sigmas: torch.Tensor,
            denoising_fn,
            model_comm_group=None,
            **kwargs,
        ):
            del sigmas, kwargs
            return denoising_fn(
                x,
                y,
                {"obs": torch.ones(2, 1, 1, 1, 1)},
                model_comm_group,
            )

    model = AnemoiTransportTendModelEncProcDec.__new__(AnemoiTransportTendModelEncProcDec)
    model.transport_source = TransportSourceBuilder()
    model.transport_model_objective = EDMDiffusionModelObjective()
    model.inference_defaults = {
        "sampling_schedule": {"schedule_type": "linear", "sigma_max": 1.0, "sigma_min": 0.1, "num_steps": 2},
        "sampler": {"sampler": "calling"},
    }
    model.edm = EdmSettings(sigma_data=1.0)
    model.n_step_output = {"obs": 1}
    model.num_output_channels = {"obs": 1}
    model._graph_name_hidden = "hidden"
    model.node_attributes = _EmptyNodeAttributes()
    model._hidden_coordinates = lambda: torch.zeros(5, 2)
    model._get_consistent_dim = lambda _batch, dim: 2 if dim == 0 else 1
    model._resolve_in_out_sharded = lambda batch: {dataset_name: False for dataset_name in batch.keys()}
    model._assert_valid_sharding = lambda *_args, **_kwargs: None
    model._build_conditioning_kwargs = lambda *_args, **_kwargs: ({"obs": {}}, {}, {"obs": {}})
    model.input_datasets = ["obs"]
    model.encoder2datasets = {"obs": ["obs"]}
    model.encoder_fusing_strategy = {"obs": "none"}
    model.encoder_src_projection = {}
    _configure_sampling_model(model, {"obs": (2, 1, 1)})

    monkeypatch.setitem(transport_samplers.DIFFUSION_SAMPLERS, "calling", CallingSampler)

    x = _sparse_batch(name="obs", data_shapes=[(2, 2), (4, 2)], variables=["in_0", "in_1"])
    target_template = _sparse_target_template(name="obs", node_counts=[3, 1], variables=["out_0"])

    with pytest.raises(NotImplementedError, match="Tendency transport.*sparse"):
        model.sample(x, target_template=target_template)


def test_sample_dispatches_stochastic_interpolant_to_default_heun_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyVectorFieldSampler:
        def __init__(self, dtype: torch.dtype = torch.float64, **kwargs):
            del kwargs
            self.dtype = dtype

        def sample(
            self,
            x: Batch,
            y: Batch,
            times: torch.Tensor,
            vector_field_fn,
            model_comm_group=None,
            **kwargs,
        ):
            del kwargs
            assert times.shape == (4,)
            assert times[0] == 0.0
            assert times[-1] == 1.0
            torch.testing.assert_close(y["ds_a"].data, torch.zeros_like(y["ds_a"].data))
            return vector_field_fn(
                x,
                y,
                {"ds_a": torch.zeros(1, 1, 1, 1, 1)},
                model_comm_group,
            )

    model = _transport_model_stub()
    model.transport_model_objective = StochasticInterpolantModelObjective()
    model.inference_defaults = {
        "sampling_schedule": {"schedule_type": "unit_time", "num_steps": 3},
        "sampler": {"sampler": "heun"},
    }
    model.n_step_output = {"ds_a": 2}
    model.num_output_channels = {"ds_a": 3}
    model._forward_transport_network = lambda _x, y, *_args, **_kwargs: y
    _configure_sampling_model(model, {"ds_a": (6, 3, 5)})
    model.build_sampling_source = lambda x, **_kwargs: {
        "ds_a": torch.zeros(
            x["ds_a"].data.shape[0],
            model.n_step_output["ds_a"],
            x["ds_a"].data.shape[2],
            x["ds_a"].data.shape[-2],
            model.num_output_channels["ds_a"],
            device=x["ds_a"].data.device,
            dtype=x["ds_a"].data.dtype,
        )
    }

    monkeypatch.setitem(transport_samplers.VECTOR_FIELD_SAMPLERS, "heun", DummyVectorFieldSampler)

    x = {"ds_a": torch.randn(1, 3, 1, 5, 6, dtype=torch.float32)}

    out = model.sample(_sampling_batch(model, x), target_template=_target_template(model, x))

    assert set(out.keys()) == {"ds_a"}
    assert out["ds_a"].data.shape == (1, 2, 1, 5, 3)


def test_sample_can_use_deterministic_vector_field_sampler_for_stochastic_interpolant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyVectorFieldSampler:
        def __init__(self, dtype: torch.dtype = torch.float64, **kwargs):
            del kwargs
            self.dtype = dtype

        def sample(
            self,
            x: Batch,
            y: Batch,
            times: torch.Tensor,
            vector_field_fn,
            model_comm_group=None,
            **kwargs,
        ):
            del kwargs
            assert times.shape == (4,)
            assert times[0] == 0.0
            assert times[-1] == 1.0
            torch.testing.assert_close(y["ds_a"].data, torch.zeros_like(y["ds_a"].data))
            return vector_field_fn(
                x,
                y,
                {"ds_a": torch.zeros(1, 1, 1, 1, 1)},
                model_comm_group,
            )

    model = _transport_model_stub()
    model.transport_model_objective = StochasticInterpolantModelObjective()
    model.inference_defaults = {
        "sampling_schedule": {"schedule_type": "unit_time", "num_steps": 3},
        "sampler": {"sampler": "heun"},
    }
    model.n_step_output = {"ds_a": 2}
    model.num_output_channels = {"ds_a": 3}
    model._forward_transport_network = lambda _x, y, *_args, **_kwargs: y
    _configure_sampling_model(model, {"ds_a": (6, 3, 5)})
    model.build_sampling_source = lambda x, **_kwargs: {
        "ds_a": torch.zeros(
            x["ds_a"].data.shape[0],
            model.n_step_output["ds_a"],
            x["ds_a"].data.shape[2],
            x["ds_a"].data.shape[-2],
            model.num_output_channels["ds_a"],
            device=x["ds_a"].data.device,
            dtype=x["ds_a"].data.dtype,
        )
    }

    monkeypatch.setitem(transport_samplers.VECTOR_FIELD_SAMPLERS, "dummy_vector", DummyVectorFieldSampler)

    x = {"ds_a": torch.randn(1, 3, 1, 5, 6, dtype=torch.float32)}

    out = model.sample(
        _sampling_batch(model, x),
        target_template=_target_template(model, x),
        sampler_params={"sampler": "dummy_vector"},
    )

    assert set(out.keys()) == {"ds_a"}
    assert out["ds_a"].data.shape == (1, 2, 1, 5, 3)


def _single_dataset_template(data: torch.Tensor) -> Batch:
    """A one-dataset gridded batch describing the field a source builder should produce."""
    return build_batch(
        data={"data": data},
        coordinates={"data": torch.zeros(data.shape[3], 2)},
        layouts={"data": TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)},
        variables={"data": [f"v{idx}" for idx in range(data.shape[4])]},
    )


def test_transport_source_builder_does_not_build_unselected_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = TransportSourceBuilder(TransportSourceSettings(kind="gaussian", scale=2.0))
    target = _single_dataset_template(torch.zeros(1, 1, 1, 2, 1))

    def reference_source_factory() -> dict[str, torch.Tensor]:
        raise AssertionError("reference source should not be built")

    def fake_randn(shape, device=None, dtype=None):
        return torch.full(shape, 3.0, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randn", fake_randn)

    source = builder.build(
        TransportSourceRequest(
            templates=target,
            default_kind="reference_state",
            custom_source_factories={"reference_state": reference_source_factory},
        )
    )

    torch.testing.assert_close(source["data"].data, torch.full_like(target["data"].data, 6.0))


def test_transport_source_builder_postprocesses_reference_source(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = TransportSourceBuilder(TransportSourceSettings(kind="reference_state", scale=0.5, noise_scale=0.25))
    target = _single_dataset_template(torch.zeros(1, 1, 1, 2, 1, dtype=torch.float64))
    reference = {"data": torch.full_like(target["data"].data, 4.0)}

    monkeypatch.setattr(
        torch, "randn", lambda shape, device=None, dtype=None: torch.full(shape, 2.0, device=device, dtype=dtype)
    )

    source = builder.build(
        TransportSourceRequest(
            templates=target,
            default_kind="gaussian",
            custom_source_factories={"reference_state": lambda: reference},
        )
    )

    assert source["data"].dtype == target["data"].dtype
    torch.testing.assert_close(source["data"].data, torch.full_like(target["data"].data, 2.5))
    torch.testing.assert_close(reference["data"], torch.full_like(target["data"].data, 4.0))


def test_tendency_sampling_source_can_use_reference_state() -> None:
    model = AnemoiTransportTendModelEncProcDec.__new__(AnemoiTransportTendModelEncProcDec)
    model.transport_source = TransportSourceBuilder(TransportSourceSettings(kind="reference_state"))
    model.target_datasets = ["ds_a"]
    model.n_step_output = {"ds_a": 2}
    model.num_output_channels = {"ds_a": 2}
    model.statistics = {"ds_a": {}}
    model.data_indices = {
        "ds_a": SimpleNamespace(
            name_to_index={"a": 0, "c": 1, "b": 2, "d": 3},
            model=SimpleNamespace(
                output=SimpleNamespace(ordered_names=("a", "b")),
                input=SimpleNamespace(positions_for_names=lambda names: [0, 2]),
            ),
        ),
    }
    x_data = torch.arange(1 * 3 * 1 * 5 * 4, dtype=torch.float32).reshape(1, 3, 1, 5, 4)
    x = build_batch(
        data={"ds_a": x_data},
        coordinates={"ds_a": torch.zeros(5, 2)},
        metadata={"static_coords": frozenset({"ds_a"})},
        layouts={"ds_a": TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)},
        variables={"ds_a": ["a", "c", "b", "d"]},
        statistics={"ds_a": {}},
    )

    source = model.build_sampling_source(
        x,
        target_template=_templates(x.select(time=[1, 2], variables={"ds_a": [0, 2]})),
    )

    expected = x_data[:, -1:, :, :, :].index_select(-1, torch.tensor([0, 2])).expand(-1, 2, -1, -1, -1)
    torch.testing.assert_close(source["ds_a"].data, expected)


def test_stochastic_interpolant_objective_returns_raw_drift_prediction() -> None:
    """The stochastic-interpolant model objective leaves drift predictions in model-output space."""
    interpolant = build_batch(
        data={"data": torch.full((1, 1, 1, 2, 1), 2.0)},
        coordinates={"data": torch.zeros(2, 2)},
        metadata={"static_coords": frozenset({"data"})},
        layouts={"data": TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)},
        variables={"data": ["x"]},
        statistics={"data": {}},
    )
    time_level = {"data": torch.full_like(interpolant["data"].data, 0.25)}
    drift = interpolant.with_data({"data": torch.full_like(interpolant["data"].data, 0.5)})
    marker = object()

    def _forward_transport_network(
        _x: Batch,
        conditioned_target: Batch,
        condition: dict[str, torch.Tensor],
        **_kwargs,
    ) -> Batch:
        assert conditioned_target is interpolant
        assert condition is time_level
        assert _kwargs["marker"] is marker
        return drift

    model = SimpleNamespace(_forward_transport_network=_forward_transport_network)

    out = StochasticInterpolantModelObjective().forward(
        model,
        interpolant.with_data({"data": torch.zeros_like(interpolant["data"].data)}),
        interpolant,
        time_level,
        marker=marker,
    )

    assert out is drift


@pytest.mark.parametrize(
    ("sampler_name", "sampler_config"),
    [
        ("heun", {"S_churn": 0.0, "S_min": 0.0, "S_max": float("inf"), "S_noise": 1.0}),
        ("dpmpp_2m", {}),
    ],
)
def test_sample_end_to_end_multi_dataset_real_sampler(
    sampler_name: str,
    sampler_config: dict[str, float],
) -> None:
    model = _transport_model_stub()
    model.inference_defaults = {
        "sampling_schedule": {
            "schedule_type": "linear",
            "sigma_max": 1.0,
            "sigma_min": 0.02,
            "num_steps": 6,
        },
        "sampler": {"sampler": sampler_name, **sampler_config},
    }
    model.n_step_output = {"dataset_a": 2, "dataset_b": 2}
    model.num_output_channels = {"dataset_a": 3, "dataset_b": 2}
    model.transport_model_objective = EDMDiffusionModelObjective()
    model.edm = EdmSettings(sigma_data=1.0)
    _configure_sampling_model(model, {"dataset_a": (4, 3, 5), "dataset_b": (6, 2, 7)})

    def _network(
        x: Batch,
        conditioned_target: Batch,
        condition: dict[str, torch.Tensor],
        model_comm_group=None,
        target_forcing=None,
    ) -> Batch:
        del model_comm_group, target_forcing
        out = {}
        for dataset_name, target_data in ((n, s.data) for n, s in conditioned_target.items()):
            condition_data = condition[dataset_name]
            assert condition_data.shape == (
                target_data.shape[0],
                1,
                target_data.shape[2],
                1,
                1,
            )
            assert condition_data.dtype == target_data.dtype == x[dataset_name].data.dtype
            out[dataset_name] = 0.8 * target_data + 0.02 * condition_data
        return conditioned_target.with_data(out)

    model._forward_transport_network = _network

    x = {
        "dataset_a": torch.randn(2, 3, 1, 5, 4, dtype=torch.float32),
        "dataset_b": torch.randn(2, 2, 1, 7, 6, dtype=torch.bfloat16),
    }

    out = model.sample(_sampling_batch(model, x), target_template=_target_template(model, x))

    assert set(out.keys()) == set(x.keys())
    assert out["dataset_a"].data.shape == (2, 2, 1, 5, 3)
    assert out["dataset_b"].data.shape == (2, 2, 1, 7, 2)
    assert out["dataset_a"].data.dtype == x["dataset_a"].dtype
    assert out["dataset_b"].data.dtype == x["dataset_b"].dtype
    assert torch.isfinite(out["dataset_a"].data).all()
    assert torch.isfinite(out["dataset_b"].data).all()


def test_sampling_statistics_follow_output_variable_order() -> None:
    model = _transport_model_stub()
    _configure_sampling_model(model, {"data": (2, 2, 3)})
    model.n_step_output = {"data": 1}
    template = _target_template(model, {"data": torch.zeros(1, 1, 1, 3, 2)})
    model.data_indices["data"].model.output.ordered_names = ("out_1", "in_0")
    model.statistics["data"] = {
        "mean": torch.tensor([10.0, 20.0, 30.0, 40.0]),
        "stdev": torch.tensor([1.0, 2.0, 3.0, 4.0]),
    }
    model.num_output_channels = {"data": 2}
    batch = model._sampling_template(template, _sampling_batch(model, {"data": torch.zeros(1, 1, 1, 3, 2)}))
    assert batch["data"].variables == ["out_1", "in_0"]
    torch.testing.assert_close(batch["data"].statistics["mean"], torch.tensor([40.0, 10.0]))
    torch.testing.assert_close(batch["data"].statistics["stdev"], torch.tensor([4.0, 1.0]))


def test_sampling_allows_empty_statistics_for_unprocessed_data() -> None:
    model = _transport_model_stub()
    _configure_sampling_model(model, {"data": (1, 1, 3)})
    model.n_step_output = {"data": 1}
    model.statistics["data"] = {}
    template = _target_template(model, {"data": torch.zeros(1, 1, 1, 3, 1)})
    model.num_output_channels = {"data": 1}
    batch = model._sampling_template(template, _sampling_batch(model, {"data": torch.zeros(1, 1, 1, 3, 1)}))
    assert batch["data"].statistics == {}
    assert batch["data"].data.shape == (1, 1, 1, 3, 1)


def test_sampling_batch_preserves_sparse_ensemble_template_layout() -> None:
    model = _transport_model_stub()
    _configure_sampling_model(model, {"obs": (1, 1, 3)})
    model.is_dataset_static["obs"] = False
    layout = TensorLayout(ensemble=0, grid=1, variables=2)
    coordinates = [torch.zeros(3, 2), torch.ones(2, 2)]
    obs = build_batch(
        data={"obs": [torch.empty(2, 3, 0), torch.empty(2, 2, 0)]},
        layouts={"obs": layout},
        coordinates={"obs": coordinates},
        timedeltas={"obs": [torch.zeros(3), torch.zeros(2)]},
        boundaries={"obs": [(slice(0, 3),), (slice(0, 2),)]},
        variables={"obs": []},
    )
    model.num_output_channels = {"obs": 1}
    model.n_step_output = {"obs": 1}

    batch = model._sampling_template(_templates(obs), obs)

    assert batch["obs"].layout == layout
    assert batch["obs"].ensemble_size == 2
    assert [coords.shape[0] for coords in batch["obs"].coordinates] == [3, 2]
    assert [sample.shape for sample in batch["obs"].data] == [(2, 3, 1), (2, 2, 1)]


def test_sampling_template_describes_interface_templates_in_output_variables() -> None:
    """Templates as the inference interface builds them become zero-stride sources in output variables."""
    model = _transport_model_stub()
    _configure_sampling_model(model, {"ds": (2, 3, 4)})
    model.n_step_output = {"ds": 2}
    model.num_output_channels = {"ds": 3}
    # Per-sample layout, no data, input-side variables; templates of datasets that are not decoded are skipped.
    template = GriddedTemplate(
        name="ds",
        variables=["in_0"],
        layout=TensorLayout(time=0, ensemble=1, grid=2, variables=3),
        coordinates=torch.zeros(4, 2),
        batch_size=1,
        ensemble_size=1,
        time_size=2,
    )
    x = _sampling_batch(model, {"ds": torch.zeros(1, 2, 1, 4, 2)})

    batch = model._sampling_template({"ds": template, "input_only": template}, x)

    assert list(batch.keys()) == ["ds"]
    source = batch["ds"]
    assert source.variables == ["out_0", "out_1", "out_2"]
    assert source.layout == TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
    assert source.data.shape == (1, 2, 1, 4, 3)
    assert set(source.data.stride()) == {0}  # zero-stride view

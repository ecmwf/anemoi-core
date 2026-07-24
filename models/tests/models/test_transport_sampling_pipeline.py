# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import pytest
import torch

from anemoi.models.data import Batch
from anemoi.models.data.tensor_layout import TensorLayout
from anemoi.models.data.views import create_source_view
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


class IdentityProcessor(torch.nn.Module):
    def forward(self, x: torch.Tensor, in_place: bool = True, inverse: bool = False, **kwargs):
        del inverse, kwargs
        if not in_place:
            x = x.clone()
        return x


def _transport_model_stub() -> AnemoiTransportModelEncProcDec:
    model = AnemoiTransportModelEncProcDec.__new__(AnemoiTransportModelEncProcDec)
    model.transport_source = TransportSourceBuilder()
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
    """Attach the metadata needed by ``_make_sampling_batch`` to a lightweight model stub.

    ``specs`` maps dataset name to ``(num_input_channels, num_output_channels, grid_size)``.
    """
    model.data_indices = {}
    model.statistics = {}
    model.is_dataset_static = {}
    model._graph_data = {}
    for dataset_name, (num_inputs, num_outputs, grid_size) in specs.items():
        input_names = tuple(f"in_{idx}" for idx in range(num_inputs))
        output_names = tuple(f"out_{idx}" for idx in range(num_outputs))
        model.data_indices[dataset_name] = _data_indices(input_names, output_names)
        model.statistics[dataset_name] = {
            "mean": torch.zeros(num_inputs + num_outputs),
            "stdev": torch.ones(num_inputs + num_outputs),
        }
        model.is_dataset_static[dataset_name] = True
        model._graph_data[dataset_name] = SimpleNamespace(x=torch.zeros(grid_size, 2))


def _sampling_batch(model: AnemoiTransportModelEncProcDec, data: dict[str, torch.Tensor]) -> Batch:
    return model._make_sampling_batch(data, variable_space="input")


def _target_template(model: AnemoiTransportModelEncProcDec, data: dict[str, torch.Tensor]) -> Batch:
    template_data = {
        name: torch.empty(
            sample.shape[0],
            model.n_step_output,
            sample.shape[2],
            sample.shape[-2],
            0,
            device=sample.device,
            dtype=sample.dtype,
        )
        for name, sample in data.items()
    }
    return model._make_sampling_batch(template_data, variable_space="output")


def _sparse_batch(
    *,
    name: str,
    data_shapes: list[tuple[int, int]],
    variables: list[str],
) -> Batch:
    data = [torch.zeros(shape, dtype=torch.float32) for shape in data_shapes]
    coordinates = [torch.full((shape[0], 2), float(index)) for index, shape in enumerate(data_shapes)]
    return Batch(
        data={name: data},
        coordinates={name: coordinates},
        metadata={name: {"boundaries": [(slice(0, shape[0]),) for shape in data_shapes]}},
        layouts={name: TensorLayout(grid=0, variables=1, time_in_grid=True)},
        variables={name: variables},
        statistics={name: {}},
    )


def _sparse_target_template(
    *,
    name: str,
    node_counts: list[int],
    variables: list[str],
) -> Batch:
    return _sparse_batch(
        name=name,
        data_shapes=[(node_count, 0) for node_count in node_counts],
        variables=variables,
    )


def test_transport_conditioning_embedding_uses_compact_condition_width() -> None:
    model = _transport_model_stub()
    model._graph_name_hidden = "hidden"
    model.node_attributes = SimpleNamespace(num_nodes={"data": 4, "hidden": 5})
    cond_dim = 8
    model._embed_noise_conditioning = lambda sigma: torch.ones(
        (*sigma.shape[:-1], cond_dim),
        device=sigma.device,
        dtype=sigma.dtype,
    )

    x = Batch(
        data={"data": torch.empty(2, 2, 3, 4, 1)},
        coordinates={"data": torch.zeros(4, 2)},
        metadata={"static_coords": frozenset({"data"})},
        layouts={"data": TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)},
        variables={"data": ["a"]},
        statistics={"data": {}},
    )
    condition = {"data": torch.zeros(2, 1, 3, 1, 1)}

    fwd_mapper_kwargs, processor_kwargs, bwd_mapper_kwargs = model._build_conditioning_kwargs(x, condition)

    data_cond, hidden_cond = fwd_mapper_kwargs["data"]["cond"]
    hidden_back_cond, data_back_cond = bwd_mapper_kwargs["data"]["cond"]
    assert data_cond.shape == data_back_cond.shape == (2 * 3 * 4, cond_dim)
    assert hidden_cond.shape == hidden_back_cond.shape == processor_kwargs["cond"].shape == (2 * 3 * 5, cond_dim)


def test_transport_conditioning_uses_sparse_target_node_counts() -> None:
    model = _transport_model_stub()
    model._graph_name_hidden = "hidden"
    model.node_attributes = SimpleNamespace(num_nodes={"hidden": 5})
    cond_dim = 6
    model._embed_noise_conditioning = lambda sigma: torch.ones(
        (*sigma.shape[:-1], cond_dim),
        device=sigma.device,
        dtype=sigma.dtype,
    )

    layout = TensorLayout(grid=0, variables=1, time_in_grid=True)
    target = Batch(
        data={"obs": [torch.empty(2, 1), torch.empty(4, 1)]},
        coordinates={"obs": [torch.zeros(2, 2), torch.zeros(4, 2)]},
        metadata={"obs": {"boundaries": [(slice(0, 2),), (slice(0, 4),)]}},
        layouts={"obs": layout},
        variables={"obs": ["a"]},
        statistics={"obs": {}},
    )
    condition = {"obs": torch.zeros(2, 1, 1, 1, 1)}

    fwd_mapper_kwargs, processor_kwargs, bwd_mapper_kwargs = model._build_conditioning_kwargs(target, condition)

    data_cond, hidden_cond = fwd_mapper_kwargs["obs"]["cond"]
    hidden_back_cond, data_back_cond = bwd_mapper_kwargs["obs"]["cond"]
    assert data_cond.shape == data_back_cond.shape == (6, cond_dim)
    assert hidden_cond.shape == hidden_back_cond.shape == processor_kwargs["cond"].shape == (2 * 5, cond_dim)


def test_transport_assemble_input_uses_sparse_target_coordinates_when_obs_do_not_align() -> None:
    model = _transport_model_stub()
    model.node_attributes = _EmptyNodeAttributes()
    layout = TensorLayout(grid=0, variables=1, time_in_grid=True)
    x = create_source_view(
        name="obs",
        data=[torch.ones(2, 2)],
        coordinates=[torch.tensor([[0.0, 0.0], [0.1, 0.1]])],
        variables=["a", "b"],
        statistics={},
        is_static=False,
        layout=layout,
        boundaries=[(slice(0, 2),)],
    )
    y_noised = create_source_view(
        name="obs",
        data=[torch.full((3, 1), 5.0)],
        coordinates=[torch.tensor([[0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])],
        variables=["a"],
        statistics={},
        is_static=False,
        layout=layout,
        boundaries=[(slice(0, 3),)],
    )

    data_coords, x_data_latent, x_skip, shard_sizes = model._assemble_input(x, y_noised, bse=1, dataset_name="obs")

    torch.testing.assert_close(data_coords, y_noised.coordinates[0])
    assert x_skip is None
    assert shard_sizes is None
    assert x_data_latent.shape == (3, 2 + 1 + 4)
    torch.testing.assert_close(x_data_latent[:, :2], torch.zeros(3, 2))
    torch.testing.assert_close(x_data_latent[:, 2:3], torch.full((3, 1), 5.0))


def test_tendency_transport_assemble_input_uses_dense_source_views_with_residual_conditioning() -> None:
    model = AnemoiTransportTendModelEncProcDec.__new__(AnemoiTransportTendModelEncProcDec)
    model.node_attributes = _EmptyNodeAttributes()
    model.condition_on_residual = True
    model.n_step_output = 1
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
    x = create_source_view(
        name="data",
        data=x_data,
        coordinates=coordinates,
        variables=["a", "b", "c", "d"],
        statistics={},
        is_static=True,
        layout=layout,
    )
    y_noised = create_source_view(
        name="data",
        data=y_noised_data,
        coordinates=coordinates,
        variables=["a", "c"],
        statistics={},
        is_static=True,
        layout=layout,
    )

    data_coords, x_data_latent, x_skip, shard_sizes = model._assemble_input(
        x,
        y_noised,
        bse=1,
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

    layout = TensorLayout(grid=0, variables=1, time_in_grid=True)
    sparse_view = create_source_view(
        name="obs",
        data=[torch.ones(2, 1)],
        coordinates=[torch.zeros(2, 2)],
        variables=["a"],
        statistics={},
        is_static=False,
        layout=layout,
        boundaries=[(slice(0, 2),)],
    )

    with pytest.raises(NotImplementedError, match="Tendency transport.*sparse"):
        model._assemble_input(sparse_view, sparse_view, bse=1, dataset_name="obs")


def test_tendency_transport_forward_network_uses_dense_source_view_override() -> None:
    model = AnemoiTransportTendModelEncProcDec.__new__(AnemoiTransportTendModelEncProcDec)
    model._graph_name_hidden = "hidden"
    model.node_attributes = _EmptyNodeAttributes()
    model.condition_on_residual = True
    model.n_step_output = 1
    model._internal_input_idx = {"data": torch.tensor([0, 1])}
    model.latent_skip = False
    model.use_encoder_data_output = {"data": True}
    model._hidden_coordinates = lambda: torch.zeros(5, 2)
    model._build_conditioning_kwargs = lambda *_args, **_kwargs: ({"data": {}}, {}, {"data": {}})
    model.boundings = {"data": torch.nn.Identity()}

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

    class _Encoder:
        def __init__(self) -> None:
            self.source_width = None

        def __call__(self, x, **_kwargs):
            self.source_width = x[0].shape[-1]
            return x[0], torch.ones_like(x[1])

    class _Processor:
        def __call__(self, x, **_kwargs):
            return x

    class _Decoder:
        def __call__(self, x, **_kwargs):
            target_features = x[1]
            return torch.zeros(target_features.shape[0], 2)

    encoder = _Encoder()
    model.encoder = {"data": encoder}
    model.processor = _Processor()
    model.decoder = {"data": _Decoder()}
    model.encoder_graph_provider = {"data": _GraphProvider()}
    model.processor_graph_provider = _GraphProvider()
    model.decoder_graph_provider = {"data": _GraphProvider()}

    layout = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
    batch = Batch(
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
    assert isinstance(out, Batch)
    assert out.data["data"].shape == (1, 1, 1, 3, 2)


def test_transport_target_dim_combines_corrupted_target_and_decoding_forcings() -> None:
    """The obs transport decoder consumes outputs + decoding forcings + coordinates."""
    model = _transport_model_stub()
    model.use_encoder_data_output = {"obs": False}
    model.is_dataset_static = {"obs": False}
    model.num_output_channels = {"obs": 3}
    model.num_input_channels_decoding_forcings = {"obs": 12}
    model.node_attributes = SimpleNamespace(num_trainable_parameters={})

    coords_dim = 4
    assert model._calculate_target_dim("obs") == 3 + 12 + coords_dim


def test_transport_decoder_uses_target_assembly_when_encoder_data_output_is_disabled() -> None:
    model = _transport_model_stub()
    model.use_encoder_data_output = {"obs": False}
    model._graph_name_hidden = "hidden"
    model.node_attributes = _EmptyNodeAttributes()
    model.latent_skip = False
    model._get_consistent_dim = lambda _batch, dim: 1
    model._resolve_in_out_sharded = lambda batch: {dataset_name: False for dataset_name in batch.keys()}
    model._assert_valid_sharding = lambda *_args, **_kwargs: None
    model._build_conditioning_kwargs = lambda *_args, **_kwargs: ({"obs": {}}, {}, {"obs": {}})
    model._hidden_coordinates = lambda: torch.zeros(5, 2)
    model._assemble_input = lambda *_args, **_kwargs: (
        torch.zeros(3, 2),
        torch.zeros(3, 22),
        None,
        None,
    )

    expected_target_data_latent = torch.zeros(3, 4)
    assembled_views = {}

    def _assemble_target_stub(view, *_args, **_kwargs):
        assembled_views["obs"] = view
        return (
            torch.zeros(3, 2),
            expected_target_data_latent,
            None,
            None,
        )

    model._assemble_target = _assemble_target_stub
    model._assemble_output = lambda _x_out, _x_skip, target, _dtype, _dataset_name: target.clone(
        data=[torch.zeros(3, 1)],
    )

    class _Encoder:
        def __call__(self, x, **_kwargs):
            return torch.zeros(3, 22), torch.ones_like(x[1])

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
    model.encoder = {"obs": _Encoder()}
    model.processor = _Processor()
    model.decoder = {"obs": decoder}
    model.encoder_graph_provider = {"obs": _GraphProvider()}
    model.processor_graph_provider = _GraphProvider()
    model.decoder_graph_provider = {"obs": _GraphProvider()}

    layout = TensorLayout(grid=0, variables=1, time_in_grid=True)
    batch = Batch(
        data={"obs": [torch.ones(3, 1)]},
        coordinates={"obs": [torch.zeros(3, 2)]},
        metadata={"obs": {"boundaries": [(slice(0, 3),)]}},
        layouts={"obs": layout},
        variables={"obs": ["a"]},
        statistics={"obs": {}},
    )

    target_forcing = batch.with_data({"obs": [torch.full((3, 2), 5.0)]})
    model._forward_transport_network(
        batch,
        batch,
        {"obs": torch.zeros(1, 1, 1, 1, 1)},
        target_forcing=target_forcing,
    )

    assert decoder.destination_features is expected_target_data_latent
    assert decoder.destination_features.shape == (3, 4)
    # The decoder target view combines the corrupted target with the forcings.
    assembled_data = assembled_views["obs"].data[0]
    assert assembled_data.shape == (3, 3)
    torch.testing.assert_close(assembled_data[:, 1:], torch.full((3, 2), 5.0))


def test_transport_conditioning_rejects_expanded_condition_shape() -> None:
    model = _transport_model_stub()
    expanded_condition = {"data": torch.zeros(2, 4, 3, 7, 1)}

    with pytest.raises(AssertionError, match="Expected condition to have shape"):
        model._assert_condition_shapes(expanded_condition)


def test_before_sampling_non_sharded_returns_none_grid_shapes() -> None:
    model = _transport_model_stub()
    _configure_sampling_model(model, {"data": (2, 2, 3)})

    batch = {"data": torch.randn(2, 4, 3, 2)}
    pre_processors = {"data": IdentityProcessor()}

    (xs,), grid_shard_sizes = model._before_sampling(
        batch,
        pre_processors,
        n_step_input=3,
        model_comm_group=None,
    )

    assert grid_shard_sizes is None
    assert isinstance(xs, Batch)
    assert xs.data["data"].shape == (2, 3, 1, 3, 2)


def test_after_sampling_postprocesses_source_views_and_returns_data() -> None:
    model = _transport_model_stub()
    layout = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
    data = torch.zeros(1, 1, 1, 3, 2)
    out = Batch(
        data={"data": data},
        coordinates={"data": torch.zeros(3, 2)},
        metadata={"static_coords": frozenset({"data"})},
        layouts={"data": layout},
        variables={"data": ["a", "b"]},
        statistics={"data": {}},
    )

    class CaptureProcessor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen = None

        def forward(self, view, in_place: bool = True, **_kwargs):
            assert in_place is False
            self.seen = view
            return view.clone(data=view.data + 1)

    processor = CaptureProcessor()

    result = model._after_sampling(
        out,
        post_processors={"data": processor},
        before_sampling_data=(out,),
    )

    assert processor.seen is not None
    assert processor.seen.data is data
    torch.testing.assert_close(result["data"], data + 1)


def test_make_sampling_batch_shards_full_template_coordinates_for_local_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _transport_model_stub()
    _configure_sampling_model(model, {"data": (1, 1, 5)})
    model.n_step_output = 1

    template = Batch(
        data={"data": torch.empty(1, 1, 1, 5, 0)},
        coordinates={"data": torch.arange(10, dtype=torch.float32).reshape(5, 2)},
        metadata={"static_coords": frozenset({"data"})},
        layouts={"data": TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)},
        variables={"data": ["out_0"]},
        statistics={"data": {}},
    )
    local_data = {"data": torch.zeros(1, 1, 1, 2, 1)}
    calls = []

    def fake_shard_tensor(input_, dim, sizes, model_comm_group):
        calls.append((tuple(input_.shape), dim, sizes, model_comm_group))
        return input_.narrow(dim, 0, sizes[0])

    monkeypatch.setattr(
        "anemoi.models.models.transport_encoder_processor_decoder.shard_tensor",
        fake_shard_tensor,
    )

    model_comm_group = object()
    out = model._make_sampling_batch(
        local_data,
        variable_space="output",
        template=template,
        model_comm_group=model_comm_group,
        grid_shard_sizes={"data": [2, 3]},
    )

    torch.testing.assert_close(out.coordinates["data"], template.coordinates["data"][:2])
    assert out.data["data"].shape[-2] == out.coordinates["data"].shape[0]
    assert len(calls) == 1
    shape, dim, sizes, group = calls[0]
    assert shape == (5, 2)
    assert dim == 0
    assert sizes == [2, 3]
    assert group is model_comm_group


def test_predict_step_iterates_items_and_casts_each_dataset_dtype() -> None:
    batch = {
        "ds_a": torch.randn(1, 3, 4, 2, dtype=torch.float32),
        "ds_b": torch.randn(1, 3, 4, 2, dtype=torch.bfloat16),
    }

    x_for_sampling = {
        "ds_a": torch.randn(1, 2, 1, 4, 2, dtype=torch.float32),
        "ds_b": torch.randn(1, 2, 1, 4, 2, dtype=torch.bfloat16),
    }
    model = _transport_model_stub()
    _configure_sampling_model(model, {"ds_a": (2, 3, 4), "ds_b": (2, 3, 4)})
    model.n_step_output = 2

    x_for_sampling_batch = _sampling_batch(model, x_for_sampling)
    target_template = _target_template(model, x_for_sampling)
    model._before_sampling = lambda *_args, **_kwargs: ((x_for_sampling_batch,), None)
    model.sample = lambda *_args, **_kwargs: x_for_sampling_batch.with_data(
        {
            "ds_a": torch.randn(1, 2, 1, 4, 3, dtype=torch.float64),
            "ds_b": torch.randn(1, 2, 1, 4, 3, dtype=torch.float64),
        }
    )

    def _after_sampling_spy(
        out,
        _post_processors,
        _before_sampling_data,
        _model_comm_group,
        _grid_shard_sizes,
        _gather_out,
        **_kwargs,
    ):
        assert out.data["ds_a"].dtype == batch["ds_a"].dtype
        assert out.data["ds_b"].dtype == batch["ds_b"].dtype
        return out.data

    model._after_sampling = _after_sampling_spy

    out = model.predict_step(
        batch=batch,
        pre_processors={"ds_a": IdentityProcessor(), "ds_b": IdentityProcessor()},
        post_processors={"ds_a": IdentityProcessor(), "ds_b": IdentityProcessor()},
        n_step_input=2,
        target_template=target_template,
    )

    assert out["ds_a"].dtype == torch.float32
    assert out["ds_b"].dtype == torch.bfloat16


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
            grid_shard_sizes=None,
            **kwargs,
        ):
            del denoising_fn, model_comm_group, grid_shard_sizes, kwargs
            assert isinstance(sigmas, torch.Tensor)
            assert sigmas.shape == (5,)
            assert sigmas[-1] == 0.0
            for dataset_name, y_data in y.data.items():
                assert y_data.dtype == sigmas.dtype
                assert y_data.shape[:4] == (
                    x.data[dataset_name].shape[0],
                    2,
                    x.data[dataset_name].shape[2],
                    x.data[dataset_name].shape[-2],
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
    model.n_step_output = 2
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
            grid_shard_sizes=None,
            **kwargs,
        ):
            del x, sigmas, denoising_fn, model_comm_group, grid_shard_sizes, kwargs
            assert [tuple(sample.shape) for sample in y.data["obs"]] == [(3, 1), (1, 1)]
            assert [tuple(coords.shape) for coords in y.coordinates["obs"]] == [(3, 2), (1, 2)]
            return y

    model = _transport_model_stub()
    model.inference_defaults = {
        "sampling_schedule": {"schedule_type": "linear", "sigma_max": 1.0, "sigma_min": 0.1, "num_steps": 2},
        "sampler": {"sampler": "spy"},
    }
    model.n_step_output = 1
    model.num_output_channels = {"obs": 1}
    model.transport_model_objective = EDMDiffusionModelObjective()
    model.edm = EdmSettings(sigma_data=1.0)
    _configure_sampling_model(model, {"obs": (2, 1, 1)})

    monkeypatch.setitem(transport_samplers.DIFFUSION_SAMPLERS, "spy", SpySampler)

    x = _sparse_batch(name="obs", data_shapes=[(2, 2), (4, 2)], variables=["in_0", "in_1"])
    target_template = _sparse_target_template(name="obs", node_counts=[3, 1], variables=["out_0"])

    out = model.sample(x, target_template=target_template)

    assert [tuple(sample.shape) for sample in out.data["obs"]] == [(3, 1), (1, 1)]


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
            grid_shard_sizes=None,
            **kwargs,
        ):
            del x, times, vector_field_fn, model_comm_group, grid_shard_sizes, kwargs
            assert [tuple(sample.shape) for sample in y.data["obs"]] == [(5, 1), (2, 1)]
            assert [tuple(coords.shape) for coords in y.coordinates["obs"]] == [(5, 2), (2, 2)]
            return y

    model = _transport_model_stub()
    model.transport_model_objective = StochasticInterpolantModelObjective()
    model.inference_defaults = {
        "sampling_schedule": {"schedule_type": "unit_time", "num_steps": 2},
        "sampler": {"sampler": "spy_vector"},
    }
    model.n_step_output = 1
    model.num_output_channels = {"obs": 1}
    _configure_sampling_model(model, {"obs": (2, 1, 1)})

    monkeypatch.setitem(transport_samplers.VECTOR_FIELD_SAMPLERS, "spy_vector", SpyVectorFieldSampler)

    x = _sparse_batch(name="obs", data_shapes=[(2, 2), (4, 2)], variables=["in_0", "in_1"])
    target_template = _sparse_target_template(name="obs", node_counts=[5, 2], variables=["out_0"])

    out = model.sample(x, target_template=target_template)

    assert [tuple(sample.shape) for sample in out.data["obs"]] == [(5, 1), (2, 1)]


def test_transport_sampling_requires_target_template() -> None:
    model = _transport_model_stub()
    model.inference_defaults = {
        "sampling_schedule": {"schedule_type": "linear", "sigma_max": 1.0, "sigma_min": 0.1, "num_steps": 2},
        "sampler": {"sampler": "heun"},
    }
    model.n_step_output = 1
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
            grid_shard_sizes=None,
            **kwargs,
        ):
            del sigmas, kwargs
            return denoising_fn(
                x,
                y,
                {"obs": torch.ones(2, 1, 1, 1, 1)},
                model_comm_group,
                grid_shard_sizes,
            )

    model = AnemoiTransportTendModelEncProcDec.__new__(AnemoiTransportTendModelEncProcDec)
    model.transport_source = TransportSourceBuilder()
    model.transport_model_objective = EDMDiffusionModelObjective()
    model.inference_defaults = {
        "sampling_schedule": {"schedule_type": "linear", "sigma_max": 1.0, "sigma_min": 0.1, "num_steps": 2},
        "sampler": {"sampler": "calling"},
    }
    model.edm = EdmSettings(sigma_data=1.0)
    model.n_step_output = 1
    model.num_output_channels = {"obs": 1}
    model._graph_name_hidden = "hidden"
    model.node_attributes = _EmptyNodeAttributes()
    model._hidden_coordinates = lambda: torch.zeros(5, 2)
    model._get_consistent_dim = lambda _batch, dim: 2 if dim == 0 else 1
    model._resolve_in_out_sharded = lambda batch: {dataset_name: False for dataset_name in batch.keys()}
    model._assert_valid_sharding = lambda *_args, **_kwargs: None
    model._build_conditioning_kwargs = lambda *_args, **_kwargs: ({"obs": {}}, {}, {"obs": {}})
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
            grid_shard_sizes=None,
            **kwargs,
        ):
            del kwargs
            assert times.shape == (4,)
            assert times[0] == 0.0
            assert times[-1] == 1.0
            torch.testing.assert_close(y.data["ds_a"], torch.zeros_like(y.data["ds_a"]))
            return vector_field_fn(
                x,
                y,
                {"ds_a": torch.zeros(1, 1, 1, 1, 1)},
                model_comm_group,
                grid_shard_sizes,
            )

    model = _transport_model_stub()
    model.transport_model_objective = StochasticInterpolantModelObjective()
    model.inference_defaults = {
        "sampling_schedule": {"schedule_type": "unit_time", "num_steps": 3},
        "sampler": {"sampler": "heun"},
    }
    model.n_step_output = 2
    model.num_output_channels = {"ds_a": 3}
    model._forward_transport_network = lambda _x, y, *_args, **_kwargs: y
    _configure_sampling_model(model, {"ds_a": (6, 3, 5)})
    model.build_sampling_source = lambda x, **_kwargs: {
        "ds_a": torch.zeros(
            x.data["ds_a"].shape[0],
            model.n_step_output,
            x.data["ds_a"].shape[2],
            x.data["ds_a"].shape[-2],
            model.num_output_channels["ds_a"],
            device=x.data["ds_a"].device,
            dtype=x.data["ds_a"].dtype,
        )
    }

    monkeypatch.setitem(transport_samplers.VECTOR_FIELD_SAMPLERS, "heun", DummyVectorFieldSampler)

    x = {"ds_a": torch.randn(1, 3, 1, 5, 6, dtype=torch.float32)}

    out = model.sample(_sampling_batch(model, x), target_template=_target_template(model, x))

    assert set(out.keys()) == {"ds_a"}
    assert out.data["ds_a"].shape == (1, 2, 1, 5, 3)


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
            grid_shard_sizes=None,
            **kwargs,
        ):
            del kwargs
            assert times.shape == (4,)
            assert times[0] == 0.0
            assert times[-1] == 1.0
            torch.testing.assert_close(y.data["ds_a"], torch.zeros_like(y.data["ds_a"]))
            return vector_field_fn(
                x,
                y,
                {"ds_a": torch.zeros(1, 1, 1, 1, 1)},
                model_comm_group,
                grid_shard_sizes,
            )

    model = _transport_model_stub()
    model.transport_model_objective = StochasticInterpolantModelObjective()
    model.inference_defaults = {
        "sampling_schedule": {"schedule_type": "unit_time", "num_steps": 3},
        "sampler": {"sampler": "heun"},
    }
    model.n_step_output = 2
    model.num_output_channels = {"ds_a": 3}
    model._forward_transport_network = lambda _x, y, *_args, **_kwargs: y
    _configure_sampling_model(model, {"ds_a": (6, 3, 5)})
    model.build_sampling_source = lambda x, **_kwargs: {
        "ds_a": torch.zeros(
            x.data["ds_a"].shape[0],
            model.n_step_output,
            x.data["ds_a"].shape[2],
            x.data["ds_a"].shape[-2],
            model.num_output_channels["ds_a"],
            device=x.data["ds_a"].device,
            dtype=x.data["ds_a"].dtype,
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
    assert out.data["ds_a"].shape == (1, 2, 1, 5, 3)


def test_transport_source_builder_does_not_build_unselected_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = TransportSourceBuilder(TransportSourceSettings(kind="gaussian", scale=2.0))
    target = {"data": torch.zeros(1, 1, 1, 2, 1)}

    def reference_source_factory() -> dict[str, torch.Tensor]:
        raise AssertionError("reference source should not be built")

    def fake_randn(shape, device=None, dtype=None):
        return torch.full(shape, 3.0, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randn", fake_randn)

    source = builder.build(
        TransportSourceRequest.from_data(
            target,
            default_kind="reference_state",
            custom_source_factories={"reference_state": reference_source_factory},
        )
    )

    torch.testing.assert_close(source["data"], torch.full_like(target["data"], 6.0))


def test_transport_source_builder_postprocesses_reference_source(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = TransportSourceBuilder(TransportSourceSettings(kind="reference_state", scale=0.5, noise_scale=0.25))
    target = {"data": torch.zeros(1, 1, 1, 2, 1, dtype=torch.float64)}
    reference = {"data": torch.full_like(target["data"], 4.0)}

    monkeypatch.setattr(
        torch, "randn", lambda shape, device=None, dtype=None: torch.full(shape, 2.0, device=device, dtype=dtype)
    )

    source = builder.build(
        TransportSourceRequest.from_data(
            target,
            default_kind="gaussian",
            custom_source_factories={"reference_state": lambda: reference},
        )
    )

    assert source["data"].dtype == target["data"].dtype
    torch.testing.assert_close(source["data"], torch.full_like(target["data"], 2.5))
    torch.testing.assert_close(reference["data"], torch.full_like(target["data"], 4.0))


def test_tendency_sampling_source_can_use_reference_state() -> None:
    model = AnemoiTransportTendModelEncProcDec.__new__(AnemoiTransportTendModelEncProcDec)
    model.transport_source = TransportSourceBuilder(TransportSourceSettings(kind="reference_state"))
    model.n_step_output = 2
    model.num_output_channels = {"ds_a": 2}
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
    x = Batch(
        data={"ds_a": x_data},
        coordinates={"ds_a": torch.zeros(5, 2)},
        metadata={"static_coords": frozenset({"ds_a"})},
        layouts={"ds_a": TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)},
        variables={"ds_a": ["a", "c", "b", "d"]},
        statistics={"ds_a": {}},
    )

    source = model.build_sampling_source(x, target_template=x.with_data({"ds_a": x_data[..., :0]}))

    expected = x_data[:, -1:, :, :, :].index_select(-1, torch.tensor([0, 2])).expand(-1, 2, -1, -1, -1)
    torch.testing.assert_close(source["ds_a"], expected)


def test_stochastic_interpolant_objective_returns_raw_drift_prediction() -> None:
    """The stochastic-interpolant model objective leaves drift predictions in model-output space."""
    interpolant = Batch(
        data={"data": torch.full((1, 1, 1, 2, 1), 2.0)},
        coordinates={"data": torch.zeros(2, 2)},
        metadata={"static_coords": frozenset({"data"})},
        layouts={"data": TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)},
        variables={"data": ["x"]},
        statistics={"data": {}},
    )
    time_level = {"data": torch.full_like(interpolant.data["data"], 0.25)}
    drift = interpolant.with_data({"data": torch.full_like(interpolant.data["data"], 0.5)})
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
        interpolant.with_data({"data": torch.zeros_like(interpolant.data["data"])}),
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
    model.n_step_output = 2
    model.num_output_channels = {"dataset_a": 3, "dataset_b": 2}
    model.transport_model_objective = EDMDiffusionModelObjective()
    model.edm = EdmSettings(sigma_data=1.0)
    _configure_sampling_model(model, {"dataset_a": (4, 3, 5), "dataset_b": (6, 2, 7)})

    def _network(
        x: Batch,
        conditioned_target: Batch,
        condition: dict[str, torch.Tensor],
        model_comm_group=None,
        grid_shard_sizes=None,
        target_forcing=None,
    ) -> Batch:
        del model_comm_group, grid_shard_sizes, target_forcing
        out = {}
        for dataset_name, target_data in conditioned_target.data.items():
            condition_data = condition[dataset_name]
            assert condition_data.shape == (
                target_data.shape[0],
                1,
                target_data.shape[2],
                1,
                1,
            )
            assert condition_data.dtype == target_data.dtype == x.data[dataset_name].dtype
            out[dataset_name] = 0.8 * target_data + 0.02 * condition_data
        return conditioned_target.with_data(out)

    model._forward_transport_network = _network

    x = {
        "dataset_a": torch.randn(2, 3, 1, 5, 4, dtype=torch.float32),
        "dataset_b": torch.randn(2, 2, 1, 7, 6, dtype=torch.bfloat16),
    }

    out = model.sample(_sampling_batch(model, x), target_template=_target_template(model, x))

    assert set(out.keys()) == set(x.keys())
    assert out.data["dataset_a"].shape == (2, 2, 1, 5, 3)
    assert out.data["dataset_b"].shape == (2, 2, 1, 7, 2)
    assert out.data["dataset_a"].dtype == x["dataset_a"].dtype
    assert out.data["dataset_b"].dtype == x["dataset_b"].dtype
    assert torch.isfinite(out.data["dataset_a"]).all()
    assert torch.isfinite(out.data["dataset_b"]).all()

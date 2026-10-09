# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for AnemoiTransportSpatialDownscalerModelEncProcDec.

Unit tests build the model via ``__new__`` and wire routing by hand; they cover
error paths and the residual arithmetic. The ``test_real_construction_*`` tests
build a small real model and cover the happy path end to end.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportModelEncProcDec
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportSpatialDownscalerModelEncProcDec
from anemoi.models.transport import TransportSourceBuilder

# ── helpers ────────────────────────────────────────────────────────────────────


def _make_index_collection(
    name_to_index: dict[str, int],
    *,
    forcing: list[str] | None = None,
    diagnostic: list[str] | None = None,
    target: list[str] | None = None,
) -> IndexCollection:
    cfg = DictConfig(
        {
            "forcing": forcing or [],
            "diagnostic": diagnostic or [],
            "target": target or [],
        },
    )
    return IndexCollection(cfg, name_to_index)


def _make_downscaler_indices() -> dict[str, IndexCollection]:
    """Three datasets: in_lres (reference), in_hres (conditioning), out_hres (target).
    The target's prognostic variables (``t2m``, ``u10``) must also be
    prognostic in the reference dataset for ``_validate_residual_reference`` to accept the
    configuration (see ``_validate_prognostics_match``).
    """
    # in_lres: reference dataset — must expose the same variables as prognostic.
    in_lres = _make_index_collection({"t2m": 0, "u10": 1})
    # in_hres: conditioning-only; role of ``z`` is unconstrained.
    in_hres = _make_index_collection(
        {"z": 0},
        forcing=["z"],
    )
    # out_hres: two output variables, both prognostic (no forcing / diagnostic).
    out_hres = _make_index_collection({"t2m": 0, "u10": 1})
    return {"in_lres": in_lres, "in_hres": in_hres, "out_hres": out_hres}


class _StaticNodeAttributes:
    """Minimal stub for ``self.node_attributes`` used in the base model.
    Callable with ``(dataset_name, batch_size)`` and provides ``attr_ndims``.
    """

    def __init__(self, attr_ndims: dict[str, int], grid: int = 4) -> None:
        self.attr_ndims = attr_ndims
        self.grid = grid
        # Node counts come from the graph in the real class; every dataset here
        # shares one grid size.
        self.num_nodes = {name: grid for name in attr_ndims}

    def __call__(self, dataset_name: str, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size * self.grid, self.attr_ndims[dataset_name])


class _AdditiveProcessor:
    """Processor stub that adds ``offset`` and records the ``data_index`` and kwargs of each call.

    Pre/post processors are modelled by the sign of ``offset``.
    """

    def __init__(self, offset: float) -> None:
        self.offset = offset
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        x: torch.Tensor,
        in_place: bool = True,
        data_index: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert in_place is False, "Downscaler must call processors with in_place=False."
        self.calls.append(
            {"data_index": None if data_index is None else data_index.tolist(), "kwargs": kwargs},
        )
        return x + self.offset


class _IdentitySpatialProjector:
    """Spatial pre-processor stub that just passes input through.

    Records every call (tensor + kwargs) so tests can assert both the *order*
    of operations in ``_before_sampling`` and the arguments passed to the
    projector (in particular ``grid_shard_sizes``, which must be the
    source-grid shard sizes so behaviour matches the training path).

    Returns ``(x, output_grid_shard_sizes)`` like ``CrossGridProjector.forward``.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, Any]:
        self.calls.append({"x": x, "kwargs": kwargs})
        return x, kwargs.get("grid_shard_sizes")


def _wire_fused_encoder_routing(
    model: AnemoiTransportModelEncProcDec,
    *,
    sources: list[str],
    target: str = "out_hres",
    reference: str = "in_lres",
    fusing_strategy: str = "concatenate_inputs_along_variable_dim",
) -> None:
    """Attach the routing state ``BaseGraphModel`` and the transport base would produce.

    One encoder fuses ``sources`` on the first of them (the anchor), and ``target``
    is decoded from the anchor of its ``reference``. Config keys (``enc0``/``dec0``)
    deliberately differ from the dataset names, as in the graphtransformer_multi_* configs.
    """
    anchor = sources[0]
    model.encoder2datasets = {"enc0": list(sources)}
    model.encoder2anchors = {"enc0": [anchor]}
    model.encoder_fusing_strategy = {"enc0": fusing_strategy}
    model.dataset2anchor = {name: anchor for name in sources}
    model.dataset2encoder = {name: "enc0" for name in sources}
    model.input_datasets = [anchor]
    model.dataset2decoder = {target: "dec0"}
    model.decoder2datasets = {"dec0": [target]}
    model.decoders_target_input = {"dec0": SimpleNamespace(dim=0)}
    model.target_datasets = [target]
    model.target_anchors = {target: reference}
    model.target2anchor = {target: anchor}


def _make_bare_model(
    *,
    n_step_input: int = 1,
    n_step_output: int = 1,
    attr_ndims: dict[str, int] | None = None,
    grid: int = 4,
    data_indices: dict[str, IndexCollection] | None = None,
) -> AnemoiTransportSpatialDownscalerModelEncProcDec:
    """Build a model via ``__new__`` with just enough attributes for unit tests."""
    model = AnemoiTransportSpatialDownscalerModelEncProcDec.__new__(
        AnemoiTransportSpatialDownscalerModelEncProcDec,
    )
    model.data_indices = data_indices or _make_downscaler_indices()
    model.dataset_names = list(model.data_indices.keys())
    model.n_step_input = n_step_input
    model.n_step_output = n_step_output
    # num_input_channels/num_output_channels mirror BaseGraphModel behaviour.
    model.num_input_channels = {name: len(indices.model.input) for name, indices in model.data_indices.items()}
    model.num_output_channels = {name: len(indices.model.output) for name, indices in model.data_indices.items()}
    model.node_attributes = _StaticNodeAttributes(
        attr_ndims or {"in_lres": 2, "in_hres": 2, "out_hres": 3},
        grid=grid,
    )
    _wire_fused_encoder_routing(model, sources=["in_lres", "in_hres"])
    return model


def test_targets_on_anchor_falls_back_to_identity_for_models_pickled_without_target2anchor() -> None:
    """Checkpoints of transport models trained before ``target2anchor`` existed must still run."""
    model = AnemoiTransportModelEncProcDec.__new__(AnemoiTransportModelEncProcDec)
    torch.nn.Module.__init__(model)
    model.target_datasets = ["data"]

    assert model._targets_on_anchor("data") == ["data"]
    assert model._anchor_of_target("data") == "data"


# ── residual references ─────────────────────────────────────────────────────


def test_validate_residual_reference_allows_multiple_targets() -> None:
    """Several targets may share one reference; each gets its own decoder."""
    model = _make_bare_model()
    # Both target datasets must have prognostic variable sets matching the reference.
    model.data_indices = {
        **_make_downscaler_indices(),
        "out_hres_2": _make_index_collection({"t2m": 0, "u10": 1}),
    }
    model.target_datasets = ["out_hres", "out_hres_2"]
    model.target_anchors = {"out_hres": "in_lres", "out_hres_2": "in_lres"}

    model._validate_residual_reference()

    assert model.residual_reference == {"out_hres": "in_lres", "out_hres_2": "in_lres"}


# ── dimension arithmetic ─────────────────────────────────────────────────────


def test_calculate_input_dim_sums_all_fused_inputs_plus_attached_target_and_node_attrs() -> None:
    """input_dim = fused input vars over history + noised attached target + anchor node attrs."""
    # in_lres has 2 input vars, in_hres has 1, out_hres has 2 output vars.
    # attr_ndims for the anchor in_lres is 2.
    model = _make_bare_model(n_step_input=2, n_step_output=1)
    # sum of input vars across fused inputs (over history): 2*(2+1) = 6
    # noised target: 1 * 2 = 2
    # anchor node attrs: 2
    assert model._calculate_input_dim("in_lres") == 6 + 2 + 2


# ── input assembly ────────────────────────────────────────────────────────────


def test_assemble_input_concatenates_fused_inputs_attached_target_and_node_attrs() -> None:
    """Encoder input tensor is [in_lres_vars | in_hres_vars | y_noised_vars | anchor node_attrs]."""
    model = _make_bare_model(n_step_input=1, n_step_output=1)
    batch = 2
    ensemble = 1
    grid = 4
    # per-dataset tensors: shape (batch, time, ensemble, grid, vars)
    x_in_lres = torch.full((batch, 1, ensemble, grid, 2), 1.0)
    x_in_hres = torch.full((batch, 1, ensemble, grid, 1), 2.0)
    # The noised target is not an encoder source: it rides along on its reference's anchor.
    y_noised = torch.full((batch, 1, ensemble, grid, 2), 7.0)

    x_dict = {"in_lres": x_in_lres, "in_hres": x_in_hres}
    conditioned_target = {"out_hres": y_noised}

    latent, _skip, _sharding = model._assemble_input(
        x=x_dict,
        y_noised=conditioned_target,
        bse=batch * ensemble,
        grid_shard_sizes=None,
        model_comm_group=None,
        dataset_name="in_lres",
    )

    # Expected feature dim = 2 (in_lres) + 1 (in_hres) + 2 (y_noised) + 2 (attrs) = 7
    assert latent.shape == (batch * ensemble * grid, 7)
    assert latent.shape[-1] == model._calculate_input_dim("in_lres")
    # Feature slices should match the sources.
    torch.testing.assert_close(latent[:, 0:2], torch.full((batch * ensemble * grid, 2), 1.0))
    torch.testing.assert_close(latent[:, 2:3], torch.full((batch * ensemble * grid, 1), 2.0))
    torch.testing.assert_close(latent[:, 3:5], torch.full((batch * ensemble * grid, 2), 7.0))
    torch.testing.assert_close(latent[:, 5:7], torch.zeros(batch * ensemble * grid, 2))


def test_assemble_input_uses_input_dataset_order_from_encoder_routing() -> None:
    """Assembly must follow the encoder's ``source_datasets`` order so the layout is stable."""
    model = _make_bare_model()
    # in_hres first makes it the anchor; the target follows its reference onto it.
    _wire_fused_encoder_routing(model, sources=["in_hres", "in_lres"])

    batch = 1
    ensemble = 1
    grid = 4
    x_in_lres = torch.full((batch, 1, ensemble, grid, 2), 1.0)
    x_in_hres = torch.full((batch, 1, ensemble, grid, 1), 2.0)
    y_noised = torch.zeros(batch, 1, ensemble, grid, 2)

    latent, _skip, _sharding = model._assemble_input(
        x={"in_lres": x_in_lres, "in_hres": x_in_hres},
        y_noised={"out_hres": y_noised},
        bse=batch * ensemble,
        grid_shard_sizes=None,
        model_comm_group=None,
        dataset_name="in_hres",
    )
    # in_hres comes first now (1 var of value 2.0), then in_lres (2 vars of value 1.0).
    torch.testing.assert_close(latent[:, 0:1], torch.full((batch * ensemble * grid, 1), 2.0))
    torch.testing.assert_close(latent[:, 1:3], torch.full((batch * ensemble * grid, 2), 1.0))


# ── sampling hooks ────────────────────────────────────────────────────────────


def test_before_sampling_applies_spatial_preprocessor_and_pre_processors() -> None:
    """``_before_sampling`` runs the spatial projector first (on raw values) then the state normalizer."""
    model = _make_bare_model()
    projector = _IdentitySpatialProjector()
    pre_lres = _AdditiveProcessor(offset=10.0)
    pre_hres = _AdditiveProcessor(offset=20.0)
    pre_out = _AdditiveProcessor(offset=30.0)
    # Post-processors are stored as inverse-style ``Processors`` — calling them
    # applies the inverse.  We model that by giving them a negative offset so
    # calling ``post_lres(x)`` yields ``x - 10`` (i.e. undoes ``pre_lres``).
    post_lres = _AdditiveProcessor(offset=-10.0)
    post_out = _AdditiveProcessor(offset=-30.0)

    batch_lres = torch.full((1, 1, 4, 2), 1.0)
    batch_hres = torch.full((1, 1, 4, 1), 2.0)
    batch_out = torch.zeros(1, 1, 4, 2)
    batch = {"in_lres": batch_lres, "in_hres": batch_hres, "out_hres": batch_out}

    result, _ = model._before_sampling(
        batch,
        pre_processors={"in_lres": pre_lres, "in_hres": pre_hres, "out_hres": pre_out},
        n_step_input=1,
        model_comm_group=None,
        spatial_pre_processors={"in_lres": projector},
        post_processors={"in_lres": post_lres, "out_hres": post_out},
    )

    xs, x_ref_by_target, ref_name_to_index_by_target = result
    # The projector must have been called with the raw lres batch first,
    # before any normalization.  In single-process runs the source-grid shard
    # sizes passed to the projector are ``None``.
    assert len(projector.calls) == 1
    torch.testing.assert_close(projector.calls[0]["x"], batch_lres.unsqueeze(2))  # add ensemble dim
    # After spatial projection, ``pre_processors["in_lres"]`` is applied (adds 10).
    torch.testing.assert_close(xs["in_lres"], batch_lres.unsqueeze(2) + 10.0)
    torch.testing.assert_close(xs["in_hres"], batch_hres.unsqueeze(2) + 20.0)
    # The denormalized projected reference is returned as a per-target dict:
    # normalized (batch + 10) then denormalized (subtract 10) = batch.
    assert set(x_ref_by_target) == {"out_hres"}
    torch.testing.assert_close(x_ref_by_target["out_hres"], batch_lres.unsqueeze(2))
    # The reference dataset's model-input name_to_index is threaded through as a per-target dict.
    assert ref_name_to_index_by_target == {"out_hres": model.data_indices["in_lres"].model.input.name_to_index}
    assert post_lres.calls[0]["kwargs"]["skip_imputation"] is True


def test_reference_on_target_grid_reads_the_model_input_layout_without_imputation() -> None:
    """Training and inference both build the reference from the normalized model input.

    ``aux`` is diagnostic in the reference, so its model-input layout differs from its full data layout.
    """
    model = _make_bare_model()
    model.data_indices = {
        **model.data_indices,
        "in_lres": _make_index_collection({"t2m": 0, "aux": 1, "u10": 2}, diagnostic=["aux"]),
    }
    post_lres = _AdditiveProcessor(offset=-1.0)
    x_lres = torch.full((1, 1, 1, 4, 2), 3.0)

    references, columns = model.reference_on_target_grid({"in_lres": x_lres}, {"in_lres": post_lres})

    torch.testing.assert_close(references["out_hres"], x_lres - 1.0)
    assert columns == {"out_hres": {"t2m": 0, "u10": 1}}
    (call,) = post_lres.calls
    assert call["kwargs"]["skip_imputation"] is True
    assert call["data_index"] == model.data_indices["in_lres"].data.input.full.tolist()


def test_before_sampling_raises_when_an_input_dataset_is_missing() -> None:
    model = _make_bare_model()

    with pytest.raises(ValueError, match="in_hres"):
        model._before_sampling(
            {"in_lres": torch.zeros(1, 1, 4, 2)},
            pre_processors={"in_lres": _AdditiveProcessor(offset=0.0)},
            n_step_input=1,
            model_comm_group=None,
            spatial_pre_processors={},
            post_processors={"in_lres": _AdditiveProcessor(offset=0.0)},
        )


def test_build_sampling_source_rejects_a_reference_that_is_not_on_the_target_grid() -> None:
    """Catches a projector whose output grid disagrees with the target node set."""
    model = _make_bare_model(grid=4)
    model.transport_source = TransportSourceBuilder()

    x = {"in_lres": torch.zeros(1, 1, 1, 7, 2), "in_hres": torch.zeros(1, 1, 1, 7, 1)}

    with pytest.raises(AssertionError, match="target grid"):
        model.build_sampling_source(x)


# ── mixed-target (prognostic + diagnostic) fixtures ─────────────────────────


def _make_mixed_downscaler_indices() -> dict[str, IndexCollection]:
    """Target with two prognostic and one diagnostic variable.
    The lres dataset has the two prognostic variables also declared as
    prognostic, matching how spatial downscaling typically pairs lres/hres
    channels.  The diagnostic variable (``precip``) has no lres counterpart —
    it is predicted directly as a state.
    """
    in_lres = _make_index_collection({"t2m": 0, "u10": 1})
    in_hres = _make_index_collection({"z": 0}, forcing=["z"])
    # ``precip`` is diagnostic so it appears in ``model.output.diagnostic``;
    # ``t2m`` and ``u10`` are prognostic (present in input and output).
    out_hres = _make_index_collection(
        {"t2m": 0, "u10": 1, "precip": 2},
        diagnostic=["precip"],
    )
    return {"in_lres": in_lres, "in_hres": in_hres, "out_hres": out_hres}


# The reference stores the target prognostics at other positions, so columns must be matched by name.
_REORDERED_REFERENCE_COLUMNS = {"u10": 0, "foo": 1, "t2m": 2}
_REORDERED_REFERENCE = torch.tensor([[[[[4.0, 999.0, 3.0]]]]])  # u10=4, foo (unused), t2m=3


# ── compute_residual / add_residual_to_state ────────────────────────────────


def test_compute_residual_uses_residual_pre_for_prognostic_and_state_pre_for_diagnostic() -> None:
    """Prognostic channels are normalized as residuals against the reference, diagnostic channels as states."""
    model = _make_bare_model(data_indices=_make_mixed_downscaler_indices())
    indices = model.data_indices["out_hres"]

    input_post = _AdditiveProcessor(offset=0.0)
    state_pre = _AdditiveProcessor(offset=100.0)
    residual_pre = _AdditiveProcessor(offset=10.0)

    y = torch.tensor([[[[[10.0, 20.0, 30.0]]]]])  # t2m, u10, precip

    out = model.compute_residual(
        y={"out_hres": y},
        x_reference_denorm={"out_hres": _REORDERED_REFERENCE},
        pre_processors_state={"out_hres": state_pre},
        pre_processors_residual={"out_hres": residual_pre},
        reference_variable_name_to_column_index_by_target={"out_hres": _REORDERED_REFERENCE_COLUMNS},
        input_post_processor={"out_hres": input_post},
        skip_imputation=True,
    )

    # t2m: (10 - 3) + 10 = 17, u10: (20 - 4) + 10 = 26, precip: 30 + 100 = 130
    torch.testing.assert_close(out["out_hres"], torch.tensor([[[[[17.0, 26.0, 130.0]]]]]))
    assert residual_pre.calls[0]["data_index"] == indices.data.output.prognostic.tolist()
    assert state_pre.calls[0]["data_index"] == indices.data.output.diagnostic.tolist()


def test_add_residual_to_state_denormalizes_prognostic_with_residual_and_diagnostic_with_state() -> None:
    """Inverse of ``compute_residual``: residual post + reference for prognostics, state post for diagnostics."""
    model = _make_bare_model(data_indices=_make_mixed_downscaler_indices())
    indices = model.data_indices["out_hres"]

    residual_post = _AdditiveProcessor(offset=-10.0)
    state_post = _AdditiveProcessor(offset=-100.0)

    state = model.add_residual_to_state(
        x_reference_denorm={"out_hres": _REORDERED_REFERENCE},
        residual={"out_hres": torch.tensor([[[[[17.0, 26.0, 130.0]]]]])},
        post_processors_state={"out_hres": state_post},
        post_processors_residual={"out_hres": residual_post},
        reference_variable_name_to_column_index_by_target={"out_hres": _REORDERED_REFERENCE_COLUMNS},
        output_pre_processor=None,
        skip_imputation=True,
    )

    # t2m: (17 - 10) + 3 = 10, u10: (26 - 10) + 4 = 20, precip: 130 - 100 = 30
    torch.testing.assert_close(state["out_hres"], torch.tensor([[[[[10.0, 20.0, 30.0]]]]]))
    assert any(call["data_index"] == indices.data.output.diagnostic.tolist() for call in state_post.calls)


def test_compute_residual_raises_when_target_prognostic_missing_from_reference() -> None:
    model = _make_bare_model(data_indices=_make_mixed_downscaler_indices())

    with pytest.raises(KeyError, match=r"t2m"):
        model.compute_residual(
            y={"out_hres": torch.zeros(1, 1, 1, 1, 3)},
            x_reference_denorm={"out_hres": torch.zeros(1, 1, 1, 1, 1)},
            pre_processors_state={"out_hres": _AdditiveProcessor(offset=0.0)},
            pre_processors_residual={"out_hres": _AdditiveProcessor(offset=0.0)},
            reference_variable_name_to_column_index_by_target={"out_hres": {"u10": 0}},
            input_post_processor={"out_hres": _AdditiveProcessor(offset=0.0)},
            skip_imputation=True,
        )


# ── _after_sampling with mixed target ───────────────────────────────────────


def test_after_sampling_mixed_target_uses_state_post_for_diagnostic_and_residual_post_for_prognostic() -> None:
    """With a diagnostic variable in the target, ``_after_sampling`` splits per-channel."""
    model = _make_bare_model(data_indices=_make_mixed_downscaler_indices())

    residual_post = _AdditiveProcessor(offset=-5.0)
    state_post = _AdditiveProcessor(offset=-50.0)

    # (batch, time, ensemble, grid, vars=3)
    residual_pred = torch.tensor([[[[[7.0, 12.0, 55.0]]]]])  # t2m, u10, precip
    x_lres_denorm = torch.tensor([[[[[3.0, 4.0]]]]])

    result = model._after_sampling(
        {"out_hres": residual_pred},
        post_processors={"out_hres": state_post},
        before_sampling_data=(
            {"in_lres": None, "in_hres": None, "out_hres": None},
            {"out_hres": x_lres_denorm},
            {"out_hres": model.data_indices["in_lres"].name_to_index},
        ),
        model_comm_group=None,
        grid_shard_sizes=None,
        gather_out=False,
        post_processors_residual={"out_hres": residual_post},
    )

    # Prognostic: (residual_pred - 5) + x_lres = (7-5)+3 = 5, (12-5)+4 = 11
    # Diagnostic: residual_pred - 50 = 55 - 50 = 5
    expected = torch.tensor([[[[[5.0, 11.0, 5.0]]]]])
    torch.testing.assert_close(result["out_hres"], expected)


# ── target routing (transport base) ───────────────────────────────────────────


def _make_target_routing_model(
    *,
    target_anchors: dict[str, str],
    sources: tuple[str, ...] = ("in_lres", "in_hres"),
    target_datasets: tuple[str, ...] = ("out_hres",),
    num_nodes: dict[str, int] | None = None,
    coordinates: dict[str, torch.Tensor] | None = None,
) -> AnemoiTransportModelEncProcDec:
    """Routing state after ``_build_encoder_routing`` / ``_build_decoder_routing`` for one fused encoder."""
    model = AnemoiTransportModelEncProcDec.__new__(AnemoiTransportModelEncProcDec)
    model.target_anchors = target_anchors
    model.dataset2encoder = {name: "enc0" for name in sources}
    model.dataset2anchor = {name: sources[0] for name in sources}
    model.target_datasets = list(target_datasets)
    num_nodes = num_nodes or {name: 4 for name in (*sources, *target_datasets)}
    model.node_attributes = SimpleNamespace(num_nodes=num_nodes)
    graph = HeteroData()
    for name, count in num_nodes.items():
        graph[name].x = (coordinates or {}).get(name, torch.zeros(count, 2))
    model._graph_data = graph
    return model


def test_target_routing_decodes_an_attached_target_from_its_reference_anchor() -> None:
    model = _make_target_routing_model(target_anchors={"out_hres": "in_lres"})

    model._build_target_routing()

    assert model.target2anchor == {"out_hres": "in_lres"}


def test_target_routing_follows_the_reference_onto_whichever_source_anchors_the_encoder() -> None:
    """The reference need not be the anchor itself, only fused into its encoder."""
    model = _make_target_routing_model(target_anchors={"out_hres": "in_lres"}, sources=("in_hres", "in_lres"))

    model._build_target_routing()

    assert model.target2anchor == {"out_hres": "in_hres"}


def test_target_routing_defaults_to_decoding_each_target_from_its_own_anchor() -> None:
    """State and tendency models pass no target anchors; nothing may change for them."""
    model = _make_target_routing_model(target_anchors={}, sources=("data",), target_datasets=("data",))

    model._build_target_routing()

    assert model.target2anchor == {"data": "data"}


def test_target_routing_rejects_a_reference_that_is_not_encoded() -> None:
    """A reference the encoder never sees would leave the noised target without a node set."""
    model = _make_target_routing_model(target_anchors={"out_hres": "in_lres"}, sources=("in_hres",))

    with pytest.raises(ValueError, match="not a source dataset of any encoder"):
        model._build_target_routing()


def test_target_routing_rejects_targets_the_decoders_do_not_declare() -> None:
    model = _make_target_routing_model(target_anchors={"out_hres": "in_lres", "other": "in_lres"})

    with pytest.raises(ValueError, match=r"\['other'\].*no decoder"):
        model._build_target_routing()


def test_target_routing_rejects_an_attached_target_that_is_also_an_encoder_source() -> None:
    """The target must not be listed in ``source_datasets`` once it is attached to its reference."""
    model = _make_target_routing_model(
        target_anchors={"out_hres": "in_lres"},
        sources=("in_lres", "in_hres", "out_hres"),
    )

    with pytest.raises(ValueError, match="must not also be a source dataset"):
        model._build_target_routing()


@pytest.mark.parametrize(
    ("num_nodes", "coordinates", "match"),
    [
        ({"in_lres": 4, "in_hres": 4, "out_hres": 5}, None, "4 nodes.*5"),
        (
            None,
            {"in_lres": torch.arange(8.0).reshape(4, 2), "out_hres": torch.arange(8.0).reshape(4, 2).flip(0)},
            "different coordinates",
        ),
    ],
    ids=["node_count", "node_order"],
)
def test_target_routing_rejects_an_anchor_on_a_different_grid_than_its_target(
    num_nodes: dict[str, int] | None,
    coordinates: dict[str, torch.Tensor] | None,
    match: str,
) -> None:
    """The decoder writes the target onto the anchor's nodes, so count and order must match."""
    model = _make_target_routing_model(
        target_anchors={"out_hres": "in_lres"},
        num_nodes=num_nodes,
        coordinates=coordinates,
    )

    with pytest.raises(ValueError, match=match):
        model._build_target_routing()


# ── residual references: reference/target prognostic consistency ───────────


def _make_reference_model(data_indices: dict[str, IndexCollection]) -> AnemoiTransportSpatialDownscalerModelEncProcDec:
    model = AnemoiTransportSpatialDownscalerModelEncProcDec.__new__(
        AnemoiTransportSpatialDownscalerModelEncProcDec,
    )
    model.data_indices = data_indices
    model.target_datasets = ["out_hres"]
    model.target_anchors = {"out_hres": "in_lres"}
    return model


def test_validate_residual_reference_accepts_a_diagnostic_without_a_reference_counterpart() -> None:
    model = _make_reference_model(
        {
            "in_lres": _make_index_collection({"t2m": 0}),
            "out_hres": _make_index_collection({"t2m": 0, "precip": 1}, diagnostic=["precip"]),
        },
    )

    model._validate_residual_reference()


def test_validate_residual_reference_rejects_target_prognostic_absent_from_reference() -> None:
    """A target prognostic that does not exist at all in the reference is a config error."""
    model = _make_reference_model(
        {
            "in_lres": _make_index_collection({"u10": 0}),
            "out_hres": _make_index_collection({"t2m": 0}),
        },
    )
    with pytest.raises(ValueError, match=r"only in target: \['t2m'\]"):
        model._validate_residual_reference()


def test_validate_residual_reference_rejects_reference_prognostic_absent_from_target() -> None:
    """A prognostic in the reference that the target does not predict prognostically is also a config error."""
    model = _make_reference_model(
        {
            "in_lres": _make_index_collection({"t2m": 0, "u10": 1}),  # both prognostic
            "out_hres": _make_index_collection({"t2m": 0}),  # only t2m
        },
    )
    with pytest.raises(ValueError, match=r"only in reference: \['u10'\]"):
        model._validate_residual_reference()


# ── end-to-end construction ─────────────────────────────────────────────────


def _make_downscaler_graph(grid: int = 4, hidden: int = 3, encoded: tuple[str, ...] = ("in_lres",)) -> HeteroData:
    """Graph with encoder edges from each ``encoded`` anchor and decoder edges to ``out_hres``.

    ``in_lres`` lives on the target grid because the projector puts it there.
    """
    graph = HeteroData()
    for name in ("out_hres", "in_lres", "in_hres"):
        graph[name].x = torch.zeros(grid, 2)
        graph[name].num_nodes = grid
    graph["hidden"].x = torch.zeros(hidden, 2)
    graph["hidden"].num_nodes = hidden

    def _dense_edges(num_src: int, num_dst: int) -> torch.Tensor:
        src = torch.arange(num_src).repeat_interleave(num_dst)
        dst = torch.arange(num_dst).repeat(num_src)
        return torch.stack([src, dst])

    relations = {(name, "to", "hidden"): (grid, hidden) for name in encoded}
    relations[("hidden", "to", "out_hres")] = (hidden, grid)
    relations[("hidden", "to", "hidden")] = (hidden, hidden)
    for relation, (num_src, num_dst) in relations.items():
        edge_index = _dense_edges(num_src, num_dst)
        graph[relation].edge_index = edge_index
        graph[relation].edge_length = torch.zeros(edge_index.shape[1], 1)
    return graph


def _gnn_mapper(target: str, num_channels: int = 8) -> dict[str, Any]:
    return {
        "_target_": target,
        "num_channels": num_channels,
        "num_chunks": 1,
        "mlp_extra_layers": 0,
        "mlp_hidden_ratio": 1,
        "cpu_offload": False,
        "layer_kernels": {},
        "sub_graph_edge_attributes": ["edge_length"],
    }


_FORWARD_MAPPER = "anemoi.models.layers.mapper.GNNForwardMapper"


def _make_downscaler_config(num_channels: int = 8) -> DictConfig:
    return DictConfig(
        {
            "num_channels": num_channels,
            "node_trainable_parameters": {name: 0 for name in ("out_hres", "in_lres", "in_hres", "hidden")},
            "model": {
                "_target_": (
                    "anemoi.models.models.transport_encoder_processor_decoder."
                    "AnemoiTransportSpatialDownscalerModelEncProcDec"
                ),
                "hidden_nodes_name": "hidden",
                "latent_skip": True,
                "transport": {
                    "objective": "edm_diffusion",
                    "noise_channels": 4,
                    "noise_cond_dim": 2,
                    "noise_embedder": {
                        "_target_": "anemoi.models.layers.diffusion.SinusoidalEmbeddings",
                        "num_channels": 4,
                        "max_period": 1000,
                    },
                },
            },
            "encoders": {
                "enc0": {
                    "source_datasets": ["in_lres", "in_hres"],
                    "dataset_fusing_strategy": "concatenate_inputs_along_variable_dim",
                    "mapper": _gnn_mapper(_FORWARD_MAPPER, num_channels),
                },
            },
            "latent_aggregator": {"_target_": "anemoi.models.layers.aggregator.SumAggregator"},
            "processor": {
                "_target_": "anemoi.models.layers.processor.PointWiseMLPProcessor",
                "num_channels": num_channels,
                "num_layers": 1,
                "num_chunks": 1,
                "mlp_hidden_ratio": 1,
                "cpu_offload": False,
                "gradient_checkpointing": False,
                "layer_kernels": {},
                "sub_graph_edge_attributes": ["edge_length"],
            },
            "decoders": {
                "dec0": {
                    "target_datasets": ["out_hres"],
                    "target_node_features": ["encoded_data"],
                    "mapper": _gnn_mapper("anemoi.models.layers.mapper.GNNBackwardMapper", num_channels),
                },
            },
            "residual": {
                "datasets": {
                    name: {"_target_": "anemoi.models.layers.residual.SkipConnection"}
                    for name in ("out_hres", "in_lres", "in_hres")
                },
            },
            "bounding": {"datasets": {name: [] for name in ("out_hres", "in_lres", "in_hres")}},
            "residual_reference": {"out_hres": "in_lres"},
        },
    )


def _build_real_downscaler(
    *,
    config: DictConfig | None = None,
    graph: HeteroData | None = None,
) -> AnemoiTransportSpatialDownscalerModelEncProcDec:
    return AnemoiTransportSpatialDownscalerModelEncProcDec(
        model_config=config if config is not None else _make_downscaler_config(),
        data_indices=_make_downscaler_indices(),
        statistics={name: None for name in ("out_hres", "in_lres", "in_hres")},
        n_step_input=1,
        n_step_output=1,
        graph_data=graph if graph is not None else _make_downscaler_graph(),
    )


def test_real_construction_builds_one_encoder_decoder_pair_anchored_at_the_reference() -> None:
    """Every other test wires routing by hand, so this is the only check that
    ``__init__`` (routing, dimension arithmetic, network build) agrees with itself.
    """
    model = _build_real_downscaler()

    # in_hres shares the anchor's encoder and node set but owns no mapper.
    assert model.input_datasets == ["in_lres"]
    assert model.target_datasets == ["out_hres"]
    assert model.residual_reference == {"out_hres": "in_lres"}
    assert model.target2anchor == {"out_hres": "in_lres"}
    assert set(model.encoder.keys()) == {"enc0"}
    assert set(model.decoder.keys()) == {"dec0"}
    assert set(model.encoder_graph_provider.keys()) == {"in_lres"}
    assert set(model.decoder_graph_provider.keys()) == {"out_hres"}

    # in_lres (2 vars) + in_hres (1 var) history, noised target (2 vars), node attrs.
    assert model.input_dim["in_lres"] == 3 + 2 + model.node_attributes.attr_ndims["in_lres"]
    assert model.target_dim["out_hres"] == model.input_dim["in_lres"]


def test_real_construction_requires_a_residual_reference() -> None:
    config = _make_downscaler_config()
    del config.residual_reference

    with pytest.raises(ValueError, match="residual_reference"):
        _build_real_downscaler(config=config)


def test_real_construction_without_fusion_encodes_each_input_on_its_own_node_set() -> None:
    """``not_supported``: separate encoders, combined in latent space by the aggregator."""
    config = _make_downscaler_config()
    config.encoders = {
        name: {
            "source_datasets": [dataset_name],
            "dataset_fusing_strategy": "not_supported",
            "mapper": _gnn_mapper(_FORWARD_MAPPER),
        }
        for name, dataset_name in (("lres", "in_lres"), ("hres", "in_hres"))
    }
    model = _build_real_downscaler(config=config, graph=_make_downscaler_graph(encoded=("in_lres", "in_hres")))

    assert model.input_datasets == ["in_lres", "in_hres"]
    assert model.target2anchor == {"out_hres": "in_lres"}
    # in_lres history (2) + noised target (2); in_hres history (1) only.
    assert model.input_dim["in_lres"] == 2 + 2 + model.node_attributes.attr_ndims["in_lres"]
    assert model.input_dim["in_hres"] == 1 + model.node_attributes.attr_ndims["in_hres"]

    batch, grid = 1, 4
    out = _run_real_predict_step(
        model,
        {"in_lres": torch.zeros(batch, 1, grid, 2), "in_hres": torch.zeros(batch, 1, grid, 1)},
    )

    assert set(out) == {"out_hres"}
    assert out["out_hres"].shape == (batch, 1, 1, grid, 2)


def _run_real_predict_step(
    model: AnemoiTransportSpatialDownscalerModelEncProcDec,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    identity = {name: _AdditiveProcessor(offset=0.0) for name in ("in_lres", "in_hres", "out_hres")}
    return model.predict_step(
        batch,
        pre_processors=identity,
        post_processors=identity,
        n_step_input=1,
        post_processors_residual={"out_hres": _AdditiveProcessor(offset=0.0)},
        schedule_params={"schedule_type": "karras", "num_steps": 2, "sigma_max": 1.0, "sigma_min": 0.1, "rho": 7.0},
        sampler_params={"sampler": "heun"},
    )


@pytest.mark.parametrize("with_target_entry", [False, True], ids=["inputs_only", "superfluous_target"])
def test_real_construction_predict_step_needs_no_batch_entry_for_the_target(with_target_entry: bool) -> None:
    """Inference must survive ``x`` and the sampled target having disjoint keys.

    The sampler sizes the target from the graph, so nothing may index the batch
    or ``x`` by target name. A placeholder target entry (older runners) is ignored.
    """
    model = _build_real_downscaler()
    batch_size, grid = 1, 4

    # (batch, time, grid, vars) — predict_step adds the ensemble dimension.
    batch = {
        "in_lres": torch.zeros(batch_size, 1, grid, 2),
        "in_hres": torch.zeros(batch_size, 1, grid, 1),
    }
    if with_target_entry:
        batch["out_hres"] = torch.zeros(batch_size, 1, grid, 2)

    out = _run_real_predict_step(model, batch)

    assert set(out) == {"out_hres"}
    assert out["out_hres"].shape == (batch_size, 1, 1, grid, 2)


def test_real_construction_fill_metadata_records_input_and_output_roles() -> None:
    """With the roles recorded, a runner needs no dataset configuration at all."""
    model = _build_real_downscaler()
    md_dict = {"metadata_inference": {name: {} for name in ("in_lres", "in_hres", "out_hres")}}

    model.fill_metadata(md_dict)

    roles = {name: md_dict["metadata_inference"][name]["role"] for name in ("in_lres", "in_hres", "out_hres")}
    assert roles == {"in_lres": "input", "in_hres": "input", "out_hres": "output"}


@pytest.mark.parametrize("fused", [True, False], ids=["fused", "not_fused"])
def test_real_construction_fill_metadata_records_input_shapes_per_input_dataset(fused: bool) -> None:
    """Each input records its own width, as a forecaster would, not the fused encoder width."""
    config = _make_downscaler_config()
    graph = None
    if not fused:
        config.encoders = {
            name: {
                "source_datasets": [dataset_name],
                "dataset_fusing_strategy": "not_supported",
                "mapper": _gnn_mapper(_FORWARD_MAPPER),
            }
            for name, dataset_name in (("lres", "in_lres"), ("hres", "in_hres"))
        }
        graph = _make_downscaler_graph(encoded=("in_lres", "in_hres"))
    model = _build_real_downscaler(config=config, graph=graph)
    md_dict = {"metadata_inference": {name: {} for name in ("in_lres", "in_hres", "out_hres")}}

    model.fill_metadata(md_dict)

    attr_ndims = model.node_attributes.attr_ndims
    for name, num_variables in (("in_lres", 2), ("in_hres", 1)):
        assert md_dict["metadata_inference"][name]["shapes"] == {
            "variables": num_variables + attr_ndims[name],
            "input_timesteps": 1,
            "ensemble": 1,
            "grid": None,
        }
    assert "shapes" not in md_dict["metadata_inference"]["out_hres"]

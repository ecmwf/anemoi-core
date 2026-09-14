# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for AnemoiTransportSpatialDownscalerModelEncProcDec.
The tests use ``__new__`` and wire attributes manually to avoid needing a full
graph and encoder/decoder stack.  They cover the pieces that are specific to
the downscaler: role inference from data_indices, input dimension arithmetic,
input assembly on the target grid, and the spatial pre-processor hooks in the
sampling flow.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportModelEncProcDec
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportSpatialDownscalerModelEncProcDec

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
    prognostic in the reference dataset for ``_resolve_roles`` to accept the
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

    def __call__(self, dataset_name: str, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size * self.grid, self.attr_ndims[dataset_name])


class _AdditiveProcessor:
    """Trivial processor stub: adds ``offset`` on every call.
    ``Processors`` objects in the codebase are always constructed as either
    forward (pre) or inverse (post); the ``inverse`` kwarg is not toggled at the
    call site.  We model the pre/post distinction with the sign of ``offset``
    rather than by handling ``inverse=True``.
    """

    def __init__(self, offset: float) -> None:
        self.offset = offset
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        x: torch.Tensor,
        in_place: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert in_place is False, "Downscaler must call processors with in_place=False."
        self.calls.append({"shape": tuple(x.shape), "kwargs": kwargs})
        return x + self.offset


class _IdentitySpatialProjector:
    """Spatial pre-processor stub that just passes input through.

    Records every call (tensor + kwargs) so tests can assert both the *order*
    of operations in ``_before_sampling`` and the arguments passed to the
    projector (in particular ``grid_shard_sizes``, which must be the
    source-grid shard sizes so behaviour matches the training path).
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.calls.append({"x": x, "kwargs": kwargs})
        return x


def _wire_fused_encoder_routing(
    model: AnemoiTransportSpatialDownscalerModelEncProcDec,
    *,
    anchor: str,
    fused: list[str],
) -> None:
    """Attach the routing state ``BaseGraphModel._build_encoder_routing`` would produce.

    Config keys (``enc0``/``dec0``) deliberately differ from the dataset names,
    as in the graphtransformer_multi_* configs.
    """
    model.encoder2datasets = {"enc0": [anchor, *fused]}
    model.encoder2anchors = {"enc0": [anchor]}
    model.dataset2anchor = {name: anchor for name in (anchor, *fused)}
    model.dataset2encoder = {name: "enc0" for name in (anchor, *fused)}
    model.input_datasets = [anchor]
    model.dataset2decoder = {anchor: "dec0"}
    model.decoder2datasets = {"dec0": [anchor]}
    model.decoders_target_input = {"dec0": SimpleNamespace(dim=0)}
    model.target_datasets = [anchor]


def _make_bare_model(
    *,
    n_step_input: int = 1,
    n_step_output: int = 1,
    attr_ndims: dict[str, int] | None = None,
    grid: int = 4,
) -> AnemoiTransportSpatialDownscalerModelEncProcDec:
    """Build a model via ``__new__`` with just enough attributes for unit tests."""
    model = AnemoiTransportSpatialDownscalerModelEncProcDec.__new__(
        AnemoiTransportSpatialDownscalerModelEncProcDec,
    )
    model.data_indices = _make_downscaler_indices()
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
    model.target_dataset_names = ["out_hres"]
    # Encoder/decoder config names deliberately differ from the dataset name,
    # as in the graphtransformer_multi_* configs.
    _wire_fused_encoder_routing(model, anchor="out_hres", fused=["in_lres", "in_hres"])
    model._roles_by_target = {
        "out_hres": {"reference": "in_lres", "target": "out_hres"},
    }
    return model


def test_encoder_node_set_maps_fused_inputs_to_the_anchor() -> None:
    """Fused inputs are encoded on the anchor's grid, so they share its node set."""
    model = _make_bare_model()

    assert model.encoder_node_set("in_lres") == "out_hres"
    assert model.encoder_node_set("in_hres") == "out_hres"


def test_encoder_node_set_leaves_the_anchor_unchanged() -> None:
    model = _make_bare_model()

    assert model.encoder_node_set("out_hres") == "out_hres"


def test_fused_input_dataset_names_excludes_the_anchor() -> None:
    """The anchor contributes the noised target, not an input history."""
    model = _make_bare_model()

    assert model._fused_input_dataset_names("out_hres") == ["in_lres", "in_hres"]


def test_forward_transport_network_resolves_encoder_and_decoder_by_routing_name() -> None:
    """Encoder/decoder keys are user-defined config names, not dataset names."""
    model = _make_bare_model()
    model._graph_name_hidden = "hidden"
    model.node_attributes.attr_ndims["hidden"] = 1
    model.latent_skip = False

    calls: list[str] = []

    def _encoder(_pair: Any, **_kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append("encoder")
        return torch.zeros(4, 1), torch.zeros(4, 1)

    def _decoder(_pair: Any, **_kwargs: Any) -> torch.Tensor:
        calls.append("decoder")
        return torch.zeros(4, 1)

    model.encoder = {"enc0": _encoder}
    model.decoder = {"dec0": _decoder}

    edges = (None, None, None)
    provider = SimpleNamespace(get_edges=lambda **_kwargs: edges)
    model.encoder_graph_provider = {"out_hres": provider}
    model.decoder_graph_provider = {"out_hres": provider}
    model.processor_graph_provider = provider
    model.processor = lambda x, **_kwargs: x
    model.latent_aggregator = lambda *_args, **_kwargs: torch.zeros(4, 1)

    model._resolve_in_out_sharded = lambda **_kwargs: {"out_hres": False}
    model._assert_valid_sharding = lambda *_args, **_kwargs: None
    model._build_conditioning_kwargs = lambda *_args, **_kwargs: (
        {"out_hres": {}},
        {},
        {"out_hres": {}},
    )
    model._assemble_input = lambda *_args, **_kwargs: (torch.zeros(4, 1), None, None)
    model._assemble_output = lambda x, *_args, **_kwargs: x

    target = torch.zeros(1, 1, 1, 4, 1)
    out = model._forward_transport_network(
        x={"out_hres": target},
        conditioned_target={"out_hres": target},
        condition={"out_hres": torch.zeros(1)},
    )

    assert calls == ["encoder", "decoder"]
    assert "out_hres" in out


def test_forward_transport_network_feeds_fused_features_of_declared_width_to_the_encoder() -> None:
    """Exercise the forward pass with the *real* input/output assembly.

    Every other test stubs ``_assemble_input``/``_assemble_output`` out, which
    leaves the contract between ``_calculate_input_dim`` (what the encoder is
    built for) and ``_assemble_input`` (what the encoder is handed) untested.
    Only the neural modules are stubbed here; they assert on the shapes they
    receive.
    """
    batch, ensemble, grid, num_channels = 2, 1, 4, 5
    model = _make_bare_model(n_step_input=1, n_step_output=1, grid=grid)
    model._graph_name_hidden = "hidden"
    model.node_attributes.attr_ndims["hidden"] = 1
    model.latent_skip = True
    model._calculate_shapes_and_indices(model.data_indices)

    seen: dict[str, tuple[int, ...]] = {}

    def _encoder(pair: tuple[torch.Tensor, torch.Tensor], **_kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        x_src, x_dst = pair
        seen["encoder_src"] = tuple(x_src.shape)
        return x_src, torch.ones(x_dst.shape[0], num_channels)

    def _decoder(pair: tuple[torch.Tensor, torch.Tensor], **_kwargs: Any) -> torch.Tensor:
        x_src, x_dst = pair
        seen["decoder_src"] = tuple(x_src.shape)
        seen["decoder_dst"] = tuple(x_dst.shape)
        return torch.zeros(x_dst.shape[0], model.output_dim["out_hres"])

    model.encoder = {"enc0": _encoder}
    model.decoder = {"dec0": _decoder}

    edges = (None, None, None)
    provider = SimpleNamespace(get_edges=lambda **_kwargs: edges)
    model.encoder_graph_provider = {"out_hres": provider}
    model.decoder_graph_provider = {"out_hres": provider}
    model.processor_graph_provider = provider
    model.processor = lambda x, **_kwargs: x
    # Mirrors SumAggregator for a single source; the real one is an nn.Module and
    # cannot be attached to a model built via __new__.
    model.latent_aggregator = lambda _hidden, latents: next(iter(latents.values()))
    model._build_conditioning_kwargs = lambda *_args, **_kwargs: (
        {"out_hres": {}},
        {},
        {"out_hres": {}},
    )

    x = {
        "in_lres": torch.full((batch, 1, ensemble, grid, 2), 1.0),
        "in_hres": torch.full((batch, 1, ensemble, grid, 1), 2.0),
    }
    conditioned_target = {"out_hres": torch.full((batch, 1, ensemble, grid, 2), 7.0)}

    out = model._forward_transport_network(
        x=x,
        conditioned_target=conditioned_target,
        condition={"out_hres": torch.zeros(batch, 1, ensemble, 1, 1)},
    )

    # The encoder must receive exactly the width the model was sized for.
    assert seen["encoder_src"] == (batch * ensemble * grid, model.input_dim["out_hres"])
    # The decoder's destination features are the encoder-updated data tensor.
    assert seen["decoder_dst"] == (batch * ensemble * grid, model.input_dim["out_hres"])
    assert seen["decoder_src"] == (batch * ensemble * grid, num_channels)
    # Output is reassembled back into (batch, time, ensemble, grid, vars).
    assert out["out_hres"].shape == (batch, 1, ensemble, grid, model.num_output_channels["out_hres"])


# ── role inference ────────────────────────────────────────────────────────────


def test_resolve_roles_identifies_target_and_input_datasets_from_encoder_decoder_roles() -> None:
    """The dataset listed as ``target`` in each triple is a target; reference/conditioning are inputs."""
    model = AnemoiTransportSpatialDownscalerModelEncProcDec.__new__(
        AnemoiTransportSpatialDownscalerModelEncProcDec,
    )
    model.data_indices = _make_downscaler_indices()
    model.dataset_names = list(model.data_indices.keys())
    config = {
        "training": {
            "transport": {
                "encoder_decoder_roles": {
                    "enc_dec_0": {"reference": "in_lres", "target": "out_hres", "conditioning": "in_hres"},
                },
            },
        },
    }
    model._resolve_roles(config)
    assert model.target_dataset_names == ["out_hres"]
    assert model._roles_by_target["out_hres"]["reference"] == "in_lres"


def test_resolve_roles_rejects_duplicate_target() -> None:
    """Two entries with the same ``target`` value must be rejected."""
    model = AnemoiTransportSpatialDownscalerModelEncProcDec.__new__(
        AnemoiTransportSpatialDownscalerModelEncProcDec,
    )
    model.data_indices = _make_downscaler_indices()
    model.dataset_names = list(model.data_indices.keys())
    config = {
        "training": {
            "transport": {
                "encoder_decoder_roles": {
                    "enc_dec_0": {"reference": "in_lres", "target": "out_hres"},
                    "enc_dec_1": {"reference": "in_lres", "target": "out_hres"},  # duplicate!
                },
            },
        },
    }
    with pytest.raises(ValueError, match="more than one entry"):
        model._resolve_roles(config)


def test_resolve_roles_allows_multiple_unique_targets() -> None:
    """Two triples with different targets are both valid — each builds its own enc/dec pair."""
    model = AnemoiTransportSpatialDownscalerModelEncProcDec.__new__(
        AnemoiTransportSpatialDownscalerModelEncProcDec,
    )
    # Both target datasets must have prognostic variable sets matching the reference.
    model.data_indices = {
        **_make_downscaler_indices(),
        "out_hres_2": _make_index_collection({"t2m": 0, "u10": 1}),
    }
    model.dataset_names = list(model.data_indices.keys())
    config = {
        "training": {
            "transport": {
                "encoder_decoder_roles": {
                    "enc_dec_0": {"reference": "in_lres", "target": "out_hres"},
                    "enc_dec_1": {"reference": "in_lres", "target": "out_hres_2"},
                },
            },
        },
    }
    model._resolve_roles(config)
    assert model.target_dataset_names == ["out_hres", "out_hres_2"]


# ── dimension arithmetic ─────────────────────────────────────────────────────


def test_calculate_input_dim_sums_all_input_datasets_plus_noised_target_and_node_attrs() -> None:
    """input_dim = sum(fused input vars over history) + noised_target + anchor node attrs."""
    # in_lres has 2 input vars, in_hres has 1, out_hres has 2 output vars.
    # attr_ndims for out_hres is 3.
    model = _make_bare_model(n_step_input=2, n_step_output=1)
    # sum of input vars across input datasets (over history): 2*(2+1) = 6
    # noised target: 1 * 2 = 2
    # target node attrs: 3
    assert model._calculate_input_dim("out_hres") == 6 + 2 + 3


def test_calculate_input_dim_falls_back_to_the_base_width_for_fused_inputs() -> None:
    """Fused inputs have no encoder of their own, so their width is never used."""
    model = _make_bare_model()

    assert model._calculate_input_dim("in_lres") == AnemoiTransportModelEncProcDec._calculate_input_dim(
        model, "in_lres"
    )


def test_calculate_shapes_and_indices_sizes_the_anchor_from_all_fused_inputs() -> None:
    """The inherited two-pass implementation must size the anchor from the fused sum."""
    model = _make_bare_model(n_step_input=2, n_step_output=1)
    model._graph_name_hidden = "hidden"
    # Attribute expected by _calculate_input_dim_latent.
    model.node_attributes.attr_ndims["hidden"] = 1

    model._calculate_shapes_and_indices(model.data_indices)

    assert set(model.num_input_channels) == set(model.data_indices)
    assert set(model.num_output_channels) == set(model.data_indices)
    # 2*(2 in_lres + 1 in_hres) + 1*2 noised target + 3 node attrs
    assert model.input_dim["out_hres"] == 6 + 2 + 3
    # Only the anchor has a decoder, so only it has a non-zero target dim.
    assert model.target_dim["out_hres"] == 0


def test_calculate_shapes_and_indices_populates_base_forcing_attributes() -> None:
    """``ForcingsFeature`` reads ``_forcing_input_idx`` / ``num_input_channels_forcings``.

    Any override of ``_calculate_shapes_and_indices`` must keep populating them,
    otherwise ``decoders.*.target_node_features: ["forcings"]`` raises
    ``AttributeError`` deep inside the decoder build.
    """
    model = _make_bare_model(n_step_input=2, n_step_output=1)
    model._graph_name_hidden = "hidden"
    model.node_attributes.attr_ndims["hidden"] = 1

    model._calculate_shapes_and_indices(model.data_indices)

    assert set(model._forcing_input_idx) == set(model.data_indices)
    assert set(model.num_input_channels_forcings) == set(model.data_indices)
    # Only in_hres declares a forcing variable (``z``).
    assert model.num_input_channels_forcings == {"in_lres": 0, "in_hres": 1, "out_hres": 0}
    # Indices address the model *input* tensor layout, as ForcingsFeature expects.
    for dataset_name, dataset_indices in model.data_indices.items():
        assert list(model._forcing_input_idx[dataset_name]) == list(dataset_indices.model.input.forcing)


# ── input assembly ────────────────────────────────────────────────────────────


def test_assemble_input_concatenates_input_datasets_y_noised_and_node_attrs() -> None:
    """Encoder input tensor is [in_lres_vars | in_hres_vars | y_noised_vars | node_attrs]."""
    model = _make_bare_model(n_step_input=1, n_step_output=1)
    batch = 2
    ensemble = 1
    grid = 4
    # per-dataset tensors: shape (batch, time, ensemble, grid, vars)
    x_in_lres = torch.full((batch, 1, ensemble, grid, 2), 1.0)
    x_in_hres = torch.full((batch, 1, ensemble, grid, 1), 2.0)
    # target-only inputs would go here — out_hres has no input variables in our fixture,
    # but the y_noised input still enters via conditioned_target.
    y_noised = torch.full((batch, 1, ensemble, grid, 2), 7.0)

    x_dict = {"in_lres": x_in_lres, "in_hres": x_in_hres}
    conditioned_target = {"out_hres": y_noised}

    latent, _skip, _sharding = model._assemble_input(
        x=x_dict,
        y_noised=conditioned_target,
        bse=batch * ensemble,
        grid_shard_sizes=None,
        model_comm_group=None,
        dataset_name="out_hres",
    )

    # Expected feature dim = 2 (in_lres) + 1 (in_hres) + 2 (y_noised) + 3 (attrs) = 8
    assert latent.shape == (batch * ensemble * grid, 8)
    # Feature slices should match the sources.
    torch.testing.assert_close(latent[:, 0:2], torch.full((batch * ensemble * grid, 2), 1.0))
    torch.testing.assert_close(latent[:, 2:3], torch.full((batch * ensemble * grid, 1), 2.0))
    torch.testing.assert_close(latent[:, 3:5], torch.full((batch * ensemble * grid, 2), 7.0))
    torch.testing.assert_close(latent[:, 5:8], torch.zeros(batch * ensemble * grid, 3))


def test_assemble_input_uses_input_dataset_order_from_encoder_routing() -> None:
    """Assembly must follow the encoder's ``source_datasets`` order so the layout is stable."""
    model = _make_bare_model()
    # Reverse the fused order; ``_assemble_input`` must respect it.
    model.encoder2datasets["enc0"] = ["out_hres", "in_hres", "in_lres"]

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
        dataset_name="out_hres",
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
    # The reference dataset's name_to_index is threaded through as a per-target dict.
    assert ref_name_to_index_by_target == {"out_hres": model.data_indices["in_lres"].name_to_index}


def test_after_sampling_adds_denormalized_lres_to_denormalized_residual() -> None:
    """``_after_sampling`` denormalizes the residual and adds the cached denormalized lres."""
    model = _make_bare_model()

    # Post-processors are inverse-style; a negative offset means "call subtracts".
    post_tend = _AdditiveProcessor(offset=-5.0)  # tendency post — subtracts 5 to denormalize
    post_state = _AdditiveProcessor(offset=-100.0)  # state post — not used here

    batch = 1
    grid = 4
    # Sampled residual for the target dataset (batch, time, ensemble, grid, vars)
    residual_pred = torch.full((batch, 1, 1, grid, 2), 3.0)
    out = {"out_hres": residual_pred}
    # Cached denormalized lres, on same grid, matching output-full channels of target.
    x_lres_denorm = torch.full((batch, 1, 1, grid, 2), 50.0)

    result = model._after_sampling(
        out,
        post_processors={"out_hres": post_state},
        before_sampling_data=(
            {"in_lres": None, "in_hres": None, "out_hres": None},
            {"out_hres": x_lres_denorm},
            {"out_hres": model.data_indices["in_lres"].name_to_index},
        ),
        model_comm_group=None,
        grid_shard_sizes=None,
        gather_out=False,
        post_processors_residual={"out_hres": post_tend},
    )

    # residual_pred is denormalized by post_tend (subtracts 5): 3 - 5 = -2
    # Add cached denormalized lres (50): -2 + 50 = 48
    torch.testing.assert_close(result["out_hres"], torch.full_like(residual_pred, 48.0))


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


def _make_mixed_bare_model() -> AnemoiTransportSpatialDownscalerModelEncProcDec:
    """Bare model with the mixed-target fixture (prognostic + diagnostic in the target)."""
    model = AnemoiTransportSpatialDownscalerModelEncProcDec.__new__(
        AnemoiTransportSpatialDownscalerModelEncProcDec,
    )
    model.data_indices = _make_mixed_downscaler_indices()
    model.dataset_names = list(model.data_indices.keys())
    model.n_step_input = 1
    model.n_step_output = 1
    model.num_input_channels = {name: len(indices.model.input) for name, indices in model.data_indices.items()}
    model.num_output_channels = {name: len(indices.model.output) for name, indices in model.data_indices.items()}
    model.node_attributes = _StaticNodeAttributes(
        {"in_lres": 2, "in_hres": 2, "out_hres": 3},
        grid=4,
    )
    model.target_dataset_names = ["out_hres"]
    _wire_fused_encoder_routing(model, anchor="out_hres", fused=["in_lres", "in_hres"])
    model._roles_by_target = {
        "out_hres": {"reference": "in_lres", "target": "out_hres"},
    }
    return model


class _IndexAwareProcessor:
    """Processor stub that adds ``offset`` and records the ``data_index`` it was called with."""

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
            {
                "shape": tuple(x.shape),
                "data_index": None if data_index is None else data_index.tolist(),
                "kwargs": kwargs,
            },
        )
        return x + self.offset


# ── compute_residual / add_residual_to_state ────────────────────────────────


def test_compute_residual_uses_residual_pre_for_prognostic_and_state_pre_for_diagnostic() -> None:
    """Prognostic channels are normalized as residuals, diagnostic channels as states.
    ``compute_residual`` accepts the *normalized* target and *denormalized* projected
    lres, denormalizes the target via ``input_post_processor`` (state post), then
    fills the output tensor per-channel using the residual pre for prognostics
    and the state pre for diagnostics.
    """
    model = _make_mixed_bare_model()
    indices = model.data_indices["out_hres"]

    # Denormalization step for the target = state post-processor (identity here so the
    # input state values pass through unchanged, keeping the arithmetic simple).
    input_post = _IndexAwareProcessor(offset=0.0)
    state_pre = _IndexAwareProcessor(offset=100.0)  # for diagnostic channels
    residual_pre = _IndexAwareProcessor(offset=10.0)  # for prognostic channels

    # (batch, time, ensemble, grid, target_vars=3)
    y = torch.tensor([[[[[10.0, 20.0, 30.0]]]]])  # t2m=10, u10=20, precip=30
    # Denormalized lres has the same variable positions for t2m and u10.
    x_lres_denorm = torch.tensor([[[[[3.0, 4.0]]]]])  # t2m_lres=3, u10_lres=4

    out = model.compute_residual(
        y={"out_hres": y},
        x_reference_denorm={"out_hres": x_lres_denorm},
        pre_processors_state={"out_hres": state_pre},
        pre_processors_residual={"out_hres": residual_pre},
        reference_variable_name_to_column_index_by_target={"out_hres": model.data_indices["in_lres"].name_to_index},
        input_post_processor={"out_hres": input_post},
        skip_imputation=True,
    )

    # Prognostic channels: residual_pre((y - x_reference) + 0) = (y - x_reference) + 10
    # t2m: (10 - 3) + 10 = 17, u10: (20 - 4) + 10 = 26
    # Diagnostic channel: state_pre(precip) = 30 + 100 = 130
    expected = torch.tensor([[[[[17.0, 26.0, 130.0]]]]])
    torch.testing.assert_close(out["out_hres"], expected)

    # Check the data_index arguments handed to each processor.
    assert residual_pre.calls[0]["data_index"] == indices.data.output.prognostic.tolist()
    assert state_pre.calls[0]["data_index"] == indices.data.output.diagnostic.tolist()


def test_add_residual_to_state_denormalizes_prognostic_with_residual_and_diagnostic_with_state() -> None:
    """Prognostic channels are denormalized with residual post + lres; diagnostics with state post."""
    model = _make_mixed_bare_model()
    indices = model.data_indices["out_hres"]

    # Inverse-style post-processors: negative offsets so ``call`` denormalizes.
    residual_post = _IndexAwareProcessor(offset=-10.0)
    state_post = _IndexAwareProcessor(offset=-100.0)

    # Normalized residual prediction (batch, time, ensemble, grid, vars=3).
    residual = torch.tensor([[[[[17.0, 26.0, 130.0]]]]])
    x_lres_denorm = torch.tensor([[[[[3.0, 4.0]]]]])

    state = model.add_residual_to_state(
        x_reference_denorm={"out_hres": x_lres_denorm},
        residual={"out_hres": residual},
        post_processors_state={"out_hres": state_post},
        post_processors_residual={"out_hres": residual_post},
        reference_variable_name_to_column_index_by_target={"out_hres": model.data_indices["in_lres"].name_to_index},
        output_pre_processor=None,
        skip_imputation=True,
    )

    # Prognostic: residual_post(residual) + x_reference = (17 - 10) + 3 = 10, (26 - 10) + 4 = 20
    # Diagnostic: state_post(residual) = 130 - 100 = 30
    expected = torch.tensor([[[[[10.0, 20.0, 30.0]]]]])
    torch.testing.assert_close(state["out_hres"], expected)

    # Both processors are called; state_post is used specifically for the
    # diagnostic slice (with the corresponding data_index).
    assert len(residual_post.calls) >= 1
    assert any(call["data_index"] == indices.data.output.diagnostic.tolist() for call in state_post.calls)


def test_compute_residual_and_add_residual_to_state_round_trip() -> None:
    """Feeding a target through ``compute_residual`` then ``add_residual_to_state`` recovers it."""
    model = _make_mixed_bare_model()

    # Symmetric pre/post with matching offsets.
    input_post = _IndexAwareProcessor(offset=0.0)
    state_pre = _IndexAwareProcessor(offset=100.0)
    state_post = _IndexAwareProcessor(offset=-100.0)
    residual_pre = _IndexAwareProcessor(offset=10.0)
    residual_post = _IndexAwareProcessor(offset=-10.0)

    y = torch.tensor([[[[[10.0, 20.0, 30.0]]]]])
    x_lres_denorm = torch.tensor([[[[[3.0, 4.0]]]]])

    residual = model.compute_residual(
        y={"out_hres": y},
        x_reference_denorm={"out_hres": x_lres_denorm},
        pre_processors_state={"out_hres": state_pre},
        pre_processors_residual={"out_hres": residual_pre},
        reference_variable_name_to_column_index_by_target={"out_hres": model.data_indices["in_lres"].name_to_index},
        input_post_processor={"out_hres": input_post},
        skip_imputation=True,
    )

    reconstructed = model.add_residual_to_state(
        x_reference_denorm={"out_hres": x_lres_denorm},
        residual=residual,
        post_processors_state={"out_hres": state_post},
        post_processors_residual={"out_hres": residual_post},
        reference_variable_name_to_column_index_by_target={"out_hres": model.data_indices["in_lres"].name_to_index},
        output_pre_processor=None,
        skip_imputation=True,
    )

    torch.testing.assert_close(reconstructed["out_hres"], y)


def test_compute_residual_aligns_lres_columns_by_name_when_layouts_differ() -> None:
    """LRES and target may store variables at different column positions.
    ``compute_residual`` must map the target's prognostic variables to LRES
    columns by *name*, using the provided ``lres_name_to_index`` mapping.
    """
    model = _make_mixed_bare_model()

    input_post = _IndexAwareProcessor(offset=0.0)
    state_pre = _IndexAwareProcessor(offset=0.0)
    residual_pre = _IndexAwareProcessor(offset=0.0)

    # Target layout: [t2m, u10, precip]. LRES layout: [u10, foo, t2m] (t2m is at column 2, u10 at 0).
    lres_name_to_index = {"u10": 0, "foo": 1, "t2m": 2}
    y = torch.tensor([[[[[10.0, 20.0, 30.0]]]]])  # target t2m, u10, precip
    x_lres_denorm = torch.tensor([[[[[4.0, 999.0, 3.0]]]]])  # u10=4, foo (unused), t2m=3

    out = model.compute_residual(
        y={"out_hres": y},
        x_reference_denorm={"out_hres": x_lres_denorm},
        pre_processors_state={"out_hres": state_pre},
        pre_processors_residual={"out_hres": residual_pre},
        reference_variable_name_to_column_index_by_target={"out_hres": lres_name_to_index},
        input_post_processor={"out_hres": input_post},
        skip_imputation=True,
    )

    # Prognostic: t2m residual = 10 - 3 = 7; u10 residual = 20 - 4 = 16.
    # Diagnostic: precip kept as-is = 30.
    expected = torch.tensor([[[[[7.0, 16.0, 30.0]]]]])
    torch.testing.assert_close(out["out_hres"], expected)


def test_add_residual_to_state_aligns_lres_columns_by_name_when_layouts_differ() -> None:
    """Round-trip counterpart to the layout-mismatch test above."""
    model = _make_mixed_bare_model()

    residual_post = _IndexAwareProcessor(offset=0.0)
    state_post = _IndexAwareProcessor(offset=0.0)

    lres_name_to_index = {"u10": 0, "foo": 1, "t2m": 2}
    residual = torch.tensor([[[[[7.0, 16.0, 30.0]]]]])
    x_lres_denorm = torch.tensor([[[[[4.0, 999.0, 3.0]]]]])

    state = model.add_residual_to_state(
        x_reference_denorm={"out_hres": x_lres_denorm},
        residual={"out_hres": residual},
        post_processors_state={"out_hres": state_post},
        post_processors_residual={"out_hres": residual_post},
        reference_variable_name_to_column_index_by_target={"out_hres": lres_name_to_index},
        output_pre_processor=None,
        skip_imputation=True,
    )

    # Prognostic: t2m = 7 + 3 = 10; u10 = 16 + 4 = 20. Diagnostic: precip = 30.
    expected = torch.tensor([[[[[10.0, 20.0, 30.0]]]]])
    torch.testing.assert_close(state["out_hres"], expected)


def test_compute_residual_raises_when_target_prognostic_missing_from_lres() -> None:
    """A clear error is raised if the reference dataset lacks a target prognostic variable."""
    model = _make_mixed_bare_model()

    # Reference has u10 but not t2m — the model can't compute the t2m residual.
    reference_name_to_index = {"u10": 0}

    with pytest.raises(KeyError, match=r"t2m"):
        model.compute_residual(
            y={"out_hres": torch.zeros(1, 1, 1, 1, 3)},
            x_reference_denorm={"out_hres": torch.zeros(1, 1, 1, 1, 1)},
            pre_processors_state={"out_hres": _IndexAwareProcessor(offset=0.0)},
            pre_processors_residual={"out_hres": _IndexAwareProcessor(offset=0.0)},
            reference_variable_name_to_column_index_by_target={"out_hres": reference_name_to_index},
            input_post_processor={"out_hres": _IndexAwareProcessor(offset=0.0)},
            skip_imputation=True,
        )


def test_add_residual_to_state_raises_when_target_prognostic_missing_from_reference() -> None:
    """Same validation as ``compute_residual`` but on the reverse operation."""
    model = _make_mixed_bare_model()
    reference_name_to_index = {"u10": 0}
    with pytest.raises(KeyError, match=r"t2m"):
        model.add_residual_to_state(
            x_reference_denorm={"out_hres": torch.zeros(1, 1, 1, 1, 1)},
            residual={"out_hres": torch.zeros(1, 1, 1, 1, 3)},
            post_processors_state={"out_hres": _IndexAwareProcessor(offset=0.0)},
            post_processors_residual={"out_hres": _IndexAwareProcessor(offset=0.0)},
            reference_variable_name_to_column_index_by_target={"out_hres": reference_name_to_index},
            output_pre_processor=None,
            skip_imputation=True,
        )


# ── _after_sampling with mixed target ───────────────────────────────────────


def test_after_sampling_mixed_target_uses_state_post_for_diagnostic_and_residual_post_for_prognostic() -> None:
    """With a diagnostic variable in the target, ``_after_sampling`` splits per-channel."""
    model = _make_mixed_bare_model()

    residual_post = _IndexAwareProcessor(offset=-5.0)
    state_post = _IndexAwareProcessor(offset=-50.0)

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


# ── _resolve_roles: explicit config-based role resolution ───────────────────


def test_resolve_roles_records_the_reference_for_each_target() -> None:
    """``_roles_by_target`` is authoritative for the residual baseline.

    Which datasets are *encoded* with the target is decided by the encoder
    routing, not here.
    """
    model = AnemoiTransportSpatialDownscalerModelEncProcDec.__new__(
        AnemoiTransportSpatialDownscalerModelEncProcDec,
    )
    model.data_indices = {
        "in_lres": _make_index_collection({"t2m": 0}),
        "out_hres": _make_index_collection({"t2m": 0, "precip": 1}, diagnostic=["precip"]),
    }
    model.dataset_names = list(model.data_indices.keys())
    config = {
        "training": {
            "transport": {
                "encoder_decoder_roles": {
                    "enc_dec_0": {"reference": "in_lres", "target": "out_hres"},
                },
            },
        },
    }
    model._resolve_roles(config)
    assert model.target_dataset_names == ["out_hres"]
    assert model._roles_by_target["out_hres"]["reference"] == "in_lres"


def test_validate_roles_match_encoder_routing_accepts_a_consistent_setup() -> None:
    model = _make_bare_model()

    model._validate_roles_match_encoder_routing()


def test_validate_roles_match_encoder_routing_rejects_a_reference_that_is_not_encoded() -> None:
    """A reference the encoder never sees would silently train on the wrong baseline."""
    model = _make_bare_model()
    model.encoder2datasets["enc0"] = ["out_hres", "in_hres"]

    with pytest.raises(ValueError, match="is not encoded with it"):
        model._validate_roles_match_encoder_routing()


def test_validate_roles_match_encoder_routing_rejects_targets_the_decoders_do_not_declare() -> None:
    model = _make_bare_model()
    model.target_dataset_names = ["out_hres", "other"]

    with pytest.raises(ValueError, match="but the decoders declare"):
        model._validate_roles_match_encoder_routing()


def test_validate_roles_match_encoder_routing_rejects_conditioning_that_is_not_encoded() -> None:
    """``conditioning`` is now only a hint; the encoder's source_datasets decide.

    Accepting a conditioning input the encoder never sees would silently train a
    different model than the config describes.
    """
    model = _make_bare_model()
    model._roles_by_target["out_hres"]["conditioning"] = "in_static"

    with pytest.raises(ValueError, match="conditioning dataset 'in_static'"):
        model._validate_roles_match_encoder_routing()


def test_resolve_roles_raises_without_encoder_decoder_roles() -> None:
    """Raises ``ValueError`` when called with a config that has no ``encoder_decoder_roles``."""
    model = AnemoiTransportSpatialDownscalerModelEncProcDec.__new__(
        AnemoiTransportSpatialDownscalerModelEncProcDec,
    )
    model.data_indices = _make_downscaler_indices()
    model.dataset_names = list(model.data_indices.keys())
    with pytest.raises(ValueError, match="encoder_decoder_roles"):
        model._resolve_roles({"training": {"transport": {}}})


# ── _resolve_roles: reference/target prognostic role consistency ───────────


def _resolve_roles_config(
    target: str = "out_hres",
    reference: str = "in_lres",
    conditioning: str | None = None,
) -> dict:
    role: dict[str, str] = {"reference": reference, "target": target}
    if conditioning is not None:
        role["conditioning"] = conditioning
    return {"training": {"transport": {"encoder_decoder_roles": {"enc_dec_0": role}}}}


def test_resolve_roles_rejects_target_prognostic_absent_from_reference() -> None:
    """A target prognostic that does not exist at all in the reference is a config error."""
    model = AnemoiTransportSpatialDownscalerModelEncProcDec.__new__(
        AnemoiTransportSpatialDownscalerModelEncProcDec,
    )
    model.data_indices = {
        "in_lres": _make_index_collection({"u10": 0}),
        "out_hres": _make_index_collection({"t2m": 0}),
    }
    model.dataset_names = list(model.data_indices.keys())
    with pytest.raises(ValueError, match=r"only in target: \['t2m'\]"):
        model._resolve_roles(_resolve_roles_config())


def test_resolve_roles_rejects_reference_prognostic_absent_from_target() -> None:
    """A prognostic in the reference that the target does not predict prognostically is also a config error."""
    model = AnemoiTransportSpatialDownscalerModelEncProcDec.__new__(
        AnemoiTransportSpatialDownscalerModelEncProcDec,
    )
    model.data_indices = {
        "in_lres": _make_index_collection({"t2m": 0, "u10": 1}),  # both prognostic
        "out_hres": _make_index_collection({"t2m": 0}),  # only t2m
    }
    model.dataset_names = list(model.data_indices.keys())
    with pytest.raises(ValueError, match=r"only in reference: \['u10'\]"):
        model._resolve_roles(_resolve_roles_config())


# ── end-to-end construction ─────────────────────────────────────────────────


def _make_downscaler_graph(grid: int = 4, hidden: int = 3) -> HeteroData:
    """Graph with encoder/decoder edges anchored at ``out_hres`` only.

    ``in_lres`` and ``in_hres`` are node sets without encoder edges: they are
    fused onto the anchor's grid, so they never anchor a mapper.
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

    for relation, (num_src, num_dst) in {
        ("out_hres", "to", "hidden"): (grid, hidden),
        ("hidden", "to", "out_hres"): (hidden, grid),
        ("hidden", "to", "hidden"): (hidden, hidden),
    }.items():
        edge_index = _dense_edges(num_src, num_dst)
        graph[relation].edge_index = edge_index
        graph[relation].edge_length = torch.zeros(edge_index.shape[1], 1)
    return graph


def _make_downscaler_config(num_channels: int = 8) -> DictConfig:
    def _gnn(target: str, **extra: Any) -> dict[str, Any]:
        return {
            "_target_": target,
            "num_channels": num_channels,
            "num_chunks": 1,
            "mlp_extra_layers": 0,
            "mlp_hidden_ratio": 1,
            "cpu_offload": False,
            "layer_kernels": {},
            "sub_graph_edge_attributes": ["edge_length"],
            **extra,
        }

    return DictConfig(
        {
            "model": {
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
                        "source_datasets": ["out_hres", "in_lres", "in_hres"],
                        "dataset_fusing_strategy": "concatenate_inputs_along_variable_dim",
                        "fusion_anchor": "out_hres",
                        "mapper": _gnn("anemoi.models.layers.mapper.GNNForwardMapper"),
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
                        "mapper": _gnn("anemoi.models.layers.mapper.GNNBackwardMapper"),
                    },
                },
                "residual": {
                    "datasets": {
                        name: {"_target_": "anemoi.models.layers.residual.SkipConnection"}
                        for name in ("out_hres", "in_lres", "in_hres")
                    },
                },
                "bounding": {"datasets": {name: [] for name in ("out_hres", "in_lres", "in_hres")}},
            },
            "training": {
                "transport": {
                    "encoder_decoder_roles": {
                        "enc_dec_0": {"reference": "in_lres", "target": "out_hres"},
                    },
                },
            },
        },
    )


def _build_real_downscaler(**config_overrides: Any) -> AnemoiTransportSpatialDownscalerModelEncProcDec:
    config = _make_downscaler_config()
    for path, value in config_overrides.items():
        OmegaConf.update(config, path.replace("__", "."), value, force_add=True)
    return AnemoiTransportSpatialDownscalerModelEncProcDec(
        model_config=config,
        data_indices=_make_downscaler_indices(),
        statistics={name: None for name in ("out_hres", "in_lres", "in_hres")},
        n_step_input=1,
        n_step_output=1,
        graph_data=_make_downscaler_graph(),
    )


def test_real_construction_builds_one_encoder_decoder_pair_anchored_at_the_target() -> None:
    """Every other test wires routing by hand, so this is the only check that
    ``__init__`` (routing, dimension arithmetic, network build) agrees with itself.
    """
    model = _build_real_downscaler()

    # Fused inputs share the anchor's encoder and node set but own no mapper.
    assert model.input_datasets == ["out_hres"]
    assert model.target_datasets == ["out_hres"]
    assert set(model.encoder.keys()) == {"enc0"}
    assert set(model.decoder.keys()) == {"dec0"}
    assert set(model.encoder_graph_provider.keys()) == {"out_hres"}
    assert set(model.decoder_graph_provider.keys()) == {"out_hres"}
    assert model.encoder_node_set("in_lres") == "out_hres"

    # in_lres (2 vars) + in_hres (1 var) history, noised target (2 vars), node attrs.
    assert model.input_dim["out_hres"] == 3 + 2 + model.node_attributes.attr_ndims["out_hres"]


def test_real_construction_forward_returns_the_target_on_its_own_grid() -> None:
    model = _build_real_downscaler()
    batch, ensemble, grid = 1, 1, 4

    x = {
        "in_lres": torch.zeros(batch, 1, ensemble, grid, 2),
        "in_hres": torch.zeros(batch, 1, ensemble, grid, 1),
    }
    conditioned_target = {"out_hres": torch.zeros(batch, 1, ensemble, grid, 2)}
    condition = {"out_hres": torch.full((batch, 1, ensemble, 1, 1), 0.5)}

    out = model._forward_transport_network(x=x, conditioned_target=conditioned_target, condition=condition)

    assert set(out) == {"out_hres"}
    assert out["out_hres"].shape == (batch, 1, ensemble, grid, 2)


def test_real_construction_rejects_a_reference_that_is_not_fused_into_the_encoder() -> None:
    """The residual baseline must be one of the encoder's source datasets."""
    with pytest.raises(ValueError, match="is not encoded with it"):
        _build_real_downscaler(
            model__encoders__enc0__source_datasets=["out_hres", "in_hres"],
        )

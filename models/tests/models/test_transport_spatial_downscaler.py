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

from typing import Any

import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
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
    """Three datasets: in_lres (forcing-like), in_hres (forcing-like), out_hres (target)."""
    # in_lres: two variables, both treated as forcing so model.output is empty.
    in_lres = _make_index_collection(
        {"t2m": 0, "u10": 1},
        forcing=["t2m", "u10"],
    )
    # in_hres: one forcing variable, model.output empty.
    in_hres = _make_index_collection(
        {"z": 0},
        forcing=["z"],
    )
    # out_hres: two output variables. No forcing / diagnostic / target overrides:
    # both variables are prognostic (present in input and output).
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
    """Spatial pre-processor stub that just passes input through."""

    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def __call__(self, x: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
        self.calls.append(x)
        return x


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
    model._resolve_roles()
    return model


# ── role inference ────────────────────────────────────────────────────────────


def test_resolve_roles_identifies_target_and_input_datasets_from_indices() -> None:
    """The single dataset with non-empty ``model.output`` is the target; the rest are inputs."""
    model = _make_bare_model()
    assert model.target_dataset_name == "out_hres"
    assert set(model.input_dataset_names) == {"in_lres", "in_hres"}


def test_resolve_roles_rejects_zero_targets() -> None:
    """If no dataset has model output variables the model refuses to init."""
    model = AnemoiTransportSpatialDownscalerModelEncProcDec.__new__(
        AnemoiTransportSpatialDownscalerModelEncProcDec,
    )
    # Both datasets have only forcing variables → no model output.
    model.data_indices = {
        "in_lres": _make_index_collection({"t2m": 0}, forcing=["t2m"]),
        "in_hres": _make_index_collection({"z": 0}, forcing=["z"]),
    }
    model.dataset_names = list(model.data_indices.keys())
    with pytest.raises(ValueError, match="exactly one target dataset"):
        model._resolve_roles()


def test_resolve_roles_rejects_more_than_one_target() -> None:
    """Two datasets with model output variables must be rejected."""
    model = AnemoiTransportSpatialDownscalerModelEncProcDec.__new__(
        AnemoiTransportSpatialDownscalerModelEncProcDec,
    )
    model.data_indices = {
        "out_a": _make_index_collection({"v": 0}),
        "out_b": _make_index_collection({"v": 0}),
    }
    model.dataset_names = list(model.data_indices.keys())
    with pytest.raises(ValueError, match="exactly one target dataset"):
        model._resolve_roles()


# ── dimension arithmetic ─────────────────────────────────────────────────────


def test_calculate_input_dim_sums_all_input_datasets_plus_noised_target_and_node_attrs() -> None:
    """input_dim = sum(input_dataset input vars over history) + noised_target + target node attrs."""
    # in_lres has 2 input vars, in_hres has 1, out_hres has 2 output vars.
    # attr_ndims for out_hres is 3.
    model = _make_bare_model(n_step_input=2, n_step_output=1)
    # sum of input vars across input datasets (over history): 2*(2+1) = 6
    # noised target: 1 * 2 = 2
    # target node attrs: 3
    assert model._calculate_input_dim("out_hres") == 6 + 2 + 3


def test_calculate_input_dim_ignores_non_target_datasets() -> None:
    """_calculate_input_dim is only meaningful for the target dataset."""
    model = _make_bare_model()
    with pytest.raises(ValueError, match="only defined for the target"):
        model._calculate_input_dim("in_lres")


def test_calculate_shapes_and_indices_only_sizes_target_but_channels_for_all() -> None:
    """Base ``_calculate_shapes_and_indices`` iterates over every dataset and

    calls ``_calculate_input_dim`` on each — which raises for input datasets.
    The downscaler must override this so that ``num_input_channels`` /
    ``num_output_channels`` are populated for every dataset (they are needed to
    compute the target's encoder input dim), while ``input_dim``, ``target_dim``
    and ``output_dim`` are only computed for the target dataset.
    """
    model = _make_bare_model(n_step_input=2, n_step_output=1)
    model._graph_name_hidden = "hidden"
    # Attribute expected by _calculate_input_dim_latent.
    model.node_attributes.attr_ndims["hidden"] = 1

    model._calculate_shapes_and_indices(model.data_indices)

    # num_input_channels / num_output_channels populated for every dataset.
    assert set(model.num_input_channels) == set(model.data_indices)
    assert set(model.num_output_channels) == set(model.data_indices)

    # But per-dataset input/target/output dims exist only for the target.
    assert set(model.input_dim) == {"out_hres"}
    assert set(model.target_dim) == {"out_hres"}
    # in_lres and in_hres have no output channels, so their output_dim is 0.
    # Only the target's output_dim is used to construct the decoder.
    assert set(model.output_dim) == {"out_hres"}


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


def test_assemble_input_uses_input_dataset_order_from_role_resolution() -> None:
    """Assembly must iterate ``input_dataset_names`` in a fixed order so the encoder sees a stable layout."""
    model = _make_bare_model()
    # Force a deterministic order by mutating the list; ``_assemble_input`` must respect it.
    model.input_dataset_names = ["in_hres", "in_lres"]

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

    xs, x_lres_denorm = result
    # The projector must have been called with the raw lres batch first,
    # before any normalization.
    assert len(projector.calls) == 1
    torch.testing.assert_close(projector.calls[0], batch_lres.unsqueeze(2))  # add ensemble dim
    # After spatial projection, ``pre_processors["in_lres"]`` is applied (adds 10).
    torch.testing.assert_close(xs["in_lres"], batch_lres.unsqueeze(2) + 10.0)
    torch.testing.assert_close(xs["in_hres"], batch_hres.unsqueeze(2) + 20.0)
    # x_lres_denorm caches the denormalized projected lres:
    # normalized (batch + 10) then denormalized (subtract 10) = batch.
    torch.testing.assert_close(x_lres_denorm, batch_lres.unsqueeze(2))


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
        before_sampling_data=({"in_lres": None, "in_hres": None, "out_hres": None}, x_lres_denorm),
        model_comm_group=None,
        grid_shard_sizes=None,
        gather_out=False,
        post_processors_tendencies={"out_hres": post_tend},
    )

    # residual_pred is denormalized by post_tend (subtracts 5): 3 - 5 = -2
    # Add cached denormalized lres (50): -2 + 50 = 48
    torch.testing.assert_close(result["out_hres"], torch.full_like(residual_pred, 48.0))


# ── mixed-target (prognostic + diagnostic) fixtures ─────────────────────────


def _make_mixed_downscaler_indices() -> dict[str, IndexCollection]:
    """Target with two prognostic and one diagnostic variable.

    The lres dataset has the two prognostic variables at the same variable
    indices as the target, matching how spatial downscaling typically pairs
    lres/hres channels.  The diagnostic variable (``precip``) has no lres
    counterpart — it is predicted directly as a state.
    """
    in_lres = _make_index_collection(
        {"t2m": 0, "u10": 1},
        forcing=["t2m", "u10"],
    )
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
    model._resolve_roles()
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
        x_lres_denorm={"out_hres": x_lres_denorm},
        pre_processors_state={"out_hres": state_pre},
        pre_processors_residual={"out_hres": residual_pre},
        input_post_processor={"out_hres": input_post},
        skip_imputation=True,
    )

    # Prognostic channels: residual_pre((y - x_lres) + 0) = (y - x_lres) + 10
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
        x_lres_denorm={"out_hres": x_lres_denorm},
        residual={"out_hres": residual},
        post_processors_state={"out_hres": state_post},
        post_processors_residual={"out_hres": residual_post},
        output_pre_processor=None,
        skip_imputation=True,
    )

    # Prognostic: residual_post(residual) + x_lres = (17 - 10) + 3 = 10, (26 - 10) + 4 = 20
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
        x_lres_denorm={"out_hres": x_lres_denorm},
        pre_processors_state={"out_hres": state_pre},
        pre_processors_residual={"out_hres": residual_pre},
        input_post_processor={"out_hres": input_post},
        skip_imputation=True,
    )

    reconstructed = model.add_residual_to_state(
        x_lres_denorm={"out_hres": x_lres_denorm},
        residual=residual,
        post_processors_state={"out_hres": state_post},
        post_processors_residual={"out_hres": residual_post},
        output_pre_processor=None,
        skip_imputation=True,
    )

    torch.testing.assert_close(reconstructed["out_hres"], y)


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
        before_sampling_data=({"in_lres": None, "in_hres": None, "out_hres": None}, x_lres_denorm),
        model_comm_group=None,
        grid_shard_sizes=None,
        gather_out=False,
        post_processors_tendencies={"out_hres": residual_post},
    )

    # Prognostic: (residual_pred - 5) + x_lres = (7-5)+3 = 5, (12-5)+4 = 11
    # Diagnostic: residual_pred - 50 = 55 - 50 = 5
    expected = torch.tensor([[[[[5.0, 11.0, 5.0]]]]])
    torch.testing.assert_close(result["out_hres"], expected)


# ── _resolve_roles TODO fix: config-aware role inference ────────────────────


def test_resolve_roles_treats_spatial_processor_source_as_input_even_when_it_has_output_vars() -> None:
    """A dataset registered under ``config.data.spatial_processors`` is always an input.

    This makes the role inference robust to configurations where the low-res
    dataset carries variables classified as prognostic (i.e. non-empty
    ``model.output``): the spatial-processor registration is authoritative.
    """
    model = AnemoiTransportSpatialDownscalerModelEncProcDec.__new__(
        AnemoiTransportSpatialDownscalerModelEncProcDec,
    )
    # in_lres has t2m as prognostic (present in input AND output), so
    # ``model.output`` is non-empty — but the config marks it as a
    # spatial-processor source, so it must still be recognised as an input.
    model.data_indices = {
        "in_lres": _make_index_collection({"t2m": 0}),
        "out_hres": _make_index_collection({"t2m": 0, "precip": 1}, diagnostic=["precip"]),
    }
    model.dataset_names = list(model.data_indices.keys())
    model_config = DictConfig(
        {"data": {"spatial_processors": {"in_lres": {"_target_": "some.CrossGridProjector"}}}},
    )
    model._resolve_roles(model_config=model_config)
    assert model.target_dataset_name == "out_hres"
    assert model.input_dataset_names == ["in_lres"]


def test_resolve_roles_without_config_falls_back_to_output_heuristic() -> None:
    """Backward-compatible: calling ``_resolve_roles`` without a config still works."""
    model = _make_bare_model()  # this calls ``_resolve_roles()`` without config
    assert model.target_dataset_name == "out_hres"
    assert set(model.input_dataset_names) == {"in_lres", "in_hres"}

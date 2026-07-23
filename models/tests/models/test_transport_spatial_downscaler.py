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

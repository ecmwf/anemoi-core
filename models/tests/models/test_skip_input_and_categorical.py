# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``skip_input`` redirects the residual without touching the encoder; categorical inputs are embedded.

The models are built with the small grid + observation configuration of the sharding tests.
"""

from collections.abc import Iterator
from functools import partial

import pytest
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

# Imported for their side effect: __subclasses__() only sees classes whose module has been imported.
import anemoi.models.models.hierarchical  # noqa: F401
import anemoi.models.models.transport_encoder_processor_decoder  # noqa: F401
from anemoi.models.data import Batch
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.layers.categorical import check_categorical_preprocessing
from anemoi.models.models.base import BaseGraphModel
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.models.ens_encoder_processor_decoder import AnemoiEnsModelEncProcDec
from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing.imputer import ConstantImputer
from anemoi.models.preprocessing.normalizer import InputNormalizer
from anemoi.models.utils.graphs import build_graph
from anemoi.utils.config import DotDict
from tests.distributed.test_model_sharding_gradients import DTYPE
from tests.distributed.test_model_sharding_gradients import GRID_VARIABLES
from tests.distributed.test_model_sharding_gradients import NUM_GRID_POINTS
from tests.distributed.test_model_sharding_gradients import _data_indices
from tests.distributed.test_model_sharding_gradients import _graph_config
from tests.distributed.test_model_sharding_gradients import _gridded
from tests.distributed.test_model_sharding_gradients import _model_config
from tests.distributed.test_model_sharding_gradients import _points_on_sphere
from tests.distributed.test_model_sharding_gradients import _statistics
from tests.distributed.test_model_sharding_gradients import _synthetic_batches

CODES = [3, 7, 11]


@pytest.fixture(autouse=True)
def _graphs_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build the graphs on the CPU, where the model and the data are, also on a GPU node."""
    monkeypatch.setenv("ANEMOI_GRAPHS_FORCE_CPU", "1")


def _build(
    model_class: type = AnemoiModelEncProcDec,
    categorical: dict | None = None,
    data_indices: dict | None = None,
    grid_target_features: list[str] | None = None,
) -> torch.nn.Module:
    generator = torch.Generator().manual_seed(0)
    latitudes, longitudes = _points_on_sphere(NUM_GRID_POINTS, generator)
    config = _model_config(model_class, obs_predicted=True)
    if categorical is not None:
        config.categorical_embeddings = OmegaConf.create(categorical)
    if grid_target_features is not None:
        config.decoders.grid.target_node_features = grid_target_features
    torch.manual_seed(1)
    static_graph_data, dynamic_graph_config = build_graph(
        DotDict(_graph_config(latitudes, longitudes, max_obs_neighbours=4096)),
        is_dataset_static={"grid": True, "obs": False},
        hidden_nodes_name="hidden",
    )
    model = model_class(
        model_config=config,
        static_graph_data=static_graph_data,
        dynamic_graph_config=dynamic_graph_config,
        data_indices=data_indices or _data_indices(),
        statistics=_statistics(),
        n_step_input={"grid": 2, "obs": 2},
        n_step_output={"grid": 1, "obs": 1},
    )
    if isinstance(model, AnemoiEnsModelEncProcDec):
        # The ensemble model draws its noise in the injector's default float32.
        model.noise_injector.forward = partial(model.noise_injector.forward, noise_dtype=DTYPE)
    return model.to(DTYPE)


def _forward(model: torch.nn.Module, batches: dict, **kwargs):
    return model(
        batches["x"],
        batches["target_forcing"],
        {name: source.template() for name, source in batches["noisy_target"].items()},
        **kwargs,
    )


def _with_grid_data(batch, data: torch.Tensor):
    return batch.replace("grid", batch["grid"].clone(data=data))


@pytest.mark.parametrize("model_class", [AnemoiModelEncProcDec, AnemoiEnsModelEncProcDec])
def test_skip_input_redirects_residual_but_not_encoder(model_class: type) -> None:
    model = _build(model_class)
    batches = _synthetic_batches(model)
    x = batches["x"]
    background = x["grid"].data + torch.randn_like(x["grid"].data)
    skip_input = _with_grid_data(x, background)

    with torch.no_grad():
        default = _forward(model, batches)
        same = _forward(model, batches, skip_input=x)
        redirected = _forward(model, batches, skip_input=skip_input)

    # Handing the input itself as the residual base changes nothing.
    assert torch.equal(same["grid"].data, default["grid"].data)
    # The SkipConnection adds the last input step to the prognostic outputs: with another base,
    # the output moves by exactly the difference of the bases there, and nowhere else.
    indices = _data_indices()["grid"]
    delta = redirected["grid"].data - default["grid"].data
    expected = (background - x["grid"].data)[:, -1:, ..., indices.model.input.prognostic]
    torch.testing.assert_close(delta[..., indices.model.output.prognostic], expected)
    assert torch.all(delta[..., indices.model.output.diagnostic] == 0)
    # The observation dataset has no residual, so its output is untouched.
    assert torch.equal(redirected["obs"].flatten().data, default["obs"].flatten().data)


def test_skip_input_shape_mismatch_raises() -> None:
    model = _build()
    batches = _synthetic_batches(model)
    x = batches["x"]
    skip_input = _with_grid_data(x, x["grid"].data[:, -1:])

    with pytest.raises(ValueError, match="skip_input"):
        _forward(model, batches, skip_input=skip_input)


def _subclasses(cls: type) -> Iterator[type]:
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


# Subclasses that replace ``forward`` and do thread ``skip_input``; each needs a behavioural test above.
_VERIFIED_SKIP_INPUT_OVERRIDES = {"AnemoiEnsModelEncProcDec"}


def test_skip_input_support_flags() -> None:
    assert BaseGraphModel.supports_skip_input is False
    assert AnemoiModelEncProcDec.supports_skip_input is True
    offenders = [
        cls.__name__
        for cls in _subclasses(AnemoiModelEncProcDec)
        if cls.forward is not AnemoiModelEncProcDec.forward
        and cls.supports_skip_input
        and cls.__name__ not in _VERIFIED_SKIP_INPUT_OVERRIDES
    ]
    assert not offenders, f"{offenders} override forward without threading skip_input, yet advertise it."


def test_overriding_forward_opts_out_automatically() -> None:
    class Overrides(AnemoiModelEncProcDec):
        def forward(self, x, **kwargs):  # noqa: ARG002
            return x

    class OptsBackIn(AnemoiModelEncProcDec):
        supports_skip_input = True

        def forward(self, x, **kwargs):  # noqa: ARG002
            return x

    class Inherits(AnemoiModelEncProcDec):
        pass

    assert Overrides.supports_skip_input is False
    assert OptsBackIn.supports_skip_input is True
    assert Inherits.supports_skip_input is True


# ── categorical embeddings ──────────────────────────────────────────────


def _categorical(embedding_dim: int = 4) -> dict:
    return {"grid": {"f": {"codes": CODES, "embedding_dim": embedding_dim}}}


def _with_codes(batches: dict, codes: list[float]) -> dict:
    """Put integer codes into the forcing ``f`` of the grid input."""
    x = batches["x"]
    data = x["grid"].data.clone()
    f = x["grid"].variables.index("f")
    data[..., f] = torch.tensor(codes, dtype=data.dtype).repeat(NUM_GRID_POINTS // len(codes) + 1)[:NUM_GRID_POINTS]
    return {**batches, "x": _with_grid_data(x, data)}


@pytest.mark.parametrize("model_class", [AnemoiModelEncProcDec, AnemoiEnsModelEncProcDec])
def test_input_dim_and_forward_with_embedding(model_class: type) -> None:
    plain = _build(model_class)
    model = _build(model_class, categorical=_categorical(embedding_dim=4))

    # Two input steps, each swapping one raw-code channel for four embedding channels.
    assert model.input_dim["grid"] == plain.input_dim["grid"] + 2 * 3
    assert model.input_dim["obs"] == plain.input_dim["obs"]

    batches = _with_codes(_synthetic_batches(model), [0.0, *CODES, 99.0])
    out = _forward(model, batches)
    out["grid"].data.sum().backward()

    weight = model.categorical_embeddings["grid"]["f"].embedding.weight
    assert weight.grad is not None
    assert torch.isfinite(out["grid"].data).all()


def test_non_integer_codes_raise() -> None:
    model = _build(categorical=_categorical())
    batches = _with_codes(_synthetic_batches(model), [0.5])

    with pytest.raises(ValueError, match="integer"):
        _forward(model, batches)


def test_unsupported_models_refuse_the_key() -> None:
    class RebuildsInput(AnemoiModelEncProcDec):
        def _assemble_input(self, *args, **kwargs):
            return super()._assemble_input(*args, **kwargs)

    assert RebuildsInput.supports_categorical_embeddings is False
    assert AnemoiEnsModelEncProcDec.supports_categorical_embeddings is True
    with pytest.raises(NotImplementedError, match="categorical_embeddings"):
        _build(RebuildsInput, categorical=_categorical())


def test_categorical_variable_must_be_a_model_input() -> None:
    with pytest.raises(ValueError, match="not model inputs"):
        _build(categorical={"grid": {"d": {"codes": CODES}}})


def test_preprocessing_check(caplog: pytest.LogCaptureFixture) -> None:
    def processors(normalizer: dict, imputer: dict) -> Processors:
        return Processors(
            [
                ["normalizer", InputNormalizer(config=normalizer)],
                ["imputer", ConstantImputer(config=imputer)],
            ],
        )

    with pytest.raises(ValueError, match="rescaled"):
        check_categorical_preprocessing("grid", ["f"], processors({"default": "mean-std"}, {"default": "none"}))

    check_categorical_preprocessing(
        "grid", ["f"], processors({"default": "mean-std", "none": ["f"]}, {"default": "none", 0.0: ["f"]})
    )
    assert "non-zero value" not in caplog.text

    check_categorical_preprocessing(
        "grid", ["f"], processors({"default": "mean-std", "none": ["f"]}, {"default": "none", 5.0: ["f"]})
    )
    assert "non-zero value" in caplog.text


# ── inference: predict_step prepares a raw residual base like the input ──


class _Shift(torch.nn.Module):
    """Stands in for a dataset's pre-processors: a shift makes the zero base detectable."""

    def forward(self, source, in_place: bool = True, **_kwargs):  # noqa: ARG002, FBT001, FBT002
        return source.map_data(lambda t: t * 2.0 + 1.0)


class _Identity(torch.nn.Module):
    def forward(self, source, in_place: bool = True, **_kwargs):  # noqa: ARG002, FBT001, FBT002
        return source


def _predict(model, batches: dict, **kwargs):
    processors = torch.nn.ModuleDict({"grid": _Shift(), "obs": _Shift()})
    return model.predict_step(
        batches["x"],
        {name: source.template() for name, source in batches["noisy_target"].items()},
        processors,
        torch.nn.ModuleDict({"grid": _Identity(), "obs": _Identity()}),
        batches["target_forcing"],
        **kwargs,
    )


def _prepared(batch):
    return batch.with_sources({name: _Shift()(source) for name, source in batch.items()})


def test_predict_step_normalises_the_raw_residual_base() -> None:
    model = _build()
    batches = _synthetic_batches(model)
    raw_background = _with_grid_data(batches["x"], batches["x"]["grid"].data + 1.0)
    out = _predict(model, batches, skip_input=Batch({"grid": raw_background["grid"]}))
    with torch.no_grad():
        expected = _forward(
            model,
            {**batches, "x": _prepared(batches["x"]), "target_forcing": _prepared(batches["target_forcing"])},
            skip_input=_prepared(raw_background),
        )
    torch.testing.assert_close(out["grid"].data, expected["grid"].data)


def test_predict_step_cold_start_base_is_zero_after_normalisation() -> None:
    model = _build()
    batches = _synthetic_batches(model)
    out = _predict(model, batches, skip_input=Batch({}))
    x = _prepared(batches["x"])
    with torch.no_grad():
        expected = _forward(
            model,
            {**batches, "x": x, "target_forcing": _prepared(batches["target_forcing"])},
            skip_input=x.map_data(torch.zeros_like),
        )
        default = _forward(model, {**batches, "x": x, "target_forcing": _prepared(batches["target_forcing"])})
    torch.testing.assert_close(out["grid"].data, expected["grid"].data)
    assert not torch.allclose(out["grid"].data, default["grid"].data)


# ── decoder forcings ─────────────────────────────────────────────────────


def _decoder_forcing_indices() -> dict:
    """The grid's "b" conditions the decoder at the output times only; "f" is a forcing."""
    indices = _data_indices()
    indices["grid"] = IndexCollection(
        DictConfig({"forcing": ["f"], "diagnostic": ["d"], "target": [], "decoder_forcing": ["b"]}),
        {name: index for index, name in enumerate(GRID_VARIABLES)},
    )
    return indices


def test_decoder_forcings_widen_the_target_forcings(monkeypatch: pytest.MonkeyPatch) -> None:
    plain = _build()
    model = _build(data_indices=_decoder_forcing_indices())
    # "b" is no longer an encoder input, but one more target-forcing channel of the decoder
    assert model.num_input_channels["grid"] == plain.num_input_channels["grid"] - 1
    assert model.num_target_forcing_channels["grid"] == plain.num_target_forcing_channels["grid"] + 1

    monkeypatch.setattr("tests.distributed.test_model_sharding_gradients._data_indices", _decoder_forcing_indices)
    batches = _synthetic_batches(model)
    coordinates = batches["target_forcing"]["grid"].coordinates
    forcings = _gridded(["b", "f"], torch.randn(1, 1, 1, NUM_GRID_POINTS, 2, dtype=DTYPE), coordinates)
    batches["target_forcing"] = batches["target_forcing"].replace("grid", forcings)
    with torch.no_grad():
        out = _forward(model, batches)
    assert torch.isfinite(out["grid"].data).all()


def test_decoder_forcings_need_the_target_forcings_feature() -> None:
    with pytest.raises(ValueError, match="target_forcings"):
        _build(data_indices=_decoder_forcing_indices(), grid_target_features=["encoded_data"])

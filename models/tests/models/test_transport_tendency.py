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
from omegaconf import DictConfig

from anemoi.models.data import Batch
from anemoi.models.data.layout import TensorLayout
from anemoi.models.data.sources import GriddedSource
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportTendModelEncProcDec
from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing.imputer import InputImputer
from anemoi.models.preprocessing.normalizer import InputNormalizer
from anemoi.models.preprocessing.postprocessor import NormalizedReluPostprocessor
from anemoi.models.preprocessing.remapper import Remapper
from tests.batch_builders import build_batch

GRIDDED_LAYOUT = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)


class IdentityProcessor(torch.nn.Module):
    def forward(self, x: torch.Tensor, in_place: bool = True, inverse: bool = False, **kwargs):
        del inverse, kwargs
        if not in_place:
            x = x.clone()
        return x


class DummyResidual(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        grid_shard_sizes=None,
        model_comm_group=None,
        n_step_output: int = 1,
    ) -> torch.Tensor:
        del model_comm_group, n_step_output
        assert grid_shard_sizes is None
        return x


def _make_index_collection() -> IndexCollection:
    data_config = DictConfig({"forcing": ["force"], "diagnostic": ["diag"], "target": []})
    name_to_index = {"prog0": 0, "prog1": 1, "force": 2, "diag": 3}
    return IndexCollection(data_config, name_to_index)


IMPUTER_STATISTICS = {
    "mean": np.array([1.0, 2.0, 3.0, 4.5, 3.0, 1.0]),
    "stdev": np.array([0.5, 0.5, 0.5, 1.0, 14.0, 1.0]),
    "minimum": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
    "maximum": np.array([11.0, 10.0, 10.0, 10.0, 10.0, 2.0]),
}


def _make_imputer_settings() -> tuple[InputImputer, IndexCollection]:
    config = DictConfig(
        {
            "diagnostics": {"log": {"code": {"level": "DEBUG"}}},
            "data": {
                "imputer": {
                    "default": "none",
                    "mean": ["y", "other"],
                    "maximum": ["x"],
                    "none": ["z"],
                    "minimum": ["q"],
                },
                "forcing": ["z", "q"],
                "diagnostic": ["other"],
            },
        },
    )
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4, "prog": 5}
    data_indices = IndexCollection(data_config=config.data, name_to_index=name_to_index)
    imputer = InputImputer(config=config.data.imputer, data_indices=data_indices)
    return imputer, data_indices


def _make_model() -> AnemoiTransportTendModelEncProcDec:
    model = AnemoiTransportTendModelEncProcDec.__new__(AnemoiTransportTendModelEncProcDec)
    model.data_indices = {"data": _make_index_collection()}
    model.target_datasets = ["data"]
    return model


def _gridded_batch(data: torch.Tensor, variables: list[str]):
    return build_batch(
        data={"data": data},
        coordinates={"data": torch.zeros(data.shape[-2], 2)},
        layouts={"data": GRIDDED_LAYOUT},
        variables={"data": variables},
        statistics={"data": {}},
    )


# Full variable order of the dataset: prognostic prog0/prog1, forcing force, diagnostic diag.
STATE_STATISTICS = {"mean": np.array([1.0, 2.0, 3.0, 4.0]), "stdev": np.array([2.0, 4.0, 1.0, 0.5])}
TENDENCY_STATISTICS = {
    "lead_times": ["6h", "12h"],
    "6h": {"mean": np.array([0.1, 0.2, 9.0, 9.0]), "stdev": np.array([0.5, 0.25, 9.0, 9.0])},
    "12h": {"mean": np.array([0.3, 0.4, 9.0, 9.0]), "stdev": np.array([1.0, 0.5, 9.0, 9.0])},
}


def _source(data: torch.Tensor, variables: list[str], statistics: dict = STATE_STATISTICS) -> GriddedSource:
    name_to_index = {"prog0": 0, "prog1": 1, "force": 2, "diag": 3}
    positions = [name_to_index[name] for name in variables]
    return GriddedSource(
        name="data",
        variables=variables,
        layout=GRIDDED_LAYOUT,
        data=data,
        coordinates=torch.zeros(data.shape[GRIDDED_LAYOUT.grid], 2),
        statistics={key: value[positions] for key, value in statistics.items()},
    )


def _normalizer_processors() -> tuple[Processors, Processors]:
    normalizer = InputNormalizer(config=DictConfig({"default": "mean-std"}))
    return Processors([["normalizer", normalizer]]), Processors([["normalizer", normalizer]], inverse=True)


def _tendency_model(n_step_output: int = 2) -> AnemoiTransportTendModelEncProcDec:
    model = _make_model()
    torch.nn.Module.__init__(model)
    model.n_step_output = {"data": n_step_output}
    return model


def test_tendency_statistics_follow_lead_times() -> None:
    model = _tendency_model()

    per_step = model.tendency_statistics("data", TENDENCY_STATISTICS)

    assert per_step == [TENDENCY_STATISTICS["6h"], TENDENCY_STATISTICS["12h"]]


@pytest.mark.parametrize(
    ("statistics", "message"),
    [
        (None, "Tendency statistics are required"),
        (STATE_STATISTICS, "per lead time"),
        ({"lead_times": ["6h"], "6h": STATE_STATISTICS}, "Expected 2 tendency statistics"),
        ({"lead_times": ["6h", "12h"], "6h": STATE_STATISTICS}, "Missing tendency statistics"),
    ],
)
def test_tendency_statistics_rejects_incomplete_statistics(statistics, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _tendency_model().tendency_statistics("data", statistics)


def test_compute_tendency_normalises_each_step_with_its_tendency_statistics() -> None:
    model = _tendency_model()
    pre_processors, post_processors = _normalizer_processors()
    per_step = model.tendency_statistics("data", TENDENCY_STATISTICS)

    state = _source(torch.randn(2, 2, 1, 3, 3), ["prog0", "prog1", "diag"])
    reference = _source(torch.randn(2, 1, 1, 3, 2), ["prog0", "prog1"])

    tendency = model.compute_tendency("data", state, reference, per_step, pre_processors, post_processors)

    def _prognostic(statistics: dict, key: str) -> torch.Tensor:
        return torch.as_tensor(statistics[key][:2], dtype=torch.float32)

    mean, stdev = _prognostic(STATE_STATISTICS, "mean"), _prognostic(STATE_STATISTICS, "stdev")
    physical_state = state.data[..., :2] * stdev + mean
    physical_reference = reference.data * stdev + mean
    for step, lead_time in enumerate(["6h", "12h"]):
        statistics = TENDENCY_STATISTICS[lead_time]
        expected = (physical_state[:, step] - physical_reference[:, 0] - _prognostic(statistics, "mean")) / _prognostic(
            statistics, "stdev"
        )
        torch.testing.assert_close(tendency.data[:, step, ..., :2], expected)
    # Diagnostics are predicted as states and keep their normalisation.
    torch.testing.assert_close(tendency.data[..., 2], state.data[..., 2])
    # The payload is normalised per lead time; the source keeps the state statistics of its variables.
    assert tendency.variables == state.variables
    assert all(np.array_equal(tendency.statistics[key], state.statistics[key]) for key in state.statistics)


def test_add_tendency_to_state_inverts_compute_tendency() -> None:
    model = _tendency_model()
    pre_processors, post_processors = _normalizer_processors()
    per_step = model.tendency_statistics("data", TENDENCY_STATISTICS)

    state = _source(torch.randn(2, 2, 1, 3, 3), ["prog0", "prog1", "diag"])
    reference = _source(torch.randn(2, 1, 1, 3, 2), ["prog0", "prog1"])
    tendency = model.compute_tendency("data", state, reference, per_step, pre_processors, post_processors)

    normalised = model.add_tendency_to_state(
        "data", reference, tendency, per_step, post_processors, pre_processors=pre_processors
    )
    physical = model.add_tendency_to_state("data", reference, tendency, per_step, post_processors)

    torch.testing.assert_close(normalised.data, state.data)
    torch.testing.assert_close(physical.data, post_processors(state, in_place=False).data)
    assert normalised.variables == state.variables


def test_tendency_roundtrip_skips_imputation() -> None:
    imputer, data_indices = _make_imputer_settings()
    model = AnemoiTransportTendModelEncProcDec.__new__(AnemoiTransportTendModelEncProcDec)
    torch.nn.Module.__init__(model)
    model.data_indices = {"data": data_indices}
    model.n_step_output = {"data": 1}

    statistics = IMPUTER_STATISTICS
    normalizer = InputNormalizer(config=DictConfig({"default": "mean-std"}))
    pre_processors = Processors([["imputer", imputer], ["normalizer", normalizer]])
    post_processors = Processors([["imputer", imputer], ["normalizer", normalizer]], inverse=True)

    output_names = list(data_indices.data.output.ordered_names)
    input_prognostic = [data_indices.name_to_index[name] for name in data_indices.prognostic]
    full_positions = [data_indices.name_to_index[name] for name in output_names]

    data = torch.randn(1, 1, 1, 2, len(output_names))
    data[0, 0, 0, 0, 0] = float("nan")
    state = GriddedSource(
        name="data",
        variables=output_names,
        layout=GRIDDED_LAYOUT,
        data=data,
        coordinates=torch.zeros(2, 2),
        statistics={key: value[full_positions] for key, value in statistics.items()},
    )
    reference = GriddedSource(
        name="data",
        variables=list(data_indices.prognostic),
        layout=GRIDDED_LAYOUT,
        data=torch.randn(1, 1, 1, 2, len(input_prognostic)),
        coordinates=torch.zeros(2, 2),
        statistics={key: value[input_prognostic] for key, value in statistics.items()},
    )
    tendency_statistics = [{key: value * 0.5 for key, value in statistics.items()}]

    tendency = model.compute_tendency("data", state, reference, tendency_statistics, pre_processors, post_processors)
    assert torch.isnan(tendency.data[0, 0, 0, 0, 0]), "Missing values must not be imputed."

    roundtrip = model.add_tendency_to_state(
        "data", reference, tendency, tendency_statistics, post_processors, pre_processors=pre_processors
    )
    torch.testing.assert_close(roundtrip.data, state.data, equal_nan=True)


def test_tendencies_of_imputed_and_missing_states_agree_where_data_exists() -> None:
    """Tendencies of imputed states match those of NaN-keeping states wherever the data is present."""
    imputer, data_indices = _make_imputer_settings()
    model = AnemoiTransportTendModelEncProcDec.__new__(AnemoiTransportTendModelEncProcDec)
    torch.nn.Module.__init__(model)
    model.data_indices = {"data": data_indices}
    model.n_step_output = {"data": 2}

    normalizer = InputNormalizer(config=DictConfig({"default": "mean-std"}))
    pre_processors = Processors([["imputer", imputer], ["normalizer", normalizer]])
    post_processors = Processors([["imputer", imputer], ["normalizer", normalizer]], inverse=True)

    output_names = list(data_indices.data.output.ordered_names)
    full_positions = [data_indices.name_to_index[name] for name in output_names]
    input_prognostic = [data_indices.name_to_index[name] for name in data_indices.prognostic]

    raw = torch.randn(1, 2, 1, 4, len(output_names))
    # Missing values in imputed prognostic variables (x, y) and in the imputed diagnostic (other).
    for step, node, name in [(0, 0, "x"), (1, 2, "y"), (0, 3, "other"), (1, 1, "x")]:
        raw[0, step, 0, node, output_names.index(name)] = float("nan")
    missing = torch.isnan(raw)
    raw_state = GriddedSource(
        name="data",
        variables=output_names,
        layout=GRIDDED_LAYOUT,
        data=raw,
        coordinates=torch.zeros(4, 2),
        statistics={key: value[full_positions] for key, value in IMPUTER_STATISTICS.items()},
    )
    imputed_state = pre_processors(raw_state, in_place=False)
    missing_state = pre_processors(raw_state, in_place=False, skip_imputation=True)

    reference = GriddedSource(
        name="data",
        variables=list(data_indices.prognostic),
        layout=GRIDDED_LAYOUT,
        data=torch.randn(1, 1, 1, 4, len(input_prognostic)),
        coordinates=torch.zeros(4, 2),
        statistics={key: value[input_prognostic] for key, value in IMPUTER_STATISTICS.items()},
    )
    tendency_statistics = [
        {key: value * 0.5 for key, value in IMPUTER_STATISTICS.items()},
        {key: value * 0.25 for key, value in IMPUTER_STATISTICS.items()},
    ]

    imputed = model.compute_tendency(
        "data", imputed_state, reference, tendency_statistics, pre_processors, post_processors
    ).data
    with_missing = model.compute_tendency(
        "data", missing_state, reference, tendency_statistics, pre_processors, post_processors
    ).data

    assert not torch.isnan(imputed).any()
    torch.testing.assert_close(torch.isnan(with_missing), missing)
    torch.testing.assert_close(with_missing[~missing], imputed[~missing])


def test_reference_state_is_latest_prognostic_input_with_its_statistics() -> None:
    model = _tendency_model(n_step_output=1)
    model.residual = torch.nn.ModuleDict({"data": DummyResidual()})
    inputs = _source(torch.randn(1, 2, 1, 3, 3), ["prog0", "prog1", "force"])

    reference = model.reference_state(Batch({"data": inputs}), grid_shard_sizes=None, model_comm_group=None)["data"]

    assert reference.variables == ["prog0", "prog1"]
    torch.testing.assert_close(reference.data, inputs.data[:, -1:, ..., :2])
    assert all(np.array_equal(reference.statistics[key], STATE_STATISTICS[key][:2]) for key in STATE_STATISTICS)


def test_apply_reference_state_truncation_without_shards() -> None:
    model = _make_model()
    torch.nn.Module.__init__(model)
    model.n_step_output = {"data": 1}
    model.residual = torch.nn.ModuleDict({"data": DummyResidual()})

    x = {"data": torch.arange(1 * 1 * 1 * 2 * 4, dtype=torch.float32).reshape(1, 1, 1, 2, 4)}
    out = model.apply_reference_state_truncation(x, grid_shard_sizes=None, model_comm_group=None)

    indices = model.data_indices["data"].model.input.prognostic
    expected = x["data"][..., indices]
    assert torch.allclose(out["data"], expected)


def test_sampled_to_physical_uses_latest_reference_and_per_step_statistics() -> None:
    model = _tendency_model()

    # Two different reference timesteps; training-style behavior should always use the last one.
    ref = torch.zeros((1, 2, 1, 2, 2), dtype=torch.float32)
    ref[:, 0] = 1.0
    ref[:, 1] = 2.0
    model.apply_reference_state_truncation = lambda *_args, **_kwargs: {"data": ref}

    captured = []

    def _spy_add_tendency(dataset_name, reference, tendency, tendency_statistics, *_args, **_kwargs):
        captured.append((dataset_name, reference.data.clone(), tendency_statistics))
        return tendency

    model.add_tendency_to_state = _spy_add_tendency

    sampled = _gridded_batch(torch.ones((1, 2, 1, 2, 3), dtype=torch.float32), ["prog0", "prog1", "diag"])
    x = _gridded_batch(torch.zeros((1, 2, 1, 2, 3), dtype=torch.float32), ["prog0", "prog1", "force"])

    model._sampled_to_physical(
        sampled,
        x,
        torch.nn.ModuleDict({"data": IdentityProcessor()}),
        model_comm_group=None,
        statistics_tendencies={"data": TENDENCY_STATISTICS},
    )

    assert len(captured) == 1
    dataset_name, reference, tendency_statistics = captured[0]
    assert dataset_name == "data"
    torch.testing.assert_close(reference, ref[:, -1:])
    assert tendency_statistics == [TENDENCY_STATISTICS["6h"], TENDENCY_STATISTICS["12h"]]


def test_sampled_to_physical_requires_tendency_statistics() -> None:
    model = _tendency_model()
    sampled = _gridded_batch(torch.ones((1, 2, 1, 2, 3)), ["prog0", "prog1", "diag"])

    with pytest.raises(ValueError, match="Tendency statistics must be provided"):
        model._sampled_to_physical(sampled, sampled, torch.nn.ModuleDict({"data": IdentityProcessor()}))


# One output step; prog0 is bounded below by 0 (physical units) by a NormalizedReluPostprocessor.
# prog0: state mean 1, stdev 2; 6h tendency mean 0.1, stdev 0.5.
ONE_STEP_TENDENCY_STATISTICS = {"lead_times": ["6h"], "6h": TENDENCY_STATISTICS["6h"]}


def _bounded_processors(model: AnemoiTransportTendModelEncProcDec) -> tuple[Processors, Processors]:
    normalizer = InputNormalizer(config=DictConfig({"default": "mean-std"}))
    relu = NormalizedReluPostprocessor(
        config=DictConfig({"default": "none", 0: ["prog0"], "normalizer": "mean-std"}),
        data_indices=model.data_indices["data"],
    )
    processors = [["normalizer", normalizer], ["relu", relu]]
    return Processors(processors), Processors(processors, inverse=True)


def _bounded_case() -> tuple[GriddedSource, GriddedSource]:
    """Normalised reference and tendency at two grid points, as (reference, tendency) of prog0.

    Grid point 0: reference 5, tendency -2 -> state 3 (inside the bound, tendency below it).
    Grid point 1: reference 1, tendency -3 -> state -2 (outside the bound).
    """
    reference = torch.zeros(1, 1, 1, 2, 2)
    reference[..., 0] = torch.tensor([(5.0 - 1.0) / 2.0, (1.0 - 1.0) / 2.0])
    tendency = torch.zeros(1, 1, 1, 2, 3)
    tendency[..., 0] = torch.tensor([(-2.0 - 0.1) / 0.5, (-3.0 - 0.1) / 0.5])
    return _source(reference, ["prog0", "prog1"]), _source(tendency, ["prog0", "prog1", "diag"])


def test_processors_skip_output_constraints() -> None:
    model = _tendency_model(n_step_output=1)
    _, post_processors = _bounded_processors(model)
    state = _source(torch.full((1, 1, 1, 1, 3), -3.0), ["prog0", "prog1", "diag"])

    assert post_processors.has_state_only_processors
    assert not _normalizer_processors()[1].has_state_only_processors
    clipped = post_processors(state, in_place=False).data[..., 0]
    unclipped = post_processors(state, in_place=False, skip_output_constraints=True).data[..., 0]
    torch.testing.assert_close(clipped, torch.zeros_like(clipped))
    torch.testing.assert_close(unclipped, torch.full_like(unclipped, -3.0 * 2.0 + 1.0))


def test_add_tendency_to_state_does_not_constrain_the_tendency() -> None:
    model = _tendency_model(n_step_output=1)
    pre_processors, post_processors = _bounded_processors(model)
    per_step = model.tendency_statistics("data", ONE_STEP_TENDENCY_STATISTICS)
    reference, tendency = _bounded_case()

    physical = model.add_tendency_to_state("data", reference, tendency, per_step, post_processors)
    normalised = model.add_tendency_to_state(
        "data", reference, tendency, per_step, post_processors, pre_processors=pre_processors
    )

    # Clipping the tendency at 0 would give 5 and 1. The state is rebuilt unconstrained ...
    torch.testing.assert_close(physical.data[0, 0, 0, :, 0], torch.tensor([3.0, -2.0]))
    torch.testing.assert_close(normalised.data[0, 0, 0, :, 0], torch.tensor([1.0, -1.5]))
    # ... and the constraint applies when the state is post-processed, like a state prediction.
    torch.testing.assert_close(
        post_processors(normalised, in_place=False).data[0, 0, 0, :, 0], torch.tensor([3.0, 0.0])
    )


def test_compute_tendency_does_not_constrain_target_states() -> None:
    """A target state outside the bound keeps its value, so the tendency roundtrip is exact."""
    model = _tendency_model(n_step_output=1)
    pre_processors, post_processors = _bounded_processors(model)
    per_step = model.tendency_statistics("data", ONE_STEP_TENDENCY_STATISTICS)
    reference, _ = _bounded_case()
    state = torch.zeros(1, 1, 1, 2, 3)
    state[..., 0] = torch.tensor([(3.0 - 1.0) / 2.0, (-2.0 - 1.0) / 2.0])  # 3 and -2 in physical units
    state = _source(state, ["prog0", "prog1", "diag"])

    tendency = model.compute_tendency("data", state, reference, per_step, pre_processors, post_processors)
    rebuilt = model.add_tendency_to_state(
        "data", reference, tendency, per_step, post_processors, pre_processors=pre_processors
    )

    torch.testing.assert_close(tendency.data[0, 0, 0, :, 0], torch.tensor([(-2.0 - 0.1) / 0.5, (-3.0 - 0.1) / 0.5]))
    torch.testing.assert_close(rebuilt.data, state.data)


def _bounded_sampling_model() -> tuple[AnemoiTransportTendModelEncProcDec, Batch, Batch]:
    model = _tendency_model(n_step_output=1)
    reference, tendency = _bounded_case()
    model.apply_reference_state_truncation = lambda *_args, **_kwargs: {"data": reference.data}
    x = Batch({"data": _source(torch.zeros(1, 1, 1, 2, 3), ["prog0", "prog1", "force"])})
    return model, x, Batch({"data": tendency})


def test_sampled_to_physical_constrains_states_not_tendencies() -> None:
    model, x, sampled = _bounded_sampling_model()
    pre_processors, post_processors = _bounded_processors(model)

    out = model._sampled_to_physical(
        sampled,
        x,
        torch.nn.ModuleDict({"data": post_processors}),
        statistics_tendencies={"data": ONE_STEP_TENDENCY_STATISTICS},
        pre_processors=torch.nn.ModuleDict({"data": pre_processors}),
    )

    torch.testing.assert_close(out["data"].data[0, 0, 0, :, 0], torch.tensor([3.0, 0.0]))


def test_sampled_to_physical_needs_pre_processors_for_output_constraints() -> None:
    model, x, sampled = _bounded_sampling_model()
    _, post_processors = _bounded_processors(model)

    with pytest.raises(ValueError, match="needs the pre-processors"):
        model._sampled_to_physical(
            sampled,
            x,
            torch.nn.ModuleDict({"data": post_processors}),
            statistics_tendencies={"data": ONE_STEP_TENDENCY_STATISTICS},
        )


def _remapped_processors(model: AnemoiTransportTendModelEncProcDec) -> tuple[Processors, Processors]:
    """Normaliser followed by a ``sqrt`` remapping of prog1 (prognostic) and diag (diagnostic)."""
    normalizer = InputNormalizer(config=DictConfig({"default": "mean-std"}))
    remapper = Remapper(config=DictConfig({"sqrt": ["prog1", "diag"]}), data_indices=model.data_indices["data"])
    relu = NormalizedReluPostprocessor(
        config=DictConfig({"default": "none", 0: ["prog0"], "normalizer": "mean-std"}),
        data_indices=model.data_indices["data"],
    )
    processors = [["normalizer", normalizer], ["remapper", remapper], ["relu", relu]]
    return Processors(processors), Processors(processors, inverse=True)


def test_processors_skip_state_only_processors_on_tendencies() -> None:
    model = _tendency_model(n_step_output=1)
    pre_processors, post_processors = _remapped_processors(model)
    tendency = _source(torch.full((1, 1, 1, 1, 3), -3.0), ["prog0", "prog1", "diag"])

    physical = post_processors(tendency, in_place=False, tendency=True).data[0, 0, 0, 0]
    # Only denormalised: no ReLU on prog0, no squaring of prog1 and diag.
    torch.testing.assert_close(physical, torch.tensor([-3.0 * 2.0 + 1.0, -3.0 * 4.0 + 2.0, -3.0 * 0.5 + 4.0]))
    # sqrt would reject the negative values.
    normalised = pre_processors(tendency.clone(data=physical.reshape(1, 1, 1, 1, 3)), in_place=False, tendency=True)
    torch.testing.assert_close(normalised.data, tendency.data)


def test_tendency_conversion_skips_remapping() -> None:
    """Tendencies are raw differences normalised with tendency statistics; diagnostics keep their remapping."""
    torch.manual_seed(0)
    model = _tendency_model()
    pre_processors, post_processors = _remapped_processors(model)
    per_step = model.tendency_statistics("data", TENDENCY_STATISTICS)
    # Normalised (remapped) states in the domain of the sqrt remapping.
    state = _source(torch.rand(2, 2, 1, 3, 3) * 2.0, ["prog0", "prog1", "diag"])
    reference = _source(torch.rand(2, 1, 1, 3, 2) * 2.0, ["prog0", "prog1"])

    tendency = model.compute_tendency("data", state, reference, per_step, pre_processors, post_processors)

    physical_state = post_processors(state, in_place=False, skip_output_constraints=True).data[..., :2]
    physical_reference = post_processors(reference, in_place=False, skip_output_constraints=True).data
    difference = physical_state - physical_reference
    assert (difference[..., 1] < 0).any(), "the test needs negative tendencies of the remapped variable"
    for step, lead_time in enumerate(["6h", "12h"]):
        statistics = {
            key: torch.as_tensor(value[:2], dtype=torch.float32)
            for key, value in TENDENCY_STATISTICS[lead_time].items()
        }
        expected = (difference[:, step] - statistics["mean"]) / statistics["stdev"]
        torch.testing.assert_close(tendency.data[:, step, ..., :2], expected)
    # The diagnostic is predicted as a state and keeps its normalised, remapped value.
    torch.testing.assert_close(tendency.data[..., 2], state.data[..., 2], rtol=0, atol=0)

    normalised = model.add_tendency_to_state(
        "data", reference, tendency, per_step, post_processors, pre_processors=pre_processors
    )
    physical = model.add_tendency_to_state("data", reference, tendency, per_step, post_processors)
    torch.testing.assert_close(normalised.data, state.data)
    torch.testing.assert_close(physical.data, post_processors(state, in_place=False, skip_output_constraints=True).data)


def test_states_rebuilt_from_tendencies_are_clipped_to_remapping_domains() -> None:
    """A rebuilt state outside the domain of a remapping is clipped to it when normalised again."""
    model = _tendency_model(n_step_output=1)
    pre_processors, post_processors = _remapped_processors(model)
    per_step = model.tendency_statistics("data", ONE_STEP_TENDENCY_STATISTICS)
    # prog1 (state mean 2, stdev 4; remapped with sqrt after normalisation): reference 6 -> normalised 1.
    reference = torch.zeros(1, 1, 1, 2, 2)
    reference[..., 1] = 1.0
    reference = _source(reference, ["prog0", "prog1"])
    # 6h tendency mean 0.2, stdev 0.25: +2 -> state 8 (in the domain), -8 -> state -2 (below the mean).
    tendency = torch.zeros(1, 1, 1, 2, 3)
    tendency[..., 1] = torch.tensor([(2.0 - 0.2) / 0.25, (-8.0 - 0.2) / 0.25])
    tendency = _source(tendency, ["prog0", "prog1", "diag"])

    physical = model.add_tendency_to_state("data", reference, tendency, per_step, post_processors)
    normalised = model.add_tendency_to_state(
        "data", reference, tendency, per_step, post_processors, pre_processors=pre_processors
    )

    torch.testing.assert_close(physical.data[0, 0, 0, :, 1], torch.tensor([8.0, -2.0]))
    torch.testing.assert_close(normalised.data[0, 0, 0, :, 1], torch.tensor([1.5**0.5, 0.0]))

    # Inference post-processes the clipped state, as a state predicted in remapped space would be.
    model.apply_reference_state_truncation = lambda *_args, **_kwargs: {"data": reference.data}
    x = Batch({"data": _source(torch.zeros(1, 1, 1, 2, 3), ["prog0", "prog1", "force"])})
    out = model._sampled_to_physical(
        Batch({"data": tendency}),
        x,
        torch.nn.ModuleDict({"data": post_processors}),
        statistics_tendencies={"data": ONE_STEP_TENDENCY_STATISTICS},
        pre_processors=torch.nn.ModuleDict({"data": pre_processors}),
    )
    torch.testing.assert_close(out["data"].data[0, 0, 0, :, 1], torch.tensor([8.0, 2.0]))

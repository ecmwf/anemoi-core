# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.models.diffusion_encoder_processor_decoder import AnemoiDiffusionTendModelEncProcDec
from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing.imputer import InputImputer


def _idx_list(idx) -> list[int] | None:
    if idx is None:
        return None
    if torch.is_tensor(idx):
        return idx.tolist()
    return list(idx)


class SequenceProcessor(torch.nn.Module):
    def __init__(self, offset: float, expected_indices: list[object]) -> None:
        super().__init__()
        self.offset = offset
        self.expected_indices = [_idx_list(idx) for idx in expected_indices]
        self.calls = 0

    def forward(self, x: torch.Tensor, in_place: bool = True, inverse: bool = False, data_index=None, **kwargs):
        del kwargs, inverse
        expected = self.expected_indices[self.calls]
        assert _idx_list(data_index) == expected
        self.calls += 1
        if not in_place:
            x = x.clone()
        return x + self.offset


class IdentityProcessor(torch.nn.Module):
    def forward(self, x: torch.Tensor, in_place: bool = True, inverse: bool = False, **kwargs):
        del inverse, kwargs
        if not in_place:
            x = x.clone()
        return x


def _make_index_collection() -> IndexCollection:
    data_config = DictConfig({"forcing": ["force"], "diagnostic": ["diag"], "target": []})
    name_to_index = {"prog0": 0, "prog1": 1, "force": 2, "diag": 3}
    return IndexCollection(data_config, name_to_index)


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
    statistics = {
        "mean": np.array([1.0, 2.0, 3.0, 4.5, 3.0, 1.0]),
        "stdev": np.array([0.5, 0.5, 0.5, 1.0, 14.0, 1.0]),
        "minimum": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
        "maximum": np.array([11.0, 10.0, 10.0, 10.0, 10.0, 2.0]),
    }
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4, "prog": 5}
    data_indices = IndexCollection(data_config=config.data, name_to_index=name_to_index)
    imputer = InputImputer(config=config.data.imputer, data_indices=data_indices, statistics=statistics)
    return imputer, data_indices


def _make_model() -> AnemoiDiffusionTendModelEncProcDec:
    model = AnemoiDiffusionTendModelEncProcDec.__new__(AnemoiDiffusionTendModelEncProcDec)
    model.data_indices = {"data": _make_index_collection()}
    return model


def test_compute_tendency_uses_expected_indices() -> None:
    model = _make_model()
    indices = model.data_indices["data"]

    x_t1 = torch.tensor([[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]])
    x_t0 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    input_post = SequenceProcessor(
        0.0,
        [indices.data.output.full, indices.data.input.prognostic],
    )
    state_proc = SequenceProcessor(100.0, [indices.data.output.diagnostic])
    tend_proc = SequenceProcessor(10.0, [indices.data.output.prognostic])

    out = model.compute_tendency(
        {"data": x_t1},
        {"data": x_t0},
        {"data": state_proc},
        {"data": tend_proc},
        {"data": input_post},
        skip_imputation=True,
    )

    assert input_post.calls == 2
    assert state_proc.calls == 1
    assert tend_proc.calls == 1

    expected = x_t1.clone()
    expected[..., indices.model.output.prognostic] = (x_t1[..., indices.model.output.prognostic] - x_t0) + 10.0
    expected[..., indices.model.output.diagnostic] = x_t1[..., indices.model.output.diagnostic] + 100.0

    assert torch.allclose(out["data"], expected)


def test_add_tendency_to_state_uses_expected_indices() -> None:
    model = _make_model()
    indices = model.data_indices["data"]

    tendency = torch.tensor([[[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]]])
    state_inp = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])

    post_tend = SequenceProcessor(1.0, [indices.data.output.full])
    post_state = SequenceProcessor(10.0, [indices.data.output.diagnostic, indices.data.input.prognostic])

    out = model.add_tendency_to_state(
        {"data": state_inp},
        {"data": tendency},
        {"data": post_state},
        {"data": post_tend},
        {"data": None},
        skip_imputation=True,
    )

    expected = tendency + 1.0
    expected[..., indices.model.output.diagnostic] = tendency[..., indices.model.output.diagnostic] + 10.0
    expected[..., indices.model.output.prognostic] += state_inp + 10.0

    assert post_tend.calls == 1
    assert post_state.calls == 2
    assert torch.allclose(out["data"], expected)


def test_tendency_roundtrip_skips_imputation() -> None:
    imputer, data_indices = _make_imputer_settings()
    model = AnemoiDiffusionTendModelEncProcDec.__new__(AnemoiDiffusionTendModelEncProcDec)
    model.data_indices = {"data": data_indices}
    indices = data_indices

    x_full = torch.tensor(
        [
            [
                [
                    [float("nan"), 1.0, 2.0, 3.0, 4.0, 5.0],
                    [6.0, float("nan"), 8.0, 9.0, 10.0, 11.0],
                ]
            ]
        ]
    )
    imputer.transform(x_full, in_place=False)

    input_post = Processors([["imputer", imputer]], inverse=True)
    identity = IdentityProcessor()

    x_t1 = torch.tensor([[[[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]]]])
    x_t0 = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]])

    tendency = model.compute_tendency(
        {"data": x_t1},
        {"data": x_t0},
        {"data": identity},
        {"data": identity},
        {"data": input_post},
        skip_imputation=True,
    )["data"]

    expected_tendency = x_t1.clone()
    expected_tendency[..., indices.model.output.prognostic] = x_t1[..., indices.model.output.prognostic] - x_t0
    expected_tendency[..., indices.model.output.diagnostic] = x_t1[..., indices.model.output.diagnostic]

    assert torch.allclose(tendency, expected_tendency, equal_nan=True)

    state = model.add_tendency_to_state(
        {"data": x_t0},
        {"data": tendency},
        {"data": identity},
        {"data": identity},
        {"data": None},
        skip_imputation=True,
    )["data"]

    assert torch.allclose(state, x_t1, equal_nan=True)


def test_apply_imputer_inverse_reinserts_nans() -> None:
    imputer, data_indices = _make_imputer_settings()

    x_full = torch.tensor(
        [
            [
                [
                    [float("nan"), 1.0, 2.0, 3.0, 4.0, 5.0],
                    [6.0, float("nan"), 8.0, 9.0, 10.0, 11.0],
                ]
            ]
        ]
    )
    imputer.transform(x_full, in_place=False)

    post_processors = torch.nn.ModuleDict({"data": Processors([["imputer", imputer]], inverse=True)})

    out = torch.ones((1, 1, 2, len(data_indices.data.output.full)), dtype=torch.float32)
    expected = imputer.inverse_transform(out, in_place=False)

    model = AnemoiDiffusionTendModelEncProcDec.__new__(AnemoiDiffusionTendModelEncProcDec)
    result = model._apply_imputer_inverse(post_processors, "data", out)

    assert torch.allclose(result, expected, equal_nan=True)


def test_after_sampling_reinserts_nans() -> None:
    imputer, data_indices = _make_imputer_settings()

    x_full = torch.tensor(
        [
            [
                [
                    [float("nan"), 1.0, 2.0, 3.0, 4.0, 5.0],
                    [6.0, float("nan"), 8.0, 9.0, 10.0, 11.0],
                ]
            ]
        ]
    )
    imputer.transform(x_full, in_place=False)

    post_processors = torch.nn.ModuleDict({"data": Processors([["imputer", imputer]], inverse=True)})

    model = AnemoiDiffusionTendModelEncProcDec.__new__(AnemoiDiffusionTendModelEncProcDec)
    model.n_step_output = 1

    def _identity_ref(x, *_args, **_kwargs):
        return x

    def _passthrough_add_tendency(_state_inp, tendency, *_args, **_kwargs):
        return tendency

    model.apply_reference_state_truncation = _identity_ref
    model.add_tendency_to_state = _passthrough_add_tendency

    out = {"data": torch.ones((1, 1, 1, 2, len(data_indices.data.output.full)), dtype=torch.float32)}
    before_sampling_data = ({}, {"data": torch.zeros((1, 1, 1, 2, 1), dtype=torch.float32)})

    result = model._after_sampling(
        out,
        post_processors,
        before_sampling_data,
        model_comm_group=None,
        grid_shard_shapes={"data": None},
        gather_out=False,
        post_processors_tendencies={"data": Processors([["imputer", imputer]], inverse=True)},
    )["data"]

    expected = imputer.inverse_transform(out["data"], in_place=False)

    assert torch.allclose(result, expected, equal_nan=True)


# ============================================================
# Helpers: SimpleNamespace-based indices + real InputNormalizer
# ============================================================

from types import SimpleNamespace


def _make_dummy_indices(
    full: list[int] | None = None,
    prognostic: list[int] | None = None,
    diagnostic: list[int] | None = None,
) -> SimpleNamespace:
    full = [0, 1, 2] if full is None else full
    prognostic = full if prognostic is None else prognostic
    diagnostic = [] if diagnostic is None else diagnostic
    return SimpleNamespace(
        data=SimpleNamespace(
            output=SimpleNamespace(full=full, prognostic=prognostic, diagnostic=diagnostic),
            input=SimpleNamespace(prognostic=prognostic),
        ),
        model=SimpleNamespace(
            output=SimpleNamespace(prognostic=prognostic, diagnostic=diagnostic),
            input=SimpleNamespace(prognostic=prognostic),
        ),
    )


def _make_tendency_model(indices: SimpleNamespace | None = None) -> AnemoiDiffusionTendModelEncProcDec:
    model = AnemoiDiffusionTendModelEncProcDec.__new__(AnemoiDiffusionTendModelEncProcDec)
    model.data_indices = {"data": _make_dummy_indices() if indices is None else indices}
    return model


def _make_normalizer_data_indices(nvars: int) -> SimpleNamespace:
    names = [f"var{i}" for i in range(nvars)]
    full = torch.arange(nvars, dtype=torch.long)
    name_to_index = {name: i for i, name in enumerate(names)}
    return SimpleNamespace(
        data=SimpleNamespace(
            input=SimpleNamespace(name_to_index=name_to_index, full=full),
            output=SimpleNamespace(name_to_index=name_to_index, full=full),
        ),
        model=SimpleNamespace(
            input=SimpleNamespace(name_to_index=name_to_index),
            output=SimpleNamespace(name_to_index=name_to_index),
        ),
    )


def _make_input_normalizer(mean: list[float], stdev: list[float]):
    from anemoi.models.preprocessing.normalizer import InputNormalizer

    mean_arr = np.asarray(mean, dtype=np.float32)
    stdev_arr = np.asarray(stdev, dtype=np.float32)
    stats = {
        "minimum": mean_arr - 100.0,
        "maximum": mean_arr + 100.0,
        "mean": mean_arr.copy(),
        "stdev": stdev_arr.copy(),
    }
    return InputNormalizer(
        config={"default": "mean-std"},
        data_indices=_make_normalizer_data_indices(nvars=len(mean)),
        statistics=stats,
    )


# ============================================================
# Round-trip: compute_tendency -> add_tendency_to_state recovers x_t1
# ============================================================


def test_tendency_roundtrip_identity_processors() -> None:
    """compute_tendency then add_tendency_to_state with identity processors recovers x_t1."""
    model = _make_model()
    identity = IdentityProcessor()

    x_t1 = torch.tensor([[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]])  # 2 prog + 1 diag
    x_t0 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])                   # prognostic only

    tendency = model.compute_tendency(
        {"data": x_t1}, {"data": x_t0},
        {"data": identity}, {"data": identity}, {"data": None},
        skip_imputation=True,
    )["data"]

    x_t1_reconstructed = model.add_tendency_to_state(
        {"data": x_t0}, {"data": tendency},
        {"data": identity}, {"data": identity}, {"data": None},
        skip_imputation=True,
    )["data"]

    assert torch.allclose(x_t1_reconstructed, x_t1, atol=1e-5)




# ============================================================
# Cross-validation: tendency vs downscaler residual computation
# Shows that state stats vs residual stats produce different normalizations,
# and that matching stats (residual for prog, state for diag) makes them equivalent.
# ============================================================


def test_tendency_vs_downscaler_residual_normalization() -> None:
    """Tendency and downscaler residual computations agree when using matching stats.

    The downscaler computes: normalize(y - x_interp) for prognostic channels.
    The tendency model computes: tendency_norm(x_t1_prog - x_t0) for prognostic channels.
    These are equivalent when:
      - x_t0 == x_interp (upsampled lres, prognostic channels)
      - tendency_norm uses residual stats (not state stats)
    Also asserts that using state stats (wrong) gives a different result.
    """
    indices = _make_dummy_indices(full=[0, 1, 2], prognostic=[0, 1], diagnostic=[2])
    model = _make_tendency_model(indices)

    x_t1 = torch.tensor([[[[8.0, 17.0, 31.0], [10.0, 19.0, 29.0]]]])   # out_hres (raw)
    x_t0_prog = torch.tensor([[[[5.0, 14.0], [6.0, 13.0]]]])            # x_interp prognostic channels

    state_norm = _make_input_normalizer(mean=[4.0, 10.0, 25.0], stdev=[2.0, 4.0, 5.0])
    resid_norm = _make_input_normalizer(mean=[1.0, 3.0, 0.0], stdev=[0.5, 1.5, 1.0])

    tendency = model.compute_tendency(
        {"data": x_t1}, {"data": x_t0_prog},
        {"data": state_norm}, {"data": resid_norm}, {"data": None},
        skip_imputation=True,
    )["data"]

    # Downscaler path: normalize (y - x_interp) with residual stats for prog,
    # and y with state stats for diagnostic — should match tendency output exactly
    from anemoi.models.preprocessing.normalizer import InputNormalizer
    resid_prog = x_t1[..., [0, 1]] - x_t0_prog                          # (y - x_interp) for prog
    resid_prog_norm = resid_norm(resid_prog, in_place=False,
                                 data_index=torch.tensor([0, 1]))
    diag_norm = state_norm(x_t1[..., [2]], in_place=False,
                           data_index=torch.tensor([2]))
    downscaler_target = torch.cat([resid_prog_norm, diag_norm], dim=-1)

    torch.testing.assert_close(tendency, downscaler_target, atol=1e-4, rtol=1e-4)

    # Sanity check: using state stats for prog normalization gives a DIFFERENT result
    tendency_wrong = model.compute_tendency(
        {"data": x_t1}, {"data": x_t0_prog},
        {"data": state_norm}, {"data": state_norm}, {"data": None},
        skip_imputation=True,
    )["data"]
    assert not torch.allclose(tendency_wrong[..., [0, 1]], tendency[..., [0, 1]], atol=1e-4), (
        "Expected state vs residual stats to produce different normalized tendencies"
    )


# ============================================================
# Edge case: all-prognostic channels
# ============================================================


def test_compute_tendency_all_prognostic() -> None:
    """When all channels are prognostic, tendency = normalize(x_t1 - x_t0)."""
    indices = _make_dummy_indices(full=[0, 1, 2], prognostic=[0, 1, 2], diagnostic=[])
    model = _make_tendency_model(indices)
    identity = IdentityProcessor()

    x_t1 = torch.tensor([[[10.0, 20.0, 30.0]]])
    x_t0 = torch.tensor([[[1.0, 2.0, 3.0]]])

    tendency = model.compute_tendency(
        {"data": x_t1}, {"data": x_t0},
        {"data": identity}, {"data": identity}, {"data": None},
        skip_imputation=True,
    )["data"]

    assert torch.allclose(tendency, x_t1 - x_t0, atol=1e-5)

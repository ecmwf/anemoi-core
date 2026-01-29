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
    )

    expected = tendency + 1.0
    expected[..., indices.model.output.diagnostic] = tendency[..., indices.model.output.diagnostic] + 10.0
    expected[..., indices.model.output.prognostic] += state_inp + 10.0

    assert post_tend.calls == 1
    assert post_state.calls == 2
    assert torch.allclose(out["data"], expected)


def test_tendency_roundtrip_with_imputer_subset() -> None:
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
    )["data"]

    full_input_list = indices.data.input.full.tolist()
    full_input_map = {full_idx: pos for pos, full_idx in enumerate(full_input_list)}
    prog_index_list = indices.data.input.prognostic.tolist()

    x_t0_with_nans = x_t0.clone()
    for idx_dst, idx_src in enumerate(prog_index_list):
        idx_src_pos = full_input_map[idx_src]
        nan_mask = imputer.nan_locations[..., idx_src_pos]
        for _ in x_t0.shape[1:-2]:
            nan_mask = nan_mask.unsqueeze(1)
        nan_mask = nan_mask.expand(-1, *x_t0.shape[1:-2], -1)
        x_t0_with_nans[..., idx_dst][nan_mask] = torch.nan

    expected_tendency = x_t1.clone()
    expected_tendency[..., indices.model.output.prognostic] = (
        x_t1[..., indices.model.output.prognostic] - x_t0_with_nans
    )
    expected_tendency[..., indices.model.output.diagnostic] = x_t1[..., indices.model.output.diagnostic]

    assert torch.allclose(tendency, expected_tendency, equal_nan=True)

    state = model.add_tendency_to_state(
        {"data": x_t0},
        {"data": tendency},
        {"data": identity},
        {"data": identity},
        {"data": None},
    )["data"]

    expected_x_t1 = input_post(x_t1, in_place=False, data_index=indices.data.output.full)

    assert torch.allclose(state, expected_x_t1, equal_nan=True)

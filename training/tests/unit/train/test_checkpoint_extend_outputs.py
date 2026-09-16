# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for training.transfer_learning_extend_outputs.

The setting exists for the fine-scale arm RUP (2026-09-16), which warm-starts a model that
predicts 68 variables into a model that predicts 70 (the same 68 plus tp and cp). Every tensor
whose leading dimension counts output variables therefore grows, and the old code dropped it and
re-initialised it at random, which silently destroyed the warm start. The tests below build a
deliberately tiny stand-in for that situation: a linear head with 68 rows and a 68-long buffer
that plays the part of the normaliser per-variable statistics.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Never

import pytest
import torch

from anemoi.training.train.tasks.base import BaseGraphModule
from anemoi.training.utils.checkpoint import transfer_learning_loading

N_IN = 5
N_OLD = 68
N_NEW = 70


class DummyHeadModel(torch.nn.Module):
    """A linear output head plus a per-output-variable buffer, and nothing else."""

    def __init__(self, n_out: int, n_in: int = N_IN, fill: float = 0.0, arange: bool = False) -> None:
        super().__init__()
        self.head = torch.nn.Linear(n_in, n_out)
        self.register_buffer("_norm_mul", torch.empty(n_out, dtype=torch.float32))
        with torch.no_grad():
            if arange:
                # Distinct, recognisable values so a row can be traced back to its index.
                rows = torch.arange(n_out, dtype=torch.float32).unsqueeze(1)
                cols = torch.arange(n_in, dtype=torch.float32).unsqueeze(0)
                self.head.weight.copy_(rows * 10.0 + cols)
                self.head.bias.copy_(torch.arange(n_out, dtype=torch.float32) + 100.0)
                self._norm_mul.copy_(torch.arange(n_out, dtype=torch.float32) + 1000.0)
            else:
                self.head.weight.fill_(fill)
                self.head.bias.fill_(fill)
                self._norm_mul.fill_(fill)


class DummyGraphModule(BaseGraphModule):
    """A LightningModule shell: transfer_learning_loading needs .device and a state dict."""

    task_type = "forecaster"

    def __init__(self) -> None:
        pass

    def _step(self, batch, validation_mode: bool = False) -> Never:  # noqa: ANN001
        raise NotImplementedError


def _make_module(model: torch.nn.Module) -> DummyGraphModule:
    module = DummyGraphModule.__new__(DummyGraphModule)
    torch.nn.Module.__init__(module)
    module.model = model
    module._device = torch.device("cpu")
    # states False is the default and is what arm RUP runs with: the processor buffers are NOT
    # rebuilt from the dataset, so they reach the shape-mismatch loop below.
    module.config = SimpleNamespace(
        training=SimpleNamespace(update_ds_stats_on_ckpt_load=SimpleNamespace(states=False, tendencies=False)),
    )
    return module


def _save_donor(tmp_path: Path, donor: torch.nn.Module, name: str = "donor.ckpt") -> Path:
    checkpoint = {
        "state_dict": _make_module(donor).state_dict(),
        "hyper_parameters": {
            "config": SimpleNamespace(model=SimpleNamespace(processor=SimpleNamespace(num_layers=1, num_chunks=1))),
            "data_indices": SimpleNamespace(name_to_index={}),
        },
    }
    path = tmp_path / name
    torch.save(checkpoint, path)
    return path


def test_extend_outputs_on_copies_rows_zeroes_weight_tail_and_keeps_buffer_tail(tmp_path: Path) -> None:
    donor = DummyHeadModel(N_OLD, arange=True)
    ckpt_path = _save_donor(tmp_path, donor)

    model = DummyHeadModel(N_NEW, fill=-7.0)
    module = _make_module(model)
    fresh = {k: v.clone() for k, v in module.state_dict().items()}

    transfer_learning_loading(module, ckpt_path, extend_output_rows=True)

    loaded = module.state_dict()
    donor_state = _make_module(donor).state_dict()

    # The 68 old output channels keep exactly the donor values.
    assert torch.equal(loaded["model.head.weight"][:N_OLD], donor_state["model.head.weight"])
    assert torch.equal(loaded["model.head.bias"][:N_OLD], donor_state["model.head.bias"])
    assert torch.equal(loaded["model._norm_mul"][:N_OLD], donor_state["model._norm_mul"])

    # A parameter named *.weight or *.bias starts its new channels at zero.
    assert torch.equal(loaded["model.head.weight"][N_OLD:], torch.zeros(N_NEW - N_OLD, N_IN))
    assert torch.equal(loaded["model.head.bias"][N_OLD:], torch.zeros(N_NEW - N_OLD))

    # A buffer keeps the freshly built model values in the tail: that is how tp and cp pick up
    # the dataset statistics while the 68 old variables keep the checkpoint ones.
    assert torch.equal(loaded["model._norm_mul"][N_OLD:], fresh["model._norm_mul"][N_OLD:])
    assert torch.equal(loaded["model._norm_mul"][N_OLD:], torch.full((N_NEW - N_OLD,), -7.0))


def test_extend_outputs_off_leaves_every_grown_tensor_re_initialised(tmp_path: Path) -> None:
    donor = DummyHeadModel(N_OLD, arange=True)
    ckpt_path = _save_donor(tmp_path, donor)

    model = DummyHeadModel(N_NEW, fill=-7.0)
    module = _make_module(model)
    fresh = {k: v.clone() for k, v in module.state_dict().items()}

    transfer_learning_loading(module, ckpt_path)

    loaded = module.state_dict()
    # Default behaviour, unchanged: every grown tensor is dropped, so the model keeps the values
    # it was built with and nothing of the donor survives.
    for key in ("model.head.weight", "model.head.bias", "model._norm_mul"):
        assert torch.equal(loaded[key], fresh[key]), key


def test_extend_outputs_leaves_unchanged_shapes_alone(tmp_path: Path) -> None:
    donor = DummyHeadModel(N_OLD, arange=True)
    ckpt_path = _save_donor(tmp_path, donor)

    model = DummyHeadModel(N_OLD, fill=-7.0)
    module = _make_module(model)

    transfer_learning_loading(module, ckpt_path, extend_output_rows=True)

    loaded = module.state_dict()
    donor_state = _make_module(donor).state_dict()
    for key in ("model.head.weight", "model.head.bias", "model._norm_mul"):
        assert torch.equal(loaded[key], donor_state[key]), key


def test_both_dimensions_grow_needs_both_flags(tmp_path: Path) -> None:
    donor = DummyHeadModel(N_OLD, n_in=N_IN, arange=True)
    ckpt_path = _save_donor(tmp_path, donor)

    module = _make_module(DummyHeadModel(N_NEW, n_in=N_IN + 2, fill=-7.0))
    with pytest.raises(ValueError, match="grew in BOTH dimensions"):
        transfer_learning_loading(module, ckpt_path, extend_output_rows=True)


def test_both_dimensions_grow_extends_columns_then_rows(tmp_path: Path) -> None:
    donor = DummyHeadModel(N_OLD, n_in=N_IN, arange=True)
    ckpt_path = _save_donor(tmp_path, donor)

    module = _make_module(DummyHeadModel(N_NEW, n_in=N_IN + 2, fill=-7.0))
    transfer_learning_loading(module, ckpt_path, extend_input_columns=True, extend_output_rows=True)

    weight = module.state_dict()["model.head.weight"]
    donor_weight = _make_module(donor).state_dict()["model.head.weight"]
    assert weight.shape == (N_NEW, N_IN + 2)
    assert torch.equal(weight[:N_OLD, :N_IN], donor_weight)
    assert torch.equal(weight[:N_OLD, N_IN:], torch.zeros(N_OLD, 2))
    assert torch.equal(weight[N_OLD:], torch.zeros(N_NEW - N_OLD, N_IN + 2))


def test_extend_inputs_branch_is_untouched_by_the_new_flag(tmp_path: Path) -> None:
    donor = DummyHeadModel(N_OLD, n_in=N_IN, arange=True)
    ckpt_path = _save_donor(tmp_path, donor)

    results = []
    for extend_outputs in (False, True):
        module = _make_module(DummyHeadModel(N_OLD, n_in=N_IN + 2, fill=-7.0))
        transfer_learning_loading(
            module, ckpt_path, extend_input_columns=True, extend_output_rows=extend_outputs
        )
        results.append(module.state_dict()["model.head.weight"].clone())

    donor_weight = _make_module(donor).state_dict()["model.head.weight"]
    assert torch.equal(results[0], results[1])
    assert torch.equal(results[0][:, :N_IN], donor_weight)
    assert torch.equal(results[0][:, N_IN:], torch.zeros(N_OLD, 2))


def test_build_input_column_maps_places_appended_variables_mid_vector() -> None:
    """Arm RUP layout: [in_lres 68->70 | in_hres 11 | y 68->70 | attrs 4], multi_step 1."""
    from anemoi.training.utils.checkpoint import build_input_column_maps

    maps = build_input_column_maps([68, 11, 68], [70, 11, 70], 1, 4, 68)
    assert set(maps) == {151, 151 + 68}
    m1 = maps[151].tolist()
    assert m1[:68] == list(range(68))                # in_lres keeps its place
    assert m1[68:79] == list(range(70, 81))          # in_hres shifted by 2
    assert m1[79:147] == list(range(81, 149))        # y_noised shifted by 2, its new vars at 149-150
    assert m1[147:151] == list(range(151, 155))      # node attributes shifted by 4
    m2 = maps[219].tolist()
    assert m2[:151] == m1 and m2[151:] == list(range(155, 155 + 68))  # decoder output block appended
    assert build_input_column_maps([68, 11, 68], [68, 11, 68], 1, 4, 68) is None
    assert build_input_column_maps([68, 11, 68], [67, 11, 68], 1, 4, 68) is None


def test_build_input_column_maps_multistep_is_time_major() -> None:
    from anemoi.training.utils.checkpoint import build_input_column_maps

    maps = build_input_column_maps([2, 1, 2], [3, 1, 3], 2, 0, 2)
    # old vector: [l0v0 l0v1 l1v0 l1v1 | h0 h1 | y0v0 y0v1 y1v0 y1v1]  (10)
    # new vector: [l0v0 l0v1 l0v2 l1v0 l1v1 l1v2 | h0 h1 | y0v0 y0v1 y0v2 y1v0 y1v1 y1v2]  (14)
    assert maps[10].tolist() == [0, 1, 3, 4, 6, 7, 8, 9, 11, 12]

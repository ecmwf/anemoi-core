# (C) Copyright 2026 Anemoi contributors.

import torch

from anemoi.models.variable.variable_dropout import VariableDropout


def test_variable_dropout_removes_only_prognostic_fields_and_keeps_one() -> None:
    torch.manual_seed(0)
    dropout = VariableDropout(dropout_rate=1.0, multi_variable_dropout=True, max_drop=8)
    x = torch.arange(4.0).reshape(1, 1, 1, 1, 4)

    retained, names, dropped = dropout(x, ["2t", "10u", "sp", "forcing"], [0, 1, 2])

    assert retained.shape[-1] == 2
    assert len(dropped) == 2
    assert "forcing" in names
    assert set(dropped.tolist()).issubset({0, 1, 2})


def test_variable_dropout_is_disabled_during_evaluation() -> None:
    dropout = VariableDropout(dropout_rate=1.0)
    dropout.eval()
    x = torch.ones(1, 1, 1, 1, 2)

    retained, names, dropped = dropout(x, ["2t", "10u"], [0, 1])

    assert retained is x
    assert names == ["2t", "10u"]
    assert dropped.numel() == 0

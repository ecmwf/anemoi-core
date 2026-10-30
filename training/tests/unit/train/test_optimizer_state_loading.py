# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""training.load_optimizer_state: a checkpoint's optimiser state, matched to the parameters by name."""

import pytest
import torch

from anemoi.training.utils.checkpoint import load_optimizer_state_by_name


class Net(torch.nn.Module):
    """Layers registered in the given order; `a` and `b` have the same shape, `c` another."""

    def __init__(self, order: str = "abc") -> None:
        super().__init__()
        shapes = {"a": (3, 3), "b": (3, 3), "c": (3, 2)}
        for name in order:
            self.add_module(name, torch.nn.Linear(*shapes[name]))
        self.register_buffer("scale", torch.ones(1))  # a buffer in the state_dict, not a parameter


def _trained_checkpoint(net: Net, lr: float = 1e-2) -> tuple[dict, torch.optim.Optimizer]:
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    for _ in range(3):
        optimizer.zero_grad()
        sum((p * torch.arange(p.numel()).reshape(p.shape)).sum() for p in net.parameters()).backward()
        optimizer.step()
    return {"state_dict": net.state_dict(), "optimizer_states": [optimizer.state_dict()]}, optimizer


def _states_by_name(net: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    return {name: optimizer.state[p] for name, p in net.named_parameters() if p in optimizer.state}


def test_each_parameter_gets_its_own_state_whatever_the_registration_order() -> None:
    old = Net("abc")
    checkpoint, old_optimizer = _trained_checkpoint(old)
    new = Net("cba")  # a and b swap places (same shape), c moves first
    new.load_state_dict(old.state_dict())
    optimizer = torch.optim.AdamW(new.parameters(), lr=5e-4)

    loaded = load_optimizer_state_by_name(optimizer, list(new.named_parameters()), checkpoint)

    assert loaded == 6
    expected = _states_by_name(old, old_optimizer)
    for name, state in _states_by_name(new, optimizer).items():
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(state[key], expected[name][key])


def test_the_learning_rate_and_hyperparameters_are_the_new_optimisers() -> None:
    old = Net()
    checkpoint, _ = _trained_checkpoint(old, lr=1e-2)
    new = Net()
    optimizer = torch.optim.AdamW(new.parameters(), lr=5e-4, weight_decay=0.05)
    load_optimizer_state_by_name(optimizer, list(new.named_parameters()), checkpoint)
    (group,) = optimizer.param_groups
    assert group["lr"] == 5e-4
    assert group["weight_decay"] == 0.05


def test_a_model_with_other_parameters_is_refused() -> None:
    checkpoint, _ = _trained_checkpoint(Net("abc"))
    smaller = Net("ab")
    with pytest.raises(ValueError, match="by name"):
        load_optimizer_state_by_name(
            torch.optim.AdamW(smaller.parameters()),
            list(smaller.named_parameters()),
            checkpoint,
        )


def test_a_state_of_another_shape_is_refused() -> None:
    checkpoint, _ = _trained_checkpoint(Net())
    other = Net()
    other.c = torch.nn.Linear(3, 4)  # same name, new shape
    with pytest.raises(ValueError, match="shape"):
        load_optimizer_state_by_name(torch.optim.AdamW(other.parameters()), list(other.named_parameters()), checkpoint)


def test_a_checkpoint_without_an_optimiser_state_is_refused() -> None:
    net = Net()
    with pytest.raises(ValueError, match="one optimiser"):
        load_optimizer_state_by_name(
            torch.optim.AdamW(net.parameters()),
            list(net.named_parameters()),
            {"state_dict": net.state_dict()},
        )

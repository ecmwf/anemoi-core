# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import torch

from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing import StepwiseProcessors


class DummyProcessor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.rand(1))

    def forward(self, x, in_place: bool = True, inverse: bool = False, **kwargs) -> torch.Tensor:
        del in_place, inverse, kwargs
        return x * self.scale


def _make_processors(name: str) -> Processors:
    return Processors([(name, DummyProcessor())])


def test_stepwise_processors_order_and_access() -> None:
    lead_times = ["6h", "12h", "18h"]
    stepwise = StepwiseProcessors(lead_times)
    proc_a = _make_processors("a")
    proc_c = _make_processors("c")

    stepwise.set("6h", proc_a)
    stepwise.set("18h", proc_c)

    assert len(stepwise) == 3
    assert stepwise.lead_times == lead_times
    assert stepwise[0] is proc_a
    assert stepwise[1] is None
    assert stepwise["12h"] is None
    assert stepwise[2] is proc_c
    assert list(stepwise) == [proc_a, None, proc_c]
    x = torch.ones(1)
    proc_out = stepwise["6h"](x, in_place=False)
    assert torch.allclose(proc_out, x * stepwise["6h"].processors["a"].scale)


def test_stepwise_processors_state_dict_has_only_set_entries() -> None:
    stepwise = StepwiseProcessors(["6h", "12h"])
    proc = _make_processors("a")
    stepwise.set("6h", proc)

    state_keys = list(stepwise.state_dict().keys())
    assert state_keys, "Expected non-empty state_dict after setting a processor."
    assert any(key.startswith("_processors.6h.") for key in state_keys)
    assert not any(key.startswith("_processors.12h.") for key in state_keys)

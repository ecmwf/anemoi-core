# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import pytest
import torch

from anemoi.models.data import Batch
from anemoi.models.data import TensorLayout
from anemoi.models.data.sources import TabularSource
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportModelEncProcDec
from anemoi.models.transport.sources import gaussian_source

OBS_LAYOUT = TensorLayout.from_tuple("grid", "variables")


def _obs_batch(samples: list[torch.Tensor], variables: list[str]) -> Batch:
    """A one-dataset batch of sparse observation samples laid out as ``(grid, variables)``."""
    nodes = [sample.shape[0] for sample in samples]
    return Batch(
        {
            "obs": TabularSource(
                name="obs",
                variables=variables,
                layout=OBS_LAYOUT,
                data=samples,
                coordinates=[torch.zeros(n, 2) for n in nodes],
                timedeltas=[torch.zeros(n) for n in nodes],
                boundaries=[(slice(0, n),) for n in nodes],
            ),
        },
    )


def test_sampling_template_supports_sparse_obs_shapes() -> None:
    output_names = ("o0", "o1", "o2", "o3")
    model = AnemoiTransportModelEncProcDec.__new__(AnemoiTransportModelEncProcDec)
    torch.nn.Module.__init__(model)
    model.num_output_channels = {"obs": len(output_names)}
    model.n_step_output = {"obs": 1}
    model.target_datasets = ["obs"]
    model.data_indices = {
        "obs": SimpleNamespace(
            name_to_index={name: idx for idx, name in enumerate(output_names)},
            model=SimpleNamespace(output=SimpleNamespace(ordered_names=output_names)),
        ),
    }
    model.statistics = {"obs": {}}
    model.is_dataset_static = {"obs": False}
    obs = _obs_batch([torch.zeros(2, 0), torch.zeros(5, 0)], variables=[])

    template = model._sampling_template({"obs": obs["obs"].template()}, obs)

    assert template["obs"].is_tabular
    assert [tuple(sample.shape) for sample in template["obs"].data] == [(2, 4), (5, 4)]
    assert template["obs"].variables == list(output_names)


def test_gaussian_source_creates_sparse_gaussian(monkeypatch: pytest.MonkeyPatch) -> None:
    target = _obs_batch([torch.zeros(2, 3), torch.zeros(5, 3)], variables=["a", "b", "c"])

    def fake_randn(shape: tuple[int, ...], device=None, dtype=None) -> torch.Tensor:
        return torch.full(shape, 4.0, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randn", fake_randn)

    source = gaussian_source(target)["obs"]

    assert source.is_tabular
    assert [sample.shape for sample in source.data] == [torch.Size([2, 3]), torch.Size([5, 3])]
    torch.testing.assert_close(source.data[0], torch.full((2, 3), 4.0))
    torch.testing.assert_close(source.data[1], torch.full((5, 3), 4.0))

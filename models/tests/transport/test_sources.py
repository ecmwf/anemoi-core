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

from anemoi.models.transport.settings import TransportSourceSettings
from anemoi.models.transport.sources import TransportSourceBuilder
from anemoi.models.transport.sources import TransportSourceRequest
from anemoi.models.transport.sources import reference_state_sampling_source
from anemoi.models.transport.sources import sampling_source_specs


def _data_indices_with_positions(
    output_names: tuple[str, ...], input_positions: list[int]
) -> dict[str, SimpleNamespace]:
    return {
        "data": SimpleNamespace(
            model=SimpleNamespace(
                output=SimpleNamespace(ordered_names=output_names),
                input=SimpleNamespace(positions_for_names=lambda names: input_positions),
            ),
        ),
    }


def test_reference_state_sampling_source_selects_latest_input_and_output_variables() -> None:
    x_data = torch.arange(1 * 3 * 1 * 4 * 5, dtype=torch.float32).reshape(1, 3, 1, 4, 5)
    x = {"data": x_data}
    data_indices = _data_indices_with_positions(("a", "b"), [0, 3])

    source = reference_state_sampling_source(x, data_indices=data_indices, n_step_output=2)

    expected = x_data[:, -1:, :, :, :].index_select(-1, torch.tensor([0, 3])).expand(-1, 2, -1, -1, -1)
    torch.testing.assert_close(source["data"], expected)


def test_reference_state_sampling_source_rejects_missing_input_variables() -> None:
    def raise_missing(names: tuple[str, ...]) -> list[int]:
        raise ValueError(f"missing variables: {names}")

    data_indices = {
        "data": SimpleNamespace(
            model=SimpleNamespace(
                output=SimpleNamespace(ordered_names=("a", "b")),
                input=SimpleNamespace(positions_for_names=raise_missing),
            ),
        ),
    }
    x = {"data": torch.zeros(1, 2, 1, 3, 4)}

    with pytest.raises(ValueError, match="reference_state transport sources require all model-output variables"):
        reference_state_sampling_source(x, data_indices=data_indices, n_step_output=1)


def test_reference_state_sampling_source_rejects_sparse_obs() -> None:
    data_indices = _data_indices_with_positions(("a",), [0])
    x = {"obs": [torch.zeros(2, 1)]}

    with pytest.raises(NotImplementedError, match="reference_state.*sparse observation"):
        reference_state_sampling_source(x, data_indices=data_indices, n_step_output=1)


def test_sampling_source_specs_support_sparse_obs_shapes() -> None:
    target_template = {"obs": [torch.zeros(2, 0), torch.zeros(5, 0)]}

    specs = sampling_source_specs(target_template, num_output_channels={"obs": 4})

    assert specs["obs"].is_sparse
    assert specs["obs"].shape == [(2, 4), (5, 4)]


def test_transport_source_builder_creates_scaled_sparse_gaussian(monkeypatch: pytest.MonkeyPatch) -> None:
    target = {"obs": [torch.zeros(2, 3), torch.zeros(5, 3)]}
    builder = TransportSourceBuilder(TransportSourceSettings(kind="gaussian", scale=2.0))

    def fake_randn(shape: tuple[int, ...], device=None, dtype=None) -> torch.Tensor:
        return torch.full(shape, 4.0, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randn", fake_randn)

    source = builder.build(TransportSourceRequest.from_data(target, default_kind="zero"))

    assert isinstance(source["obs"], list)
    assert [sample.shape for sample in source["obs"]] == [torch.Size([2, 3]), torch.Size([5, 3])]
    torch.testing.assert_close(source["obs"][0], torch.full((2, 3), 8.0))
    torch.testing.assert_close(source["obs"][1], torch.full((5, 3), 8.0))

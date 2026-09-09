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

from anemoi.models.transport.sources import reference_state_sampling_source


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

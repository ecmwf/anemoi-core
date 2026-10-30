# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import einops
import pytest
import torch

from anemoi.models.layers.input_transform import BaseInputTransform
from anemoi.models.layers.input_transform import FlattenInputTransform


def test_flatten_matches_rearrange_and_concat():
    batch, time, ensemble, grid, n_vars, n_attributes = 2, 3, 1, 5, 4, 6
    x = torch.randn(batch, time, ensemble, grid, n_vars)
    node_attributes = torch.randn(batch * ensemble * grid, n_attributes)

    out = FlattenInputTransform()(x, node_attributes, variables=[f"v{i}" for i in range(n_vars)])

    expected = torch.cat(
        (einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"), node_attributes),
        dim=-1,
    )
    torch.testing.assert_close(out, expected)


def test_flatten_ignores_metadata_and_keeps_variable_width():
    transform = FlattenInputTransform(variable_metadata={})

    assert transform.output_dim is None
    assert not list(transform.parameters())


def test_base_input_transform_is_abstract():
    with pytest.raises(TypeError):
        BaseInputTransform()

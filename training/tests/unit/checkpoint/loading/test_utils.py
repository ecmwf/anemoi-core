# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for state dict utilities."""

import torch

from anemoi.training.checkpoint.loading.utils import shape_mismatches


def test_shape_mismatches_gives_only_shared_keys_with_different_shapes() -> None:
    target = {"same": torch.zeros(10, 5), "differs": torch.zeros(30, 10)}
    source = {"same": torch.zeros(10, 5), "differs": torch.zeros(20, 10), "not_in_target": torch.zeros(1)}

    assert shape_mismatches(target, source).keys() == {"differs"}

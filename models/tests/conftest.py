# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os

import pytest
import torch

# Fix: Triton autotune config selection happens at module-import time, so tests need to set
# an explicit flag before importing the kernels. Without this, pytest silently used the full
# autotune grid because the old PYTEST_VERSION check was never set by the test runner.
os.environ.setdefault("ANEMOI_TRITON_TEST_MODE", "1")


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

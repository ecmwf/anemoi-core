# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path

import pytest

from anemoi.training.migrations.testing import ConfigFromContent


@pytest.fixture
def config_from_content(tmp_path: Path) -> ConfigFromContent:
    return ConfigFromContent(tmp_path / "document.yaml")

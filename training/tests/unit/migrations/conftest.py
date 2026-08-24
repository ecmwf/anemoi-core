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

from anemoi.training.migrations.testing import ConfigFromContents
from anemoi.training.migrations.testing import DocumentFromContent


@pytest.fixture
def document_from_content(tmp_path: Path) -> DocumentFromContent:
    return DocumentFromContent(tmp_path / "document.yaml")


@pytest.fixture
def config_from_contents(tmp_path: Path) -> ConfigFromContents:
    return ConfigFromContents(tmp_path)

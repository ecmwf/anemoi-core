# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path

from anemoi.training.migrations.config import Config
from anemoi.training.migrations.document import Document


class DocumentFromContent:
    def __init__(self, path: Path) -> None:
        self._path = path

    def __call__(self, content: str, prefix: Sequence[str] = ()) -> Document:
        self._path.write_text(content)
        return Document(self._path, prefix)


class ConfigFromContents:
    def __init__(self, path: Path) -> None:
        self._path = path

    def __call__(self, configs: Mapping[str, str]) -> Config:
        for filename, content in configs.items():
            path = self._path / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        return Config(self._path)

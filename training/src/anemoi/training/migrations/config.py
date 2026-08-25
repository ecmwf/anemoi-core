# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Generator
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Any

from anemoi.training.migrations.document import Document
from anemoi.training.migrations.document import Ops
from anemoi.training.migrations.interpolations import InterpolationReferences


class Config:
    def __init__(self, path: PathLike | str) -> None:
        self._path = Path(path)
        self._interpolations = InterpolationReferences()
        self._ops: list[tuple[Ops, tuple[Any, ...]]] = []

    @cached_property
    def documents(self) -> dict[str, Document]:
        documents: dict[str, Document] = {}
        for path in self._path.glob("**/*.yaml"):
            parts = path.relative_to(self._path).parts
            name = "/".join(parts)
            documents[name] = Document(path, ".".join(parts[:-1]))
        return documents

    def parse_interpolations(self) -> None:
        for document in self.documents.values():
            self._interpolations.parse_document(document)

    def selected_documents(self) -> Generator[Document, None, None]:
        yield from self.documents.values()

    def drop_key(self, keys: str) -> None:
        self._ops.append((Ops.DROP, (keys,)))

    def add_key(self, keys: str, value: Any) -> None:
        self._ops.append((Ops.ADD, (keys, value)))

    def rename_key(self, start: str, end: str) -> None:
        self._ops.append((Ops.RENAME, (start, end)))

    def exec_ops(self) -> None:
        for document in self.documents.values():
            document.exec_ops(self._ops)
        self._ops = []

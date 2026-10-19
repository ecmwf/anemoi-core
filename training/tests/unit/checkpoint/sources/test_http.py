# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for HTTPSource."""

from anemoi.training.checkpoint.sources.base import CheckpointSource
from anemoi.training.checkpoint.sources.http import HTTPSource


def test_http_source_extends_checkpoint_source() -> None:
    assert issubclass(HTTPSource, CheckpointSource)

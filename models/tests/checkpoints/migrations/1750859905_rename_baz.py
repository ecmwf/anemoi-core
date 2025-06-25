# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from anemoi.models.migrations import CkptType
from anemoi.models.migrations import Versions

versions: Versions = {
    "migration": "1.0.0",
    "anemoi-models": "0.9.0",
}


def migrate(ckpt: CkptType) -> CkptType:
    """Migrate the checkpoint"""
    ckpt["test"] = ckpt["baz"]
    del ckpt["baz"]
    return ckpt


def rollback(ckpt: CkptType) -> CkptType:
    """Rollback the migration"""
    ckpt["baz"] = ckpt["test"]
    del ckpt["test"]
    return ckpt

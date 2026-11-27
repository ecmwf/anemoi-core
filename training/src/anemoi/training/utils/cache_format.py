# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""File format and file layout of the records in the dataset cache.

The dataset cache and the cache server both use this module. It imports only numpy and numcodecs.
Thus, the server process starts quickly.
"""

import hashlib
import io
from pathlib import Path

import numpy as np
from numcodecs import Blosc

# Records are read and written inside dataloader workers. lz4 decodes ~2x and encodes ~3-8x faster
# than zstd there, for a similar compression ratio.
_BLOSC = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)

FULL_GRID = "all"


def entries_path(cache_path: Path, dataset_id: str) -> Path:
    """Return the directory of the records of one dataset."""
    return Path(cache_path) / hashlib.sha256(dataset_id.encode()).hexdigest()[:20] / "entries"


def record_path(entries: Path, sequence: int, position: int, grid_id: str = FULL_GRID) -> Path:
    """Return the file of one record."""
    return Path(entries) / grid_id / str(sequence) / f"{position}.npy.blosc"


def encode(value: np.ndarray) -> bytes:
    """Return an uncompressed .npy header followed by the blosc-compressed array data."""
    value = np.ascontiguousarray(value)
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, np.lib.format.header_data_from_array_1_0(value))
    return header.getvalue() + _BLOSC.encode(value)


def decode(data: bytes | memoryview) -> np.ndarray:
    """Decode the output of encode."""
    view = memoryview(data)
    # a version 1.0 .npy header: 6-byte magic string, 2-byte version, 2-byte header length
    header_end = 10 + int.from_bytes(view[8:10], "little")
    source = io.BytesIO(view[:header_end])
    np.lib.format.read_magic(source)
    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(source)
    value = np.empty(shape, dtype=dtype, order="F" if fortran_order else "C")
    _BLOSC.decode(view[header_end:], out=value)
    return value


def read_payload(path: Path) -> bytes | None:
    """Return the compressed bytes of a record, or None if the record is not in the cache."""
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None

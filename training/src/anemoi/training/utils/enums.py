# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from enum import IntEnum


class TensorDim(IntEnum):
    BATCH_SIZE = 0
    ENSEMBLE_DIM = 1
    GRID = 2
    VARIABLE = 3

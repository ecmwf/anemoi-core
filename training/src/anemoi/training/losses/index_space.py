# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from enum import Enum


class IndexSpace(str, Enum):
    """Tensor index-space for variable resolution in loss wrappers."""

    MODEL_OUTPUT = "model_output"
    DATA_OUTPUT = "data_output"
    DATA_FULL = "data_full"

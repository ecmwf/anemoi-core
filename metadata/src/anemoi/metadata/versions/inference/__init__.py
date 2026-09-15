# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Classes for metadata inference information"""

from typing import Union

from .v1 import DatasetInferenceConfig as data_inf_v1
from .v1 import InferenceMetadata as inf_meta_v1

InferenceMetadata = Union[inf_meta_v1]
DatasetInferenceConfig = Union[data_inf_v1]

__all__ = ["InferenceMetadata", "DatasetInferenceConfig"]

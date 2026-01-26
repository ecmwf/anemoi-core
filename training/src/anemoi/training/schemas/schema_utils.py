# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Literal
from typing import TypeAlias
from typing import TypeVar

# The DatasetDict type alias is intended to standardize the structure of dataset-related dictionaries
# across the codebase, improving type safety and code readability.
# The dataset-specific configurations are represented as a dictionary where keys are the dataset names
T = TypeVar("T")
DatasetDict: TypeAlias = dict[Literal["datasets"], dict[str, T]]

# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.models.utils.instantiate import InstantiationError
from anemoi.models.utils.instantiate import current_backend
from anemoi.models.utils.instantiate import get_class
from anemoi.models.utils.instantiate import get_object
from anemoi.models.utils.instantiate import instantiate
from anemoi.models.utils.instantiate import instantiation_backend
from anemoi.models.utils.instantiate import set_instantiation_backend

__all__ = [
    "instantiate",
    "get_object",
    "get_class",
    "InstantiationError",
    "set_instantiation_backend",
    "instantiation_backend",
    "current_backend",
]

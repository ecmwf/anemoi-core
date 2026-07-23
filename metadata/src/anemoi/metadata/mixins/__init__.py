# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Mixins providing shared functionality across metadata versions.

These mixins are inherited by the Metadata interface class to provide
common methods for variable selection and environment validation.
"""

from .validation import ValidationMixin
from .variables import VariablesMixin

__all__ = ["VariablesMixin", "ValidationMixin"]

# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import datetime

from anemoi.utils.dates import frequency_to_timedelta


def normalise_offset(x):
    return offset_to_string(offset_to_timedelta(x))


def offset_to_string(x):
    # copied here to make sure that the automatically generated keys are stable
    # so we don't use frequency_to_string from anemoi.utils

    assert isinstance(x, datetime.timedelta), type(x)

    total_seconds = int(x.total_seconds())

    if not total_seconds:
        return "0h"

    if total_seconds < 0:
        return f"-{offset_to_string(-x)}"

    if total_seconds % (24 * 3600) == 0 and total_seconds >= 10 * (24 * 3600):
        return f"{total_seconds // (24 * 3600)}d"

    if total_seconds % 3600 == 0:
        return f"{total_seconds // 3600}h"

    if total_seconds % 60 == 0:
        return f"{total_seconds // 60}m"

    return f"{total_seconds}s"


def offset_to_timedelta(x):

    if isinstance(x, str) and x.startswith("m"):
        x = "-" + x[1:]

    return frequency_to_timedelta(x)


def sum_offsets(a, b):
    a = offset_to_timedelta(a)
    b = offset_to_timedelta(b)
    x = a + b
    return offset_to_string(x)


def substract_offsets(a, b):
    a = offset_to_timedelta(a)
    b = offset_to_timedelta(b)
    x = a - b
    return offset_to_string(x)

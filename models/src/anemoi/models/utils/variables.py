# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import re


def parse_feature_name(name, variable_only=False):
    """'q_850' -> ('q', 850, True), '10u' -> ('u', 10, True), 'lsm' -> ('lsm', 0, False).

    With variable_only=True, returns just the physical variable name (e.g. 'q_850' -> 'q').
    """
    match = re.match(r"^(\d+)([a-z]+)$", name)
    if match:
        level, variable = match.groups()
        return variable if variable_only else (variable, int(level), True)
    match = re.match(r"^([a-z]+)_(\d+)$", name)
    if match:
        variable, level = match.groups()
        return variable if variable_only else (variable, int(level), True)
    return name if variable_only else (name, 0, False)

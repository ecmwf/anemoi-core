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
    """'q_850' -> ('q', 850, True), '10u' -> ('u', 0, False), 'lsm' -> ('lsm', 0, False).

    With variable_only=True, returns just the physical variable name (e.g. 'q_850' -> 'q',
    '10u' -> 'u').

    A leading-digit name like '10u' or '2t' is a near-surface diagnostic (metres above
    ground - 10u is 10m wind, 2t is 2m temperature), not a position on the pressure-level
    axis '_850' names use (hPa) - has_level=False groups it with the other single-level
    variables (lsm, etc.) instead of colliding with an unrelated pressure level that happens
    to share the same number (e.g. '2t' and 't_2' would otherwise both claim level 2).
    """
    match = re.match(r"^(\d+)([a-z]+)$", name)
    if match:
        _height, variable = match.groups()
        return variable if variable_only else (variable, 0, False)
    match = re.match(r"^([a-z]+)_(\d+)$", name)
    if match:
        variable, level = match.groups()
        return variable if variable_only else (variable, int(level), True)
    return name if variable_only else (name, 0, False)

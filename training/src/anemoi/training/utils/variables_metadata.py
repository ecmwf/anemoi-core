# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations


def get_variable_group_and_level(
    variable_name: str,
    group_variables: dict,
    metadata_variables: dict | None = None,
    default_group: str = "sfc",
) -> tuple[str, str, int]:
    """Get the group and level of a variable.

    Parameters
    ----------
    variable_name : str
        Name of the variable.
    group_variables : dict
        Dictionary with variable names as keys and groups as values.
    metadata_variables : dict, optional
        Dictionary with variable names as keys and metadata as values, by default None
    default_group : str, optional
        Default group to return if the variable is not found in the group_variables dictionary, by default "sfc"

    Returns
    -------
    str
        Group of the variable given in the training-config file.
    str
        Variable reference which corresponds to the variable name without the variable level
    str
        Variable level, i.e. pressure level or model level

    """
    variable_level = None
    mars_metadata_available = (
        metadata_variables and variable_name in metadata_variables and metadata_variables[variable_name].get("mars")
    )
    if mars_metadata_available and metadata_variables[variable_name]["mars"].get("param"):
        # if metadata is available: get variable name and level from metadata
        variable_level = metadata_variables[variable_name]["mars"].get("levelist")
        variable_name = metadata_variables[variable_name]["mars"]["param"]
    else:
        # if metadata not available: split variable name into variable name and level
        split = variable_name.split("_")
        if len(split) > 1 and split[-1].isdigit():
            variable_level = int(split[-1])
            variable_name = variable_name[: -len(split[-1]) - 1]
    if variable_name in group_variables:
        return group_variables[variable_name], variable_name, variable_level
    return default_group, variable_name, variable_level

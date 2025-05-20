# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations


class ExtractVariableGroupAndLevel:
    """Extract the group and level of a variable from dataset metadata and training-config file.

    Extract variables group from the training-config file and variable level from the dataset metadata.
    If dataset metadata is not available, the variable level is extracted from the variable name.

    Parameters
    ----------
    variable_groups : dict
        Dictionary with groups as keys and variable names as values
    metadata_variables : dict, optional
        Dictionary with variable names as keys and metadata as values, by default None
    """

    def __init__(
        self,
        variable_groups: dict,
        metadata_variables: dict | None = None,
    ) -> None:
        self.variable_groups = variable_groups
        # turn dictionary around
        self.group_variables = {}
        for group, variables in self.variable_groups.items():
            if isinstance(variables, str):
                variables = [variables]
            for variable in variables:
                self.group_variables[variable] = group
        assert "default" in self.variable_groups, "Default group not defined in variable_groups"
        self.default_group = self.variable_groups["default"]
        self.metadata_variables = metadata_variables

    def get_group_variables(self, group_name: str) -> list[str]:
        return self.variable_groups[group_name]

    def get_group_and_level(self, variable_name: str) -> tuple[str, str, int]:
        """Get the group and level of a variable.

        Parameters
        ----------
        variable_name : str
            Name of the variable.

        Returns
        -------
        group : str
            Group of the variable given in the training-config file.
        variable_name : str
            Variable reference which corresponds to the variable name without the variable level
        variable_level : str
            Variable level, i.e. pressure level or model level
        """
        variable_level = None
        mars_metadata_available = (
            self.metadata_variables
            and variable_name in self.metadata_variables
            and self.metadata_variables[variable_name].get("mars")
        )
        if mars_metadata_available and self.metadata_variables[variable_name]["mars"].get("param"):
            # if metadata is available: get variable name and level from metadata
            variable_level = self.metadata_variables[variable_name]["mars"].get("levelist")
            variable_name = self.metadata_variables[variable_name]["mars"]["param"]
        else:
            # if metadata not available: split variable name into variable name and level
            split = variable_name.split("_")
            if len(split) > 1 and split[-1].isdigit():
                variable_level = int(split[-1])
                variable_name = variable_name[: -len(split[-1]) - 1]
        if variable_name in self.group_variables:
            return self.group_variables[variable_name], variable_name, variable_level

        return self.default_group, variable_name, variable_level

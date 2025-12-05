# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from functools import lru_cache

from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.transform.variables import Variable

LOG = logging.getLogger(__name__)
GROUP_SPEC = str | list[str] | bool


@lru_cache
def _crack_variable_name(variable_name: str) -> tuple[str, str | None]:
    """Attempt to crack the variable name into parameter name and level.

    If cannot split, will return variable_name unchanged, and None

    Parameters
    ----------
    variable_name : str
        Name of the variable.

    Returns
    -------
    parameter : str
        Parameter reference which corresponds to the variable_name without the variable level.
        If cannot be split, will be variable_name unchanged.
    variable_level : str | None
        Variable level, i.e. pressure level or model level.
        If cannot be split, will be None.
    """
    split = variable_name.split("_")
    if len(split) > 1 and split[-1].isdigit():
        return variable_name[: -len(split[-1]) - 1], int(split[-1])

    return variable_name, None


def filter_variables(variables: dict[str, Variable], criteria: dict) -> list[Variable]:
    """Filter variables based on criteria."""
    result = []
    for var in variables.values():
        match = True
        assert isinstance(criteria, dict), f"variable_groups criteria must be a dict. Got: {type(criteria)}"
        for key, value in criteria.items():
            attr = getattr(var, key, None)
            if isinstance(value, list):
                if attr not in value:
                    match = False
                    break
            else:
                if attr != value:
                    match = False
                    break

        if match:
            result.append(var)
    return result


def build_variable_groups(
    metadata_variables: dict[str, Variable],
    variable_groups: dict,
    default_group: str | None = None,
) -> dict[str, list[Variable]]:
    """Build variable groups from a dictionary"""
    groups = {}
    for group_name, group_spec in variable_groups.items():
        # If the criteria is a list, it represent the variable names
        group_spec = {"name": group_spec} if isinstance(group_spec, str | list) else group_spec
        groups[group_name] = filter_variables(metadata_variables, group_spec)

    if default_group is not None:
        groups[default_group] = [
            var
            for var in metadata_variables.values()
            if var.name not in {v.name for group in groups.values() for v in group}
        ]

    return groups


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
        variable_groups: dict[str, GROUP_SPEC | dict[str, GROUP_SPEC]],
        metadata_variables: dict[str, dict | Variable] | None = None,
    ) -> None:

        if isinstance(variable_groups, DictConfig):
            variable_groups = OmegaConf.to_container(variable_groups, resolve=True)

        variable_groups = variable_groups.copy()

        assert "default" in variable_groups, "Default group not defined in variable_groups"
        self.default_group = variable_groups.pop("default")

        self.metadata_variables: dict[str, Variable] = {
            name: Variable.from_dict(name, val) if not isinstance(val, Variable) else val
            for name, val in (metadata_variables or {}).items()
        }
        self.variable_group_spec = variable_groups
        self.variable_groups = build_variable_groups(
            self.metadata_variables,
            variable_groups,
            default_group=self.default_group,
        )

    def get_group_specification(self, group_name: str) -> GROUP_SPEC | dict[str, GROUP_SPEC]:
        """Get the specification of a group."""
        return self.variable_group_spec.get(group_name, "default")

    def get_group(self, variable_name: str) -> str:
        """Get the group of a variable.

        Parameters
        ----------
        variable_name : str
            Name of the variable.

        Returns
        -------
        group : str
            Group of the variable
        """
        for group_name, variables in self.variable_groups.items():
            if variable_name in [v.name for v in variables]:
                LOG.debug(
                    "Variable %r is in group %r through specification : %r.",
                    variable_name,
                    group_name,
                    self.get_group_specification(group_name),
                )
                return group_name
        
        raise ValueError(f"{self.__class__.__name__} only supports variables found in the dataset. Variable {variable_name} not found.")

    def _is_metadata_trusted(self, variable_name: str) -> bool:
        """Check if the metadata for a variable is trusted.

        This checks if the variable has metadata and checks
        for valid relations.

        Parameters
        ----------
        variable_name : str
            Name of the variable.

        Returns
        -------
        bool
            True if the metadata is trusted, False otherwise.
        """
        if variable_name not in self.metadata_variables:
            return False

        level = self.metadata_variables[variable_name].level
        is_vertical_level = not self.metadata_variables[variable_name].is_surface_level

        # If level is not None and is not a surface level, True
        # If level is None and is a surface level, True
        return is_vertical_level ^ (level is None)

    def get_param(self, variable_name: str) -> str:
        """Get the parameter from a variable_name.

        Tries to use the metadata, but if not given
        will attempt to crack the name. If cannot
        crack will be the variable_name unchanged.

        Parameters
        ----------
        variable_name : str
            Name of the variable.

        Returns
        -------
        param : str
            Parameter of the variable.
            Either from the metadata or cracked
            name.
        """
        if self._is_metadata_trusted(variable_name):
            # if metadata is available: get param from metadata
            return self.metadata_variables[variable_name].param

        return _crack_variable_name(variable_name)[0]

    def get_level(self, variable_name: str) -> int | None:
        """Get the level of a variable.

        Parameters
        ----------
        variable_name : str
            Name of the variable.

        Returns
        -------
        variable_level : int | None
            Variable level, checks the variable metadata, or attempts
            to crack the name, if not found None.
        """
        if self._is_metadata_trusted(variable_name):
            # if metadata is available: get level from metadata
            return self.metadata_variables[variable_name].level

        return _crack_variable_name(variable_name)[1]

    def get_group_and_level(self, variable_name: str) -> tuple[str, str, int | None]:
        """Get the group and level of a variable.

        Parameters
        ----------
        variable_name : str
            Name of the variable.

        Returns
        -------
        group : str
            Group of the variable given in the training-config file.
        parameter : str
            Parameter reference which corresponds to the variable_name without the variable level.
            If cannot be split, will be variable_name unchanged.
        variable_level : int | None
            Variable level, i.e. pressure level or model level.
            If variable_name cannot be split, will be None.
        """
        return self.get_group(variable_name), self.get_param(variable_name), self.get_level(variable_name)

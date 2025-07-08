# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Union

from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.transform.variables import Variable

LOG = logging.getLogger(__name__)
GROUP_SPEC = Union[str, list[str], bool]


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


class ExtractVariableGroupAndLevel:
    """Extract the group and level of a variable from dataset metadata and training-config file."""

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
        self.variable_groups = variable_groups

        # Build metadata dictionary, fallback to _crack_variable_name() if needed
        self.metadata_variables: dict[str, Variable] = {}
        if metadata_variables:
            for name, val in metadata_variables.items():
                if isinstance(val, Variable):
                    self.metadata_variables[name] = val
                else:
                    self.metadata_variables[name] = Variable.from_dict(name, val)

    def get_group_specification(self, group_name: str) -> GROUP_SPEC | dict[str, GROUP_SPEC]:
        """Get the specification of a group."""
        return self.variable_groups[group_name]

    def get_group(self, variable_name: str) -> str:
        """Get the group of a variable."""
        for group_name, group_spec in self.variable_groups.items():
            if isinstance(group_spec, (list, str)):
                # Simple group: match on cracked param
                if self.get_param(variable_name) in (group_spec if isinstance(group_spec, list) else [group_spec]):
                    LOG.debug("Variable %r is in group %r", variable_name, group_name)
                    return group_name

            elif isinstance(group_spec, dict):
                # Complex group: requires metadata
                var_metadata = self.metadata_variables.get(variable_name)
                if var_metadata is None:
                    # Try to construct it from name
                    try:
                        param, level = _crack_variable_name(variable_name)
                        var_metadata = Variable.from_dict(variable_name, {"param": param, "level": level})
                        self.metadata_variables[variable_name] = var_metadata
                    except (ValueError, TypeError, KeyError):
                        continue  # Can't crack, skip this group

                if all(
                    getattr(var_metadata, key) in (val if isinstance(val, list) else [val])
                    for key, val in group_spec.items()
                ):
                    LOG.debug(
                        "Variable %r is in group %r through specification : %r.",
                        variable_name,
                        group_name,
                        group_spec,
                    )
                    return group_name

        # Fallback to default group
        return self.default_group

    def get_param(self, variable_name: str) -> str:
        """Get the parameter from a variable_name."""
        if variable_name in self.metadata_variables:
            return self.metadata_variables[variable_name].param

        return _crack_variable_name(variable_name)[0]

    def get_level(self, variable_name: str) -> str | None:
        """Get the level of a variable."""
        if variable_name in self.metadata_variables:
            return self.metadata_variables[variable_name].level

        return _crack_variable_name(variable_name)[1]

    def get_group_and_level(self, variable_name: str) -> tuple[str, str, int | None]:
        """Get the group, param, and level of a variable."""
        return (
            self.get_group(variable_name),
            self.get_param(variable_name),
            self.get_level(variable_name),
        )

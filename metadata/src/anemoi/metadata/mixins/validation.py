# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Mixin for environment validation."""

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import MetadataContract


class ValidationMixin:
    """Mixin providing environment validation methods.

    This mixin requires the class to have a ``_raw`` attribute that is
    a MetadataContract instance with an ``environment`` field that is a plain
    dict (e.g. ``{"python_version": "3.11.0", ...}``).
    """

    _raw: "MetadataContract"

    def validate_environment(self) -> list[str]:
        """Check checkpoint environment against current environment.

        Compares the Python version recorded in the checkpoint's environment
        dict against the current runtime.  Returns an empty list when the
        environment dict is missing or empty.

        Returns
        -------
        list[str]
            List of warning messages for any mismatches found.
            Empty list if environments match or no environment info is stored.

        Examples
        --------
        >>> warnings = metadata.validate_environment()
        >>> if warnings:
        ...     for w in warnings:
        ...         print(f"Warning: {w}")
        """
        warnings: list[str] = []

        env: dict = getattr(self._raw, "environment", {})
        if not env:
            return warnings

        checkpoint_python: str | None = env.get("python_version")
        if checkpoint_python is not None:
            current_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            if checkpoint_python != current_python:
                warnings.append(f"Python version mismatch: checkpoint={checkpoint_python}, current={current_python}")

        return warnings

    def get_environment_info(self) -> dict:
        """Get environment information from the checkpoint.

        Returns the raw environment dict stored in the checkpoint.  The
        contents are version-dependent; callers should use ``.get()`` for
        optional keys.

        Returns
        -------
        dict
            Dictionary containing environment details, or an empty dict if
            no environment information was recorded.

        Examples
        --------
        >>> info = metadata.get_environment_info()
        >>> print(info.get("python_version"))
        """
        return getattr(self._raw, "environment", {})

# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Exceptions for the anemoi.metadata package."""


class MetadataError(Exception):
    """Base exception for all metadata-related errors.

    All custom exceptions in this package inherit from this class,
    allowing users to catch all metadata errors with a single except clause.
    """


class VersionError(MetadataError):
    """Raised when a value needs a newer version of its component than the checkpoint has.

    This occurs when calling a component adapter method that is only available
    from a certain component version onwards (see
    :func:`~anemoi.metadata.components.base.requires`).
    """


class UnknownVersionError(MetadataError):
    """Raised when no component adapter is registered for a requested key.

    Reading a checkpoint never raises this for an unknown version: it falls
    back to the newest known adapter with a warning.  It is raised by exact
    lookups in :class:`~anemoi.metadata.registry.ComponentRegistry`.
    """


class CheckpointError(MetadataError):
    """Raised when there's an error reading/writing checkpoint metadata.

    This includes errors such as missing metadata files, corrupted
    archives, or I/O failures.
    """

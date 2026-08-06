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
    """Raised when a feature requires a newer schema version.

    This occurs when calling a method that is only available from a certain
    schema version onwards (e.g., calling ``get_provenance()`` on V1 metadata
    when provenance is only available in V2+).
    """


class UnknownVersionError(MetadataError):
    """Raised when an unknown metadata version is encountered.

    This typically occurs when loading metadata from a checkpoint that
    was created with a newer version of the package.
    """


class MigrationError(MetadataError):
    """Raised when a migration between versions fails.

    This can occur if a required migration function is not registered
    or if the migration logic encounters invalid data.
    """


class CheckpointError(MetadataError):
    """Raised when there's an error reading/writing checkpoint metadata.

    This includes errors such as missing metadata files, corrupted
    archives, or I/O failures.
    """

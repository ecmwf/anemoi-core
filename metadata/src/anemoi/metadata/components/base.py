# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Base class of component adapters.

A component adapter reads one structured, versioned part of the checkpoint
metadata (``data`` or ``task``) from the raw dictionary.  It implements its
getters against one layout.  An adapter for a newer or older layout subclasses
the nearest registered adapter (or version reader) and overrides only what
differs; it registers itself with
:meth:`anemoi.metadata.registry.ComponentRegistry.register`.
"""

from collections.abc import Callable
from collections.abc import Mapping
from functools import wraps
from typing import Any
from typing import ClassVar
from typing import TypeVar
from typing import cast

from packaging.version import Version

from anemoi.metadata.exceptions import VersionError

_F = TypeVar("_F", bound=Callable[..., Any])


def requires(min_version: str | None = None, max_version: str | None = None) -> Callable[[_F], _F]:
    """Gate a component adapter method on the version of its component.

    Use it on a method whose value older component versions do not record and
    that has no fallback for them: on a file whose component is older than
    *min_version* the method raises :class:`~anemoi.metadata.exceptions.VersionError`
    instead of returning ``None`` or an empty value, which would read as *not
    set*.  The version compared is the one of the adapter, that is of the
    component version the checkpoint has (stamped or inferred).  Stack it
    under ``@property`` for a property.

    If *max_version* is given, the method also raises
    :class:`~anemoi.metadata.exceptions.VersionError` on adapters of newer
    versions.

    Parameters
    ----------
    min_version : str | None
        Oldest version of the component the method works with (e.g. ``"1.0"``).
    max_version : str | None
        Newest version of the component the method works with (e.g. ``"2.0"``).

    Raises
    ------
    VersionError
        When the adapter version is older than *min_version*, newer than
        *max_version*, or missing; the message names the component, the
        required and the actual version.

    Examples
    --------
    >>> class LineageV1_1(LineageV1_0):
    ...     @property
    ...     @requires(min_version="1.1")
    ...     def new_value(self): ...
    """
    minimum = Version(min_version) if min_version is not None else None
    maximum = Version(max_version) if max_version is not None else None

    if minimum is None and maximum is None:
        message = "At least one of min_version or max_version must be specified."
        raise ValueError(message)
    if minimum is not None and maximum is not None and minimum > maximum:
        message = "min_version cannot be greater than max_version."
        raise ValueError(message)

    def decorator(func: _F) -> _F:
        @wraps(func)
        def wrapper(self: "ComponentAdapter", *args: Any, **kwargs: Any) -> Any:
            actual = getattr(self, "version", None)
            component = getattr(self, "component", type(self).__name__)
            needs = f"{component}.{func.__name__} requires {component} version"
            if minimum is not None:
                needs += f" >= {min_version}"
            if maximum is not None:
                needs += f" <= {max_version}"
            if (
                actual is None
                or (minimum is not None and Version(actual) < minimum)
                or (maximum is not None and Version(actual) > maximum)
            ):
                message = f"{needs}, but this checkpoint has {component} version {actual}"
                raise VersionError(message)
            return func(self, *args, **kwargs)

        return cast("_F", wrapper)

    return decorator


class ComponentAdapter:
    """Reads one component of the raw metadata dictionary.

    Reading is lenient: unknown keys are ignored and missing optional keys
    give empty values rather than errors.  The adapter never modifies the
    dictionary.

    Parameters
    ----------
    raw : Mapping[str, Any]
        The raw metadata dictionary of the checkpoint.

    Attributes
    ----------
    component : str
        Component name; set by the registration decorator.
    version : str
        Component version this adapter class reads; set by the decorator.
    name : str | None
        Behaviour name: the task name of a task adapter, the ``data_type`` of a
        data adapter; ``None`` for a generic adapter.  Set by the decorator.
    """

    component: ClassVar[str]
    version: ClassVar[str]
    name: ClassVar[str | None] = None

    def __init__(self, raw: Mapping[str, Any]) -> None:
        self._raw = raw

    @property
    def raw(self) -> Mapping[str, Any]:
        """The raw metadata dictionary this adapter reads."""
        return self._raw

    # ------------------------------------------------------------------
    # Helpers shared by the components
    # ------------------------------------------------------------------

    def _section(self, name: str) -> dict[str, Any]:
        """Return the top-level dict *name*, or an empty dict if absent or not a dict."""
        value = self._raw.get(name)
        return value if isinstance(value, dict) else {}

    @property
    def _inference(self) -> dict[str, Any]:
        """The ``metadata_inference`` block, or an empty dict."""
        return self._section("metadata_inference")

    def dataset_names(self) -> list[str]:
        """Return the ordered dataset names from ``metadata_inference.dataset_names``."""
        names = self._inference.get("dataset_names")
        return list(names) if isinstance(names, list) else []

    def resolve_dataset(self, dataset_name: str | None = None) -> str:
        """Return *dataset_name*, or the first dataset when it is ``None``.

        Raises
        ------
        IndexError
            If *dataset_name* is ``None`` and there are no datasets.
        KeyError
            If *dataset_name* is not one of the dataset names.
        """
        names = self.dataset_names()
        if dataset_name is None:
            if not names:
                msg = "Cannot get default dataset: dataset_names is empty"
                raise IndexError(msg)
            return names[0]
        if dataset_name not in names:
            msg = f"Dataset {dataset_name!r} not found. Available datasets: {names!r}"
            raise KeyError(msg)
        return dataset_name

    def _dataset_block(self, dataset_name: str | None = None) -> dict[str, Any]:
        """Return the ``metadata_inference.<dataset>`` block, or an empty dict if absent."""
        block = self._inference.get(self.resolve_dataset(dataset_name))
        return block if isinstance(block, dict) else {}

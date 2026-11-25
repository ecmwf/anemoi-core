# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Registry of component adapters.

Checkpoint metadata is read through small, layered *component adapters*.  A
component is a structured part of the metadata that is versioned on its own
(``data`` and ``task``).  Every adapter registers itself with
:meth:`ComponentRegistry.register` under ``(component, version)`` plus a
*name*: the task name for ``task`` adapters (``"forecaster"``, ...; an unnamed
adapter is the generic one) and the ``data_type`` for ``data`` adapters
(``"gridded"``, ``"tabular"``).  In practice the adapters of both components are
built and registered per version by ``register_task_version`` and
``register_data_version``.

An adapter for a newer or older layout subclasses the nearest registered
adapter and overrides only what differs, so a new component version needs only
the code that changed.

Examples
--------
>>> @ComponentRegistry.register("task", "1.1", name="forecaster")
... class Forecaster1_1(ForecasterV1_0):
...     def _timesteps(self, dataset_name): ...
>>> ComponentRegistry.resolve("task", "1.1", name="forecaster") is Forecaster1_1
True
"""

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import TypeVar

from packaging.version import InvalidVersion
from packaging.version import Version

from .exceptions import UnknownVersionError

if TYPE_CHECKING:
    from .components.base import ComponentAdapter

_AdapterT = TypeVar("_AdapterT", bound="ComponentAdapter")
_Key = tuple[str, str | None, str]


class ComponentRegistry:
    """Registry of component adapters.

    Adapters are keyed by ``(component, name, version)``: the name is the task
    name or the data type.  The unnamed entry (``name=None``) is the generic
    adapter used for task names that have no adapter of their own.

    The class uses class methods and class state only and is not meant to be
    instantiated.  Tests that register adapters should isolate the state with
    :meth:`snapshot` and :meth:`restore`.
    """

    _adapters: ClassVar[dict[_Key, type["ComponentAdapter"]]] = {}

    def __init__(self) -> None:
        msg = "ComponentRegistry is a static class and cannot be instantiated"
        raise RuntimeError(msg)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(
        cls,
        component: str,
        version: str,
        *,
        name: str | None = None,
    ) -> Callable[[type[_AdapterT]], type[_AdapterT]]:
        """Return a class decorator registering an adapter.

        Parameters
        ----------
        component : str
            Component name (``"data"`` or ``"task"``).
        version : str
            Component version the adapter reads (e.g. ``"1.0"``).
        name : str | None, optional
            Task name of a task adapter or ``data_type`` of a data adapter;
            ``None`` registers the generic adapter of the component.

        Returns
        -------
        Callable
            Decorator that registers the class and records ``component``,
            ``version`` and ``name`` on it.

        Raises
        ------
        ValueError
            If *version* is not a valid version string or the key is already
            registered.
        """
        try:
            Version(version)
        except InvalidVersion as exc:
            msg = f"Invalid component version {version!r}"
            raise ValueError(msg) from exc

        def decorator(adapter_cls: type[_AdapterT]) -> type[_AdapterT]:
            key = (component, name, version)
            if key in cls._adapters:
                label = f"{component} {name} {version}" if name else f"{component} {version}"
                msg = f"Component adapter {label} already registered"
                raise ValueError(msg)
            adapter_cls.component = component
            adapter_cls.version = version
            adapter_cls.name = name
            cls._adapters[key] = adapter_cls
            return adapter_cls

        return decorator

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, component: str, version: str, name: str | None = None) -> type["ComponentAdapter"]:
        """Return the adapter registered for exactly ``(component, name, version)``.

        Raises
        ------
        UnknownVersionError
            If no adapter is registered under that key.
        """
        try:
            return cls._adapters[(component, name, version)]
        except KeyError:
            raise UnknownVersionError(
                f"No {component} adapter registered for version {version!r}" + (f" and task {name!r}" if name else ""),
            ) from None

    @classmethod
    def names(cls, component: str) -> list[str]:
        """Return the sorted names that have named adapters for *component*."""
        return sorted({n for (c, n, _) in cls._adapters if c == component and n is not None})

    @classmethod
    def versions(cls, component: str, name: str | None = None) -> list[str]:
        """Return the versions registered for ``(component, name)``, oldest first."""
        found = [v for (c, n, v) in cls._adapters if c == component and n == name]
        return sorted(found, key=Version)

    @classmethod
    def newest_version(cls, component: str, name: str | None = None, *, any_name: bool = False) -> str:
        """Return the newest registered version of a component.

        Parameters
        ----------
        component : str
            Component name.
        name : str | None, optional
            Restrict to this task name or data type (``None`` is the generic adapter).
        any_name : bool, optional
            If ``True``, consider the adapters of every name.

        Raises
        ------
        UnknownVersionError
            If nothing is registered.
        """
        found = [v for (c, _, v) in cls._adapters if c == component] if any_name else cls.versions(component, name)
        if not found:
            msg = f"No {component} adapters registered"
            raise UnknownVersionError(msg)
        return max(found, key=Version)

    @classmethod
    def resolve(cls, component: str, version: str | None, name: str | None = None) -> type["ComponentAdapter"]:
        """Return the adapter to use for a component version, with fallbacks.

        * A task *name* without adapters of its own falls back to the generic
          adapter, with a warning once the component has named adapters.
        * An exact ``(name, version)`` match is returned as is.
        * Otherwise the nearest older registered version is used.  A warning
          is issued when *version* is newer than anything known for the
          component (a checkpoint from a newer package), unparseable, or
          older than the oldest adapter available.

        Parameters
        ----------
        component : str
            Component name.
        version : str | None
            Wanted version, from a stamp or inferred from the layout.
        name : str | None, optional
            Task name or data type.

        Returns
        -------
        type[ComponentAdapter]
            The adapter class.

        Raises
        ------
        UnknownVersionError
            If no adapter is registered for the component at all.
        """
        chosen = name
        if name is not None and not cls.versions(component, name):
            # A component without any named adapter has no name specific
            # behaviour, so the generic adapter is the expected one.
            if cls.names(component):
                warnings.warn(
                    f"No {component} adapter for {name!r}; using the generic {component} adapter.",
                    UserWarning,
                    stacklevel=3,
                )
            chosen = None

        available = cls.versions(component, chosen)
        if not available:
            msg = f"No {component} adapters registered"
            raise UnknownVersionError(msg)

        newest_known = cls.newest_version(component, any_name=True)

        try:
            wanted = Version(version) if version is not None else None
        except InvalidVersion:
            wanted = None
        if wanted is None:
            warnings.warn(
                f"Unusable {component} version {version!r}; using the newest known adapter ({available[-1]}).",
                UserWarning,
                stacklevel=3,
            )
            return cls._adapters[(component, chosen, available[-1])]

        if version in available:
            return cls._adapters[(component, chosen, version)]

        older = [v for v in available if Version(v) <= wanted]
        if older:
            use = older[-1]
            if wanted > Version(newest_known):
                warnings.warn(
                    f"Unknown {component} version {version!r}; using the newest known adapter ({use}).",
                    UserWarning,
                    stacklevel=3,
                )
        else:
            use = available[0]
            warnings.warn(
                f"{component} version {version!r} is older than any known adapter; using {use}.",
                UserWarning,
                stacklevel=3,
            )
        return cls._adapters[(component, chosen, use)]

    # ------------------------------------------------------------------
    # Test isolation
    # ------------------------------------------------------------------

    @classmethod
    def snapshot(cls) -> dict[_Key, type["ComponentAdapter"]]:
        """Return a copy of the registry state, to be passed to :meth:`restore`."""
        return dict(cls._adapters)

    @classmethod
    def restore(cls, state: dict[_Key, type["ComponentAdapter"]]) -> None:
        """Reset the registry to a state returned by :meth:`snapshot`."""
        cls._adapters = dict(state)

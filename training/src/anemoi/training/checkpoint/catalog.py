# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Component catalog for pipeline components with automatic discovery.

This module provides automatic discovery of checkpoint pipeline components
through module inspection. It dynamically finds all available sources,
loaders, and modifiers by scanning the package structure.

The catalog discovers components by:
- Scanning the appropriate modules (sources, loaders, modifiers)
- Finding all classes that inherit from the base classes
- Building a registry automatically at import time

This provides true discovery without hardcoding, making it easy to add
new components without updating the catalog.

Example
-------
>>> from anemoi.training.checkpoint.catalog import ComponentCatalog
>>>
>>> # List dynamically discovered components
>>> print(ComponentCatalog.list_sources())
>>> ['local', 's3', 'http', 'gcs', 'azure']
>>>
>>> # Get component target path
>>> target = ComponentCatalog.get_source_target('s3')
>>> print(target)
>>> 'anemoi.training.checkpoint.sources.S3Source'
>>>
>>> # Create component using standard Hydra config
>>> from hydra.utils import instantiate
>>> config = {
...     '_target_': 'anemoi.training.checkpoint.sources.S3Source',
...     'bucket': 'my-bucket',
...     'key': 'model.ckpt'
... }
>>> source = instantiate(config)
"""

import importlib
import inspect
import logging

logger = logging.getLogger(__name__)


class ComponentCatalog:
    """Dynamic catalog for checkpoint pipeline components.

    This catalog automatically discovers available components by scanning
    the package modules. It finds all classes that inherit from the base
    classes (CheckpointSource, LoadingStrategy, ModelModifier) and builds
    a registry dynamically.

    The discovery happens once at module import time and is cached for
    efficiency. This makes it easy to add new components without needing
    to update the catalog.

    Attributes
    ----------
    _sources : dict or None
        Cached mapping of source names to target paths
    _loaders : dict or None
        Cached mapping of loader names to target paths
    _modifiers : dict or None
        Cached mapping of modifier names to target paths

    Examples
    --------
    >>> # List dynamically discovered components
    >>> sources = ComponentCatalog.list_sources()
    >>> loaders = ComponentCatalog.list_loaders()
    >>>
    >>> # Get component class
    >>> source_class = ComponentCatalog.get_source_target('s3')
    >>>
    >>> # Use with Hydra
    >>> config = {
    ...     '_target_': 'anemoi.training.checkpoint.sources.S3Source',
    ...     'bucket': 'my-bucket'
    ... }
    >>> source = instantiate(config)
    """

    # Cached registries (populated on first access)
    _sources: dict[str, str] | None = None
    _loaders: dict[str, str] | None = None
    _modifiers: dict[str, str] | None = None

    @classmethod
    def _discover_components(cls, module_name: str, base_class_name: str) -> dict[str, str]:
        """Discover all classes in a module that inherit from a base class.

        This method finds concrete implementations by looking for classes that:
        1. Are defined in the target module
        2. Have a base class with the specified name
        3. Are not abstract (don't inherit directly from ABC)

        Parameters
        ----------
        module_name : str
            Full module path to scan (e.g., 'anemoi.training.checkpoint.sources')
        base_class_name : str
            Name of the base class to look for (e.g., 'CheckpointSource')

        Returns
        -------
        dict
            Mapping of component names to their full target paths
        """
        components = {}

        try:
            # Import the module to scan
            module = importlib.import_module(module_name)

            # Import ABC locally for checking abstract classes
            from abc import ABC

            # Scan all classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Skip if it's not defined in this module
                if obj.__module__ != module_name:
                    continue

                # Check if this class has the expected base class in its hierarchy
                # Look through the Method Resolution Order (MRO)
                has_expected_base = False
                for base in inspect.getmro(obj):
                    if base.__name__ == base_class_name and base != obj:
                        has_expected_base = True
                        break

                if not has_expected_base:
                    continue

                # Skip abstract classes using hybrid detection:
                # 1. ABC-based: Classes that inherit from ABC or have abstract methods
                # 2. Name-based: Classes following the "Base*" naming convention
                is_abstract = (
                    ABC in obj.__bases__
                    or (hasattr(obj, "__abstractmethods__") and obj.__abstractmethods__)
                    or obj.__name__.startswith("Base")
                )

                if is_abstract:
                    reason = []
                    if ABC in obj.__bases__:
                        reason.append("inherits from ABC")
                    if hasattr(obj, "__abstractmethods__") and obj.__abstractmethods__:
                        reason.append("has abstract methods")
                    if obj.__name__.startswith("Base"):
                        reason.append("follows Base* naming convention")

                    logger.debug(f"Skipping abstract class {name}: {', '.join(reason)}")
                    continue

                # This is a concrete implementation!
                simple_name = cls._class_to_simple_name(name)
                full_path = f"{module_name}.{name}"
                components[simple_name] = full_path
                logger.debug(f"Discovered {simple_name} -> {full_path}")

        except ImportError as e:
            # This is expected if the module doesn't exist yet
            logger.debug(f"Module {module_name} not found (this is normal if not yet implemented): {e}")
        except Exception as e:
            logger.error(f"Error discovering components in {module_name}: {e}")

        return components

    @classmethod
    def _class_to_simple_name(cls, class_name: str) -> str:
        """Convert a class name to a simple identifier.

        Examples
        --------
        - S3Source -> s3
        - LocalSource -> local
        - WeightsOnlyLoader -> weights_only
        - TransferLearningLoader -> transfer_learning

        Parameters
        ----------
        class_name : str
            The class name to convert

        Returns
        -------
        str
            Simple identifier for the component
        """
        # Remove common suffixes
        name = class_name
        for suffix in ["Source", "Loader", "Modifier", "Strategy"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break

        # Convert from CamelCase to snake_case
        import re

        name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
        name = re.sub("([A-Z]+)([A-Z][a-z])", r"\1_\2", name)

        return name.lower()

    @classmethod
    def _get_sources(cls) -> dict[str, str]:
        """Get the registry of checkpoint sources, discovering if needed."""
        if cls._sources is None:
            cls._sources = cls._discover_components("anemoi.training.checkpoint.sources", "CheckpointSource")
        return cls._sources

    @classmethod
    def _get_loaders(cls) -> dict[str, str]:
        """Get the registry of loaders, discovering if needed."""
        if cls._loaders is None:
            cls._loaders = cls._discover_components("anemoi.training.checkpoint.loaders", "LoadingStrategy")
        return cls._loaders

    @classmethod
    def _get_modifiers(cls) -> dict[str, str]:
        """Get the registry of modifiers, discovering if needed."""
        if cls._modifiers is None:
            cls._modifiers = cls._discover_components("anemoi.training.checkpoint.modifiers", "ModelModifier")
        return cls._modifiers

    @classmethod
    def list_sources(cls) -> list[str]:
        """List available checkpoint sources.

        Returns
        -------
        list of str
            Names of available checkpoint sources (dynamically discovered)

        Examples
        --------
        >>> sources = ComponentCatalog.list_sources()
        >>> print(sources)
        ['local', 's3', 'http', 'gcs', 'azure']
        """
        return sorted(cls._get_sources().keys())

    @classmethod
    def list_loaders(cls) -> list[str]:
        """List available loading strategies.

        Returns
        -------
        list of str
            Names of available loaders (dynamically discovered)

        Examples
        --------
        >>> loaders = ComponentCatalog.list_loaders()
        >>> print(loaders)
        ['cold_start', 'standard', 'transfer_learning', 'warm_start', 'weights_only']
        """
        return sorted(cls._get_loaders().keys())

    @classmethod
    def list_modifiers(cls) -> list[str]:
        """List available model modifiers.

        Returns
        -------
        list of str
            Names of available modifiers (dynamically discovered)

        Examples
        --------
        >>> modifiers = ComponentCatalog.list_modifiers()
        >>> print(modifiers)
        ['freeze', 'lora', 'prune', 'quantize']
        """
        return sorted(cls._get_modifiers().keys())

    @classmethod
    def get_source_target(cls, name: str) -> str:
        """Get the target path for a checkpoint source.

        Parameters
        ----------
        name : str
            Simple name of the source (e.g., 's3')

        Returns
        -------
        str
            Full target path for Hydra instantiation

        Raises
        ------
        ValueError
            If the source name is not recognized

        Examples
        --------
        >>> target = ComponentCatalog.get_source_target('s3')
        >>> print(target)
        'anemoi.training.checkpoint.sources.S3Source'
        """
        sources = cls._get_sources()
        if name not in sources:
            available = ", ".join(sorted(sources.keys()))
            raise ValueError(f"Unknown checkpoint source: '{name}'. Available sources: {available}")
        return sources[name]

    @classmethod
    def get_loader_target(cls, name: str) -> str:
        """Get the target path for a loading strategy.

        Parameters
        ----------
        name : str
            Simple name of the loader (e.g., 'weights_only')

        Returns
        -------
        str
            Full target path for Hydra instantiation

        Raises
        ------
        ValueError
            If the loader name is not recognized
        """
        loaders = cls._get_loaders()
        if name not in loaders:
            available = ", ".join(sorted(loaders.keys()))
            raise ValueError(f"Unknown loader strategy: '{name}'. Available loaders: {available}")
        return loaders[name]

    @classmethod
    def get_modifier_target(cls, name: str) -> str:
        """Get the target path for a model modifier.

        Parameters
        ----------
        name : str
            Simple name of the modifier (e.g., 'freeze')

        Returns
        -------
        str
            Full target path for Hydra instantiation

        Raises
        ------
        ValueError
            If the modifier name is not recognized
        """
        modifiers = cls._get_modifiers()
        if name not in modifiers:
            available = ", ".join(sorted(modifiers.keys()))
            raise ValueError(f"Unknown model modifier: '{name}'. Available modifiers: {available}")
        return modifiers[name]

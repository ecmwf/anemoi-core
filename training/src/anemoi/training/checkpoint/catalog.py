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

from __future__ import annotations

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
    def _is_abstract_class(cls, obj: type) -> bool:
        """Check if a class is abstract."""
        from abc import ABC

        return (
            # Only consider it abstract if ABC is a direct base class
            ABC in obj.__bases__
            # Only abstract if it has unimplemented abstract methods
            or (hasattr(obj, "__abstractmethods__") and bool(obj.__abstractmethods__))
            # Convention-based check for base classes
            or obj.__name__.startswith("Base")
        )

    @classmethod
    def _has_base_class(cls, obj: type, base_class_name: str) -> bool:
        """Check if a class has the expected base class in its hierarchy."""
        return any(base.__name__ == base_class_name and base != obj for base in inspect.getmro(obj))

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

            # Scan all classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Skip if it's not defined in this module
                if obj.__module__ != module_name:
                    continue

                # Check if this class has the expected base class in its hierarchy
                if not cls._has_base_class(obj, base_class_name):
                    continue

                # Skip abstract classes
                if cls._is_abstract_class(obj):
                    logger.debug("Skipping abstract class %s", name)
                    continue

                # This is a concrete implementation!
                simple_name = cls._class_to_simple_name(name)
                full_path = f"{module_name}.{name}"
                components[simple_name] = full_path
                logger.debug("Discovered %s -> %s", simple_name, full_path)

        except ImportError as e:
            # This is expected if the module doesn't exist yet
            logger.debug("Module %s not found (this is normal if not yet implemented): %s", module_name, e)
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(
                "Error discovering components in %s: %s. "
                "This may indicate malformed component classes. "
                "Components should inherit from the appropriate base class (%s) "
                "and be properly importable.",
                module_name,
                e,
                base_class_name,
            )

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

        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
        name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)

        return name.lower()

    @classmethod
    def _warn_about_discovery_issues(cls, component_type: str, discovered: dict[str, str]) -> None:
        """Provide smart warnings about component discovery results."""
        if not discovered:
            logger.warning(
                "No %s components were discovered. This might indicate:\n"
                "  • The %s module is not yet implemented\n"
                "  • Import errors in the %s module\n"
                "  • No concrete classes inherit from the base class\n"
                "To check for issues, try importing the module manually:\n"
                "  >>> from anemoi.training.checkpoint.%s import *",
                component_type,
                component_type,
                component_type,
                component_type,
            )
        elif len(discovered) < cls._get_expected_component_count(component_type):
            expected = cls._get_expected_component_count(component_type)
            logger.info(
                "Discovered %d %s components (expected ~%d). Available: %s\n"
                "This is normal during development when not all components are implemented yet.",
                len(discovered),
                component_type,
                expected,
                list(discovered.keys()),
            )
        else:
            logger.debug(
                "Successfully discovered %d %s components: %s",
                len(discovered),
                component_type,
                list(discovered.keys()),
            )

    @classmethod
    def _get_expected_component_count(cls, component_type: str) -> int:
        """Get expected number of components for smart warnings."""
        expectations = {
            "sources": 5,  # local, s3, http, gcs, azure
            "loaders": 4,  # weights_only, transfer_learning, warm_start, cold_start
            "modifiers": 3,  # freeze, lora, quantize (initially)
        }
        return expectations.get(component_type, 1)

    @classmethod
    def _get_sources(cls) -> dict[str, str]:
        """Get the registry of checkpoint sources, discovering if needed."""
        if cls._sources is None:
            cls._sources = cls._discover_components("anemoi.training.checkpoint.sources", "CheckpointSource")
            cls._warn_about_discovery_issues("sources", cls._sources)
        return cls._sources

    @classmethod
    def _get_loaders(cls) -> dict[str, str]:
        """Get the registry of loaders, discovering if needed."""
        if cls._loaders is None:
            cls._loaders = cls._discover_components("anemoi.training.checkpoint.loaders", "LoadingStrategy")
            cls._warn_about_discovery_issues("loaders", cls._loaders)
        return cls._loaders

    @classmethod
    def _get_modifiers(cls) -> dict[str, str]:
        """Get the registry of modifiers, discovering if needed."""
        if cls._modifiers is None:
            cls._modifiers = cls._discover_components("anemoi.training.checkpoint.modifiers", "ModelModifier")
            cls._warn_about_discovery_issues("modifiers", cls._modifiers)
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
            available = sorted(sources.keys())
            if not available:
                from .exceptions import CheckpointConfigError

                error_msg = (
                    f"Unknown checkpoint source: '{name}' - no checkpoint sources are currently available.\n"
                    "This usually means:\n"
                    "  • The checkpoint.sources module hasn't been implemented yet\n"
                    "  • There are import errors in the sources module\n"
                    "  • No concrete CheckpointSource classes were found\n"
                    "Check that checkpoint source classes inherit from CheckpointSource and are importable."
                )
                raise CheckpointConfigError(
                    error_msg,
                    config_path=f"source.type='{name}'",
                )

            # Provide helpful suggestions based on similar names
            # Find suggestions using list comprehension
            name_lower = name.lower()
            suggestions = [
                available_name
                for available_name in available
                if name_lower in available_name.lower() or available_name.lower() in name_lower
            ]

            error_msg = f"Unknown checkpoint source: '{name}'. Available sources: {', '.join(available)}"
            if suggestions:
                error_msg += f"\nDid you mean: {', '.join(suggestions)}?"
            error_msg += f"\n\nExample usage:\n  source:\n    type: {available[0]}"

            from .exceptions import CheckpointConfigError

            raise CheckpointConfigError(error_msg, config_path=f"source.type='{name}'")
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
            cls._handle_unknown_loader(name, loaders)
        return loaders[name]

    @classmethod
    def _handle_unknown_loader(cls, name: str, loaders: dict[str, str]) -> None:
        """Handle unknown loader with helpful error message.

        Parameters
        ----------
        name : str
            The requested loader name
        loaders : dict
            Available loaders mapping

        Raises
        ------
        CheckpointConfigError
            Always raises with helpful error message
        """
        from .exceptions import CheckpointConfigError

        available = sorted(loaders.keys())
        if not available:
            error_message = (
                f"Unknown loader strategy: '{name}' - no loaders are currently available.\n"
                "This usually means:\n"
                "  • The checkpoint.loaders module hasn't been implemented yet\n"
                "  • There are import errors in the loaders module\n"
                "  • No concrete LoadingStrategy classes were found\n"
                "Check that loader classes inherit from LoadingStrategy and are importable."
            )
            raise CheckpointConfigError(error_message, config_path=f"loading.type='{name}'")

        # Build error message with suggestions
        error_msg = cls._build_loader_error_message(name, available)
        raise CheckpointConfigError(error_msg, config_path=f"loading.type='{name}'")

    @classmethod
    def _build_loader_error_message(cls, name: str, available: list[str]) -> str:
        """Build detailed error message for unknown loader.

        Parameters
        ----------
        name : str
            The requested loader name
        available : list[str]
            List of available loader names

        Returns
        -------
        str
            Detailed error message with suggestions
        """
        error_msg = f"Unknown loader strategy: '{name}'. Available loaders: {', '.join(available)}"

        # Add suggestions based on similar names
        suggestions = cls._find_similar_names(name, available)
        if suggestions:
            error_msg += f"\nDid you mean: {', '.join(suggestions)}?"

        # Add context about loader types
        error_msg += cls._get_loader_type_descriptions(available)
        error_msg += f"\n\nExample usage:\n  loading:\n    type: {available[0]}"

        return error_msg

    @classmethod
    def _get_loader_type_descriptions(cls, available: list[str]) -> str:
        """Get descriptions of available loader types.

        Parameters
        ----------
        available : list[str]
            List of available loader names

        Returns
        -------
        str
            Formatted descriptions of loader types
        """
        if not available:
            return ""

        descriptions = "\n\nCommon loader types:"
        loader_docs = {
            "weights_only": "Load only model weights (fastest)",
            "transfer_learning": "Flexible loading with mismatch handling",
            "warm_start": "Resume training with full state",
            "cold_start": "Fresh training from pretrained weights",
        }

        for loader_name, description in loader_docs.items():
            if loader_name in available:
                descriptions += f"\n  • {loader_name}: {description}"

        return descriptions

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
            cls._handle_unknown_modifier(name, modifiers)
        return modifiers[name]

    @classmethod
    def _handle_unknown_modifier(cls, name: str, modifiers: dict[str, str]) -> None:
        """Handle unknown modifier with helpful error message.

        Parameters
        ----------
        name : str
            The requested modifier name
        modifiers : dict
            Available modifiers mapping

        Raises
        ------
        CheckpointConfigError
            Always raises with helpful error message
        """
        from .exceptions import CheckpointConfigError

        available = sorted(modifiers.keys())
        if not available:
            error_message = (
                f"Unknown model modifier: '{name}' - no modifiers are currently available.\n"
                "This usually means:\n"
                "  • The checkpoint.modifiers module hasn't been implemented yet\n"
                "  • There are import errors in the modifiers module\n"
                "  • No concrete ModelModifier classes were found\n"
                "Check that modifier classes inherit from ModelModifier and are importable."
            )
            raise CheckpointConfigError(error_message, config_path=f"modifiers[].type='{name}'")

        # Build error message with suggestions
        error_msg = cls._build_modifier_error_message(name, available)
        raise CheckpointConfigError(error_msg, config_path=f"modifiers[].type='{name}'")

    @classmethod
    def _build_modifier_error_message(cls, name: str, available: list[str]) -> str:
        """Build detailed error message for unknown modifier.

        Parameters
        ----------
        name : str
            The requested modifier name
        available : list[str]
            List of available modifier names

        Returns
        -------
        str
            Detailed error message with suggestions
        """
        error_msg = f"Unknown model modifier: '{name}'. Available modifiers: {', '.join(available)}"

        # Add suggestions based on similar names
        suggestions = cls._find_similar_names(name, available)
        if suggestions:
            error_msg += f"\nDid you mean: {', '.join(suggestions)}?"

        # Add context about modifier types
        error_msg += cls._get_modifier_type_descriptions(available)
        error_msg += f"\n\nExample usage:\n  modifiers:\n    - type: {available[0]}"

        return error_msg

    @classmethod
    def _get_modifier_type_descriptions(cls, available: list[str]) -> str:
        """Get descriptions of available modifier types.

        Parameters
        ----------
        available : list[str]
            List of available modifier names

        Returns
        -------
        str
            Formatted descriptions of modifier types
        """
        if not available:
            return ""

        descriptions = "\n\nCommon modifier types:"
        modifier_docs = {
            "freeze": "Freeze specific layers/parameters",
            "lora": "Low-Rank Adaptation fine-tuning",
            "quantize": "Model quantization for efficiency",
            "prune": "Remove less important connections",
        }

        for modifier_name, description in modifier_docs.items():
            if modifier_name in available:
                descriptions += f"\n  • {modifier_name}: {description}"

        return descriptions

    @classmethod
    def _find_similar_names(cls, name: str, available: list[str]) -> list[str]:
        """Find similar names from available options.

        Parameters
        ----------
        name : str
            The requested name
        available : list[str]
            List of available names

        Returns
        -------
        list[str]
            List of similar names that might be suggestions
        """
        name_lower = name.lower()
        return [
            available_name
            for available_name in available
            if name_lower in available_name.lower() or available_name.lower() in name_lower
        ]

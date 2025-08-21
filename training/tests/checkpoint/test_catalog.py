# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for component catalog with dynamic discovery."""

from abc import ABC
from abc import abstractmethod
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from anemoi.training.checkpoint.catalog import ComponentCatalog


class TestComponentCatalog:
    """Test ComponentCatalog class with dynamic discovery."""

    def test_list_sources(self):
        """Test listing available sources."""
        # Since we're using dynamic discovery, the actual sources depend on
        # what's implemented. For now, we just test the method works.
        sources = ComponentCatalog.list_sources()

        assert isinstance(sources, list)
        # The list might be empty if modules don't exist yet
        assert sources == sorted(sources)  # Check it's sorted

    def test_list_loaders(self):
        """Test listing available loaders."""
        loaders = ComponentCatalog.list_loaders()

        assert isinstance(loaders, list)
        assert loaders == sorted(loaders)  # Check it's sorted

    def test_list_modifiers(self):
        """Test listing available modifiers."""
        modifiers = ComponentCatalog.list_modifiers()

        assert isinstance(modifiers, list)
        assert modifiers == sorted(modifiers)  # Check it's sorted

    def test_class_to_simple_name(self):
        """Test converting class names to simple identifiers."""
        # Test various naming patterns
        assert ComponentCatalog._class_to_simple_name("S3Source") == "s3"
        assert ComponentCatalog._class_to_simple_name("LocalSource") == "local"
        assert ComponentCatalog._class_to_simple_name("HTTPSource") == "http"
        assert ComponentCatalog._class_to_simple_name("WeightsOnlyLoader") == "weights_only"
        assert ComponentCatalog._class_to_simple_name("TransferLearningLoader") == "transfer_learning"
        assert ComponentCatalog._class_to_simple_name("FreezeModifier") == "freeze"
        assert ComponentCatalog._class_to_simple_name("LoRAModifier") == "lo_ra"

    @patch("anemoi.training.checkpoint.catalog.importlib.import_module")
    def test_discover_components(self, mock_import):
        """Test component discovery mechanism."""
        # Create mock module with test classes
        mock_module = Mock()
        mock_module.__name__ = "anemoi.training.checkpoint.sources"

        # Create a mock base class that inherits from ABC
        class MockCheckpointSource(ABC):
            @abstractmethod
            def acquire(self):
                pass

        # Create concrete implementations
        class MockLocalSource(MockCheckpointSource):
            def acquire(self):
                return "local"

        class MockS3Source(MockCheckpointSource):
            def acquire(self):
                return "s3"

        # Set up the module attributes
        MockLocalSource.__module__ = "anemoi.training.checkpoint.sources"
        MockS3Source.__module__ = "anemoi.training.checkpoint.sources"
        MockCheckpointSource.__module__ = "anemoi.training.checkpoint.base"

        # Set up inspect.getmembers to return our mock classes
        # Note: Only include classes that belong to the target module
        mock_module_members = [
            ("MockLocalSource", MockLocalSource),
            ("MockS3Source", MockS3Source),
        ]

        with patch("anemoi.training.checkpoint.catalog.inspect.getmembers") as mock_getmembers:
            mock_getmembers.return_value = mock_module_members
            mock_import.return_value = mock_module

            # Test discovery
            components = ComponentCatalog._discover_components(
                "anemoi.training.checkpoint.sources", "MockCheckpointSource",
            )

            # Should find the concrete implementations
            assert "mock_local" in components
            assert "mock_s3" in components
            # Should not include the abstract base class
            assert "mock_checkpoint" not in components

            assert components["mock_local"] == "anemoi.training.checkpoint.sources.MockLocalSource"
            assert components["mock_s3"] == "anemoi.training.checkpoint.sources.MockS3Source"

    @patch("anemoi.training.checkpoint.catalog.importlib.import_module")
    def test_discover_components_import_error(self, mock_import):
        """Test that discovery handles import errors gracefully."""
        mock_import.side_effect = ImportError("Module not found")

        # Should return empty dict and not raise
        components = ComponentCatalog._discover_components("non.existent.module", "BaseClass")

        assert components == {}

    @patch("anemoi.training.checkpoint.catalog.importlib.import_module")
    def test_discover_components_hybrid_abstract_detection(self, mock_import):
        """Test hybrid abstract class detection (ABC + name-based)."""
        # Create mock module
        mock_module = Mock()
        mock_module.__name__ = "anemoi.training.checkpoint.sources"

        # Create base class that follows Base* naming convention but doesn't inherit from ABC
        class BaseNamedSource:
            def acquire(self):
                pass

        # Create ABC-based abstract class
        from abc import ABC
        from abc import abstractmethod

        class ABCSource(ABC):
            @abstractmethod
            def acquire(self):
                pass

        # Create concrete implementations
        class ConcreteNamedSource(BaseNamedSource):
            def acquire(self):
                return "concrete_named"

        class ConcreteABCSource(ABCSource):
            def acquire(self):
                return "concrete_abc"

        # Set up module attributes
        ConcreteNamedSource.__module__ = "anemoi.training.checkpoint.sources"
        ConcreteABCSource.__module__ = "anemoi.training.checkpoint.sources"
        BaseNamedSource.__module__ = "anemoi.training.checkpoint.sources"
        ABCSource.__module__ = "anemoi.training.checkpoint.sources"

        # Set up mock members (include everything for testing filtering)
        mock_module_members = [
            ("BaseNamedSource", BaseNamedSource),  # Should be filtered by name
            ("ABCSource", ABCSource),  # Should be filtered by ABC
            ("ConcreteNamedSource", ConcreteNamedSource),  # Should be included
            ("ConcreteABCSource", ConcreteABCSource),  # Should be included
        ]

        with patch("anemoi.training.checkpoint.catalog.inspect.getmembers") as mock_getmembers:
            mock_getmembers.return_value = mock_module_members
            mock_import.return_value = mock_module

            # Test discovery - should find concrete classes but skip both types of abstract classes
            components = ComponentCatalog._discover_components(
                "anemoi.training.checkpoint.sources", "BaseNamedSource",  # This matches BaseNamedSource
            )

            # Should find concrete implementations but skip abstract ones
            assert "concrete_named" in components

            # Now test with ABCSource as base
            components_abc = ComponentCatalog._discover_components(
                "anemoi.training.checkpoint.sources", "ABCSource",  # This matches ABCSource
            )

            # Should find the ABC-based concrete implementation
            assert "concrete_abc" in components_abc

            # Verify paths are correct
            assert components["concrete_named"] == "anemoi.training.checkpoint.sources.ConcreteNamedSource"
            assert components_abc["concrete_abc"] == "anemoi.training.checkpoint.sources.ConcreteABCSource"

    def test_get_source_target_when_empty(self):
        """Test getting source target when no sources are discovered."""
        # Clear the cached sources
        ComponentCatalog._sources = {}

        with pytest.raises(ValueError) as exc_info:
            ComponentCatalog.get_source_target("s3")

        assert "Unknown checkpoint source: 's3'" in str(exc_info.value)
        assert "Available sources:" in str(exc_info.value)

    def test_get_loader_target_when_empty(self):
        """Test getting loader target when no loaders are discovered."""
        # Clear the cached loaders
        ComponentCatalog._loaders = {}

        with pytest.raises(ValueError) as exc_info:
            ComponentCatalog.get_loader_target("weights_only")

        assert "Unknown loader strategy: 'weights_only'" in str(exc_info.value)
        assert "Available loaders:" in str(exc_info.value)

    def test_get_modifier_target_when_empty(self):
        """Test getting modifier target when no modifiers are discovered."""
        # Clear the cached modifiers
        ComponentCatalog._modifiers = {}

        with pytest.raises(ValueError) as exc_info:
            ComponentCatalog.get_modifier_target("freeze")

        assert "Unknown model modifier: 'freeze'" in str(exc_info.value)
        assert "Available modifiers:" in str(exc_info.value)

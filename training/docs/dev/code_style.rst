.. _code-style:

############################
 Code Quality and Style
############################

We enforce consistent code style using pre-commit hooks. All code must follow
these guidelines:

*******************
Code Formatting
*******************

#. We follow PEP 8 with some modifications:

   - Line length is 120 characters (not 79).
   - Uses ``black`` for consistent formatting.
   - Uses ``isort`` for import sorting with:
     - Single line imports.
     - Black-compatible formatting.
     - Project imports grouped under ``anemoi``.

#. Import organization:

   - Imports are sorted using ``isort``.
   - Force single-line imports.
   - Group order: standard library, third-party, anemoi packages.

*******************
 Linting
*******************

We use ``ruff`` for code linting, which checks for:

- Code style violations.
- Potential bugs.
- Complexity issues.
- Best practices.

*******************
Documentation Style
*******************

#. Documentation follows strict guidelines:

   - RST files are formatted using ``rstfmt``.
   - Docstrings must match function signatures (checked by ``docsig``).
   - Sphinx documentation is linted (``sphinx-lint``).

*******************
Pre-commit Checks
*******************

All code is automatically checked using pre-commit hooks that verify:

#. Code formatting:
   - Black formatting.
   - Import sorting.
   - Line endings and trailing whitespace.

#. Code quality:
   - No debugger statements.
   - No merge conflicts.
   - Type annotations.
   - No blanket ``noqa`` statements.

#. Documentation:
   - Docstring validation.
   - RST formatting.
   - Sphinx linting.


*******************
 File Organization
*******************

Proper file organization is crucial for maintaining a clean and maintainable codebase.
Follow these guidelines when adding new files or modifying existing ones:

Directory Structure
==================

#. Place new files in the appropriate package directory:

   - Core functionality goes in ``src/anemoi/<package_name>/``.
   - Tests go in ``tests/``.
   - Documentation in ``docs/``.
   - Group related functionality together in the same module for better organization
   and maintainability.

.. note::

   When adding new files, ensure they are properly included in
   ``__init__.py`` files if they should be part of the public API. Keep it minimal.
   Use it to define package-level exports using __all__.

.. note::

   Utility Functions Organization:

   - Use ``utils.py`` only for package-specific helper functions that don't fit in other modules.
   - If a utility function could be useful across multiple packages:
     - Move it to ``anemoi-utils`` package.
     - Document its general-purpose nature.
     - Ensure it remains stateless and reusable.
   - Avoid using ``utils.py`` as a catch-all; if multiple related utilities emerge,
     consider creating a dedicated module.


File Structure
=============

Within each file:

#. Start with the license header and imports:

   - Anemoi contributors license header.
   - Standard library imports.
   - Third-party imports.
   - Local imports.

#. Follow with any module-level constants or configurations.

#. Define classes and functions in a logical order:

   - Base classes before derived classes.
   - Related functions grouped together.
   - Public API before private implementations.

.. note::

   Use absolute imports within the package. Avoid wildcard (*) imports.


********************
 Naming Conventions
********************

#. Use descriptive names that clearly indicate purpose or functionality.

#. Files and Modules:

   - Use lowercase with underscores
   - Examples:
     - ``reduced_gaussian_grid.py`` ✅
     - ``ReducedGaussianGrid.py`` ❌
     - ``rgrid.py`` ❌ (too vague)

#. Classes:

   - Use PascalCase (CapWords)
   - Examples:
     - ``ReducedGaussianGridNodes`` ✅
     - ``MultiScaleEdges`` ✅
     - ``reduced_gaussian_grid_nodes`` ❌
     - ``Rgn`` ❌ (too cryptic)

#. Functions and Variables:

   - Use snake_case
   - Use verbs for functions, nouns for variables
   - Examples:
     - ``calculate_edge_weights()`` ✅
     - ``get_coordinates()`` ✅
     - ``node_attributes`` ✅
     - ``calculateEdgeWeights()`` ❌
     - ``crds`` ❌ (too vague)

#. Constants:

   - Use uppercase with underscores
   - Examples:
     - ``MAX_GRID_RESOLUTION`` ✅
     - ``DEFAULT_BATCH_SIZE`` ✅
     - ``MaxGridResolution`` ❌

#. Private Names:

   - Prefix with single underscore for internal use
   - Examples:
     - ``_validate_input()`` ✅
     - ``_cached_result`` ✅

#. Special Methods:

   - Use double underscores for Python special methods
   - Examples:
     - ``__init__`` ✅
     - ``__call__`` ✅

#. Type Variables:

   - Use PascalCase, preferably single letters or short names
   - Examples:
     - ``T`` ✅ (for generic type)
     - ``NodeType`` ✅
     - ``EdgeAttr`` ✅

#. Enums:

   - Use PascalCase for enum class names
   - Use UPPERCASE for enum members
   - Examples:
     - ``class NodeType(Enum):
     -     SOURCE = "source"
     -     TARGET = "target"``

#. Test Names:

   - Prefix with ``test_`` (methods) or ``Test`` (classes).
   - Be descriptive about what is being tested.
   - Include the scenario and expected outcome.
   - Examples:
     - ``test_reduced_gaussian_grid_with_invalid_resolution`` ✅
     - ``test_edge_builder_handles_empty_graph`` ✅
     - ``test_coordinates_are_in_radians`` ✅
     - ``testGrid`` ❌ (too vague)
     - ``test1`` ❌ (meaningless)

.. note::

   Avoid abbreviations unless they are widely understood in the domain
   (e.g., ``lat``, ``lon`` for latitude/longitude). Clarity is more
   important than brevity.

.. _development-guidelines:

########################
 Development Guidelines
########################

Please follow these development guidelines:

#. Open an issue before starting a feature or bug fix to discuss the
   approach with maintainers adn other users.
#. Ensure high-quality code with appropriate tests, documentation,
   linting, and style checks. TODO: Add reference to tests
#. Follow the :ref:`branching-guidelines`.
#. Follow the :ref:`commit-guidelines`.

.. _branching-guidelines:

**********************
 Branching Guidelines
**********************

-  Use feature branches for new features (e.g., `feature/your-feature`)
-  Use fix branches for bug fixes (e.g., `fix/your-bug`)
-  Use a descriptive name that indicates the purpose of the branch
-  Keep branches up to date with `main` before opening a Pull Request

.. _commit-guidelines:

*******************
 Commit Guidelines
*******************

When making commits to the repository, please follow these guidelines:

#. Make small, focused commits with clear and concise messages.

#. Follow the `Conventional Commits guidelines
   <https://www.conventionalcommits.org/>`_. The format is:
   ``type[(scope)][!]: description``. For example:

   - ``feat(training): add new loss function``
   - ``fix(graphs): resolve node indexing bug``
   - ``docs(readme): update installation steps``
   - ``feat(models)!: change model input format`` (breaking change)
   - ``refactor!: restructure project layout`` (breaking change)

   Common types include:
   
   - ``feat``: New feature
   - ``fix``: Bug fix
   - ``docs``: Documentation only
   - ``style``: Code style changes
   - ``refactor``: Code changes that neither fix bugs nor add features
   - ``test``: Adding or modifying tests
   - ``chore``: Maintenance tasks

   Add ``!`` after the type/scope to indicate a breaking change.

#. Use present tense and imperative mood in commit messages (e.g., "Add
   feature" not "Added feature").

#. Reference relevant issue numbers in commit messages when applicable
   (e.g., "fix: resolve data loading issue #123").

While these commit message conventions are recommended for all branches in
Anemoi, they are strictly enforced only for commits to the ``main``
branch. This enforcement is particularly important as our automated
release process (`release-please<https://github.com/googleapis/release-please>`_) relies on conventional commits to
generate changelogs and determine version bumps automatically.

.. _pullrequest-guidelines:

*************************
 Pull Request Guidelines
*************************

When submitting Pull Requests (PRs), please follow these guidelines:

#. Open a draft Pull Request early in your development process. This helps:
   
   - Make your work visible to other contributors.
   - Get early feedback on your approach.
   - Avoid duplicate efforts.
   - Track progress on complex changes.

#. Fill the PR template completely, including:
   
   - Clear description of the changes.
   - Link to related issues using GitHub keywords (e.g., "Fixes #123").
   - List of notable changes.
   - Any breaking changes or deprecations.
   - Testing instructions if applicable.

#. Ensure the PR title follows the :ref:`commit-guidelines`, as this will become
   the squash commit message when merged to ``main``.

#. Keep your PR focused and of reasonable size:
   
   - One PR should address one concern.
   - Split large changes into smaller, logical PRs.
   - Update documentation along with code changes.

#. Before marking as ready for review:
   
   - Ensure all tests pass locally.
   - Address any automated check failures.
   - Review your own changes.
   - Update based on any feedback received while in draft.

#. When ready for review:
   
   - Mark the PR as "Ready for Review"
   - Request reviews from appropriate team members.
   - Be responsive to review comments.
   - Update the PR description if significant changes are made.

#. After approval:
   
   - PRs are merged using squash merge to maintain a clean history.
   - The squash commit message will use the PR title.



****************
 Code Standards
****************


*************
 Type Hints
*************


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
   ``__init__.py`` files if they should be part of the public API. Keep it minimal—avoid adding heavy logic.
   Use it to define package-level exports using __all__.

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

Documentation
============

Each new file should include:

#. Docstring explaining the purpose of the module.

#. Clear documentation for public APIs.

#. Example usage in docstrings where appropriate.

#. References to related files or documentation.

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

********************************
 Version Control Best Practices
********************************

#. Always use pre-commit hooks to ensure code quality and consistency.
#. Never commit directly to the `develop` branch.
#. Create a new branch for your feature or bug fix, e.g.,
   `feature/<feature_name>` or `bugfix/<bug_name>`.
#. Submit a Pull Request from your branch to `develop` for peer review
   and testing.

******************************
 Code Style and Documentation
******************************

#. Follow PEP 8 guidelines for Python code style, the pre-commit hooks
   will help enforce this.
#. Write clear, concise docstrings for all classes and functions using
   the Numpy style.
#. Use type hints to improve code readability and catch potential
   errors.
#. Add inline comments for complex logic or algorithms.
#. Use absolute imports within the package.
#. Avoid wildcard (*) imports.


*********
 Testing
*********

#. Write unit tests for new features using pytest.
#. Ensure all existing tests pass before submitting a Pull Request.
#. Aim for high test coverage, especially for critical functionality.

****************************
 Performance Considerations
****************************

#. Profile your code to identify performance bottlenecks.
#. Optimize critical paths and frequently called functions.
#. Consider using vectorized operations when working with large
   datasets.

By following these guidelines, you'll contribute to a maintainable and
robust codebase for Anemoi Training.


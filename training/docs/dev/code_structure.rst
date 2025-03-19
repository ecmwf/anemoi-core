################
 Code Structure
################

Understanding and maintaining the code structure is crucial for
sustainable development of Anemoi Training. This guide outlines best
practices for contributing to the codebase.

******************************
 Subclassing for New Features
******************************

When creating a new feature, the recommended practice is to subclass
existing base classes rather than modifying them directly. This approach
preserves functionality for other users while allowing for
customization.

Example:
========

In `anemoi/training/diagnostics/callbacks.py`, the `BasePlotCallback`
serves as a foundation for other plotting callbacks. New plotting
features should subclass this base class.

*******************
 File Organization
*******************

When developing multiple new functions for a feature:

#. Create a new file in the folder (e.g., `callbacks/<new_feature>.py`)
   to avoid confusion with base functions.
#. Group related functionality together for better organization and
   maintainability.

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

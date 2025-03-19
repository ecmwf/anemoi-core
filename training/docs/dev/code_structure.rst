.. _code-structure:

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

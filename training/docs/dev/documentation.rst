###############
 Documentation
###############

We welcome contributions to the documentation of Anemoi, whether
improving existing documentation or adding documentation for specific
use cases or features. Your contributions help make Anemoi better for
everyone.

.. _documentation-guidelines:

**************************
 Documentation Guidelines
**************************

Docstrings
==========

Each new file should include:

#. Docstring explaining the purpose of the module.
#. Clear documentation for public APIs.
#. Example usage in docstrings where appropriate.
#. References to related files or documentation.

Follow the NumPy or Google style for docstrings to ensure consistency.

Documentation on ReadTheDocs
============================

When adding a new feature, modifying existing functionality, or working
on an undocumented use case, please consider updating the high-level
documentation. This ensures other users can understand and build upon
your work.

Steps for contributing documentation:

-  Identify the most appropriate section for your addition (e.g.,
   overall Anemoi documentation, package-level documentation, user
   guide, getting started, or developer guide).

-  Consider maintainability — are interfaces still evolving?

-  Write clear and concise documentation using simple, direct sentences.

-  Add references to related sections or external documentation where
   applicable.

-  Incorporate feedback from your reviewer regarding documentation
   clarity and completeness.

************************
 Building Documentation
************************

For each Pull Request, documentation is automatically built and hosted
on ReadTheDocs for review. It is automatically linked in the PR
description.

You can build the documentation locally to preview changes before
submitting a Pull Request. We use Sphinx for documentation.

You can install the dependencies for building the documentation with:

.. code:: bash

   pip install '.[docs]'

To build the documentation locally:

.. code:: bash

   cd docs
   make html

The generated documentation will be in `docs/_build/html/index.html`.

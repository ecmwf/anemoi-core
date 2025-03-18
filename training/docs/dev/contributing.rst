##############
 Contributing
##############

Thank you for your interest in Anemoi! This guide will show you how to
contribute to the Anemoi packages.

If you encounter a bug or have a feature request, the first step is to
let us know by raising an issue on GitHub using the following steps:

#. Check the existing issues to avoid duplicates.

#. If it's a new issue, create a detailed bug report or feature request
   by filling in the issue template.

#. Use clear, descriptive titles and provide as much relevant
   information as possible.

#. If you have a bug, include the steps to reproduce it.

#. If you have a feature request, describe the use case and expected
   behaviour.

#. If you are interested in solving the issue yourself, assign the issue
   to yourself and follow the steps in the
   :ref:`contributing-to-development` section.

.. _contributing-to-development:

### Contributing to Development

If you are interested in contributing to the development of the Anemoi
packages, please follow the steps below:

#. Fork the anemoi repository on GitHub to your personal/organisation
   account. See the `GitHub tutorial
   <https://docs.github.com/en/get-started/quickstart/fork-a-repo>`_.

#. Set up the development environment following the instructions in the
   :ref:`setting-up-the-development-environment` section.

#. Create a new branch for your developments, following the
   :ref:`branching-guidelines`.

#. Make your changes and ensure that your changes adhere to the
   :ref:`development-guidelines`.

#. Commit the changes using the :ref:`commit-guidelines`.

#. Push your branch to your fork on GitHub.

#. Open a Pull Request against the `main` branch of the original
   repository and fill in the Pull Request template.

#. Request a review from maintainers or other contributors, which will
   follow the :ref:`code-review-process`.

.. _setting-up-the-development-environment:

****************************************
 Setting Up the Development Environment
****************************************

#. Clone the repository:

   .. code:: bash

      git clone https://github.com/ecmwf/anemoi-core/
      cd anemoi-${package}

#. Install dependencies:

   .. code:: bash

      # For all dependencies
      pip install -e .

      # For development dependencies
      pip install -e '.[dev]'

#. (macOS only) Install pandoc for documentation building:

   .. code:: bash

      brew install pandoc

.. _pre-commit-hooks:

******************
 Pre-Commit Hooks
******************

We use pre-commit hooks to ensure code quality and consistency. To set
them up:

#. Install pre-commit hooks:

   .. code:: bash

      pre-commit install

#. Run hooks on all files to verify installation:

   .. code:: bash

      pre-commit run --all-files

.. _development-guidelines:

************************
 Development Guidelines
************************

Please follow these development guidelines:

#. Ensure high-quality code with appropriate tests, documentation,
   linting, and style checks.

#. Follow the :ref:`branching-guidelines`.

#. Open an issue before starting a feature or bug fix to discuss the
   approach with maintainers.

#. Make small, focused commits with clear and concise messages.

#. Follow the `Conventional Commits guidelines
   <https://www.conventionalcommits.org/>`_, e.g., "feat:", "fix:",
   "docs:", etc.

#. Use present tense and imperative mood in commit messages (e.g., "Add
   feature" not "Added feature").

#. Reference relevant issue numbers in commit messages when applicable.

.. _branching-guidelines:

### Branching Guidelines

-  Use feature branches for new features (e.g., `feature/your-feature`)
-  Use fix branches for bug fixes (e.g., `fix/your-bug`)
-  Use a descriptive name that indicates the purpose of the branch
-  Keep branches up to date with `main` before opening a Pull Request

.. _running-tests:

***************
 Running Tests
***************

We use pytest for our test suite. To run tests:

.. code:: bash

   # Run all tests
   pytest

   # Run tests in a specific file
   pytest tests/test_<file>.py

Note: Some tests, like `test_gnn.py`, may run slower on CPU and are
better suited for GPU execution.

To run integration tests:

.. code:: bash

   pytest --slowtest

.. _building-documentation:

************************
 Building Documentation
************************

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

Documentation is also automatically generated for Pull Requests on
ReadTheDocs.

.. _code-review-process:

*********************
 Code Review Process
*********************

The Anemoi packages have a set of automated checks to enforce coding
guidelines. These checks are run via GitHub Actions on every Pull
Request. For security reasons, maintainers must review code changes
before enabling automated checks.

### Review Steps

#. Ensure that all the :ref:`development-guidelines` criteria are met
   before submitting a Pull Request.
#. Request a review from maintainers or other contributors, noting that
   support is on a best-efforts basis.
#. After an initial review, a maintainer will enable automated checks to
   run on the Pull Request.
#. Reviewers may provide feedback or request changes to your
   contribution.
#. Once approved, a maintainer will merge your Pull Request into the
   appropriate branch.

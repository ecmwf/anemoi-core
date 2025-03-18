##############
 Contributing
##############

Thank you for your interest in Anemoi! This guide will show you how to
contribute to the Anemoi packages.

If you encounter a bug or have a feature request, the first step
is to let us know by raising an issue on GitHub using the following steps:
#. Check the existing issues to avoid duplicates.
#. If it's a new issue, create a detailed bug report or feature request
   by filling in the issue template.
#. Use clear, descriptive titles and provide as much relevant
   information as possible.
#. If you have a bug, include the steps to reproduce it.
#. If you have a feature request, describe the use case and expected
   behaviour.
#. If you are interested in solving the issue yourself, assign
   the issue to yourself and follow the steps below.

If you are interested in contributing to the development of
the Anemoi packages, please follow the steps below:
#. Fork the anemoi repository on GitHub to your personal/organisation
   account.
#TODO: add link to Github tutorial
#. Set up the development environment following the instructions below.
#TODO: add reference
#. Create a new branch for your developments.
#TODO: Add branch guideline. Add reference
#. Make your changes and ensure that your changes adheres to the
   coding guidelines.
#. Commit the changes using the `Commit Guidelines`_ above.
#. Push your branch to your fork on GitHub.
#. Open a Pull Request against the `main` branch of the original
   repository and fill in the Pull Request template.
#. Request a review from maintainers or other contributors, which
   will follow the code review process. # TODO add link to section below

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

***********************
 Development Guidelines
***********************

#TODO add coding guidelines in terms quality: tests, documentation, linting, style, etc.
#TODO add branching guidelines

Ideally, open an issue for the feature or bug fix you're working on
before starting development, to discuss the approach with maintainers.

When committing code changes:

#. Make small, focused commits with clear and concise messages.

#. Follow the `Conventional Commits guidelines
   <https://www.conventionalcommits.org/>`_, e.g., "feat:", "fix:",
   "docs:", etc.

#. Use present tense and imperative mood in commit messages (e.g., "Add
   feature" not "Added feature").

#. Reference relevant issue numbers in commit messages when applicable.

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

# TODO: add here how to run integration tests --slowtest, etc.

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

# TODO: mention documentation is automatically generated in PRs on readthedocs

*********************
 Code Review Process
*********************
The anemoi packages have a set of automated checks to enforce
the coding guidelines.
This is done through GitHub Actions, which will run the checks
on every Pull Request.
For security reasons, the maintainers of the Anemoi packages
first need to review the code changes before running the automated checks.

The code review process contains the following steps:
#. Ensure that all the coding guidelines criteria are met before
   submitting a Pull Request.
#. Request a review from maintainers or other contributors,
   keeping in mind these packages are supported on a best endeavours basis.
#. After a first review, the maintainer will enable the automated
   checks to run on the Pull Request.
#. Reviewers may have feedback or comments on your contributions.
#. Once approved, a maintainer will merge your Pull Request into
   the appropriate branch.
#TODO: rephrase

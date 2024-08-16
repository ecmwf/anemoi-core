##############
 Contributing
##############

Thank you for your interest in contributing to Anemoi Training! This
guide will help you get started with the development process.

****************************************
 Setting Up the Development Environment
****************************************

#. Clone the repository:

   .. code:: bash

      git clone https://github.com/ecmwf/anemoi-training/
      cd anemoi-training

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

*******************
 Commit Guidelines
*******************

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

#. Update the ``CHANGELOG.md`` file with a human-friendly summary of
   your changes.

**********************
 Pull Request Process
**********************

#. Create a new branch for your feature or bug fix.
#. Make your changes and commit them using the guidelines above.
#. Push your branch to your fork on GitHub.
#. Open a Pull Request against the `develop` branch of the main
   repository.
#. Ensure all tests pass and the code adheres to the project's style
   guidelines.
#. Request a review from maintainers or other contributors.

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

*********************
 Code Review Process
*********************

#. All code changes must be reviewed before merging.
#. Address any feedback or comments from reviewers promptly.
#. Once approved, a maintainer will merge your Pull Request.

******************
 Reporting Issues
******************

If you encounter a bug or have a feature request:

#. Check the existing issues to avoid duplicates.
#. If it's a new issue, create a detailed bug report or feature request.
#. Use clear, descriptive titles and provide as much relevant
   information as possible.

Thank you for contributing to Anemoi Training! Your efforts help improve
the project for everyone.

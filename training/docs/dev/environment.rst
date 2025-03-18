.. _setting-up-the-development-environment:

########################################
 Setting Up the Development Environment
########################################

To create a development environment follow the steps below,

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

We use `pre-commit<https://pre-commit.com>`_ hooks to ensure code
quality and consistency. To set them up:

#. Install pre-commit hooks:

   .. code:: bash

      pre-commit install

#. Run hooks on all files to verify installation:

   .. code:: bash

      pre-commit run --all-files

These pre-commit hooks will run for each commit.

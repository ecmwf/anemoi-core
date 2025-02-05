############
 Installing
############

****************
 Python Version
****************

-  Python (3.9 to 3.12)

We require at least Python 3.9, and currently do not support >3.12. This
is due to an issue with one of the required dependencies.

**************
 Installation
**************

Environments
============

We currently do not provide a conda build of anemoi-training so the
suggested installation is through Python virtual environments.

For linux the process to make and use a venv is as follows,

.. code:: bash

   python -m venv /path/to/my/venv
   source /path/to/my/venv/bin/activate

Instructions
============

To install the package, you can use the following command:

.. code:: bash

   python -m pip install anemoi-training

We also maintain other dependency sets for different subsets of
functionality:

.. code:: bash

   python -m pip install "anemoi-training[profile]" # Install optional dependencies for profiling gpu usage
   python -m pip install "anemoi-training[docs]" # Install optional dependencies for generating docs

.. literalinclude:: ../../pyproject.toml
   :language: toml
   :start-at: [project.optional-dependencies.all]
   :end-before: [project.urls.Changelog]

**********************
 Development versions
**********************

To install the most recent development version, install from github:

.. code::

   $ python -m pip install git+https://github.com/ecmwf/anemoi-core.git#subdirectory=training

*********
 Testing
*********

To run the test suite after installing anemoi-training, install (via
pypi) `py.test <https://pytest.org>`__ and run ``pytest`` in the
``training`` directory of the anemoi-core repository.

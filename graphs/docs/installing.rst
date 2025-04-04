.. _installing-graphs:

############
 Installing
############

****************
 Python Version
****************

-  Python (> 3.9)

We require at least Python 3.9.

**************
 Installation
**************

Environments
============

We currently do not provide a conda build of anemoi-graphs so the
suggested installation is through Python virtual environments.

For linux the process to make and use a venv is as follows,

.. code:: bash

   python -m venv /path/to/my/venv
   source /path/to/my/venv/bin/activate

Instructions
============

To install the package, you can use the following command:

.. code:: bash

   python -m pip install anemoi-graphs

We also maintain other dependency sets for different subsets of
functionality:

.. code:: bash

   python -m pip install "anemoi-graphs[docs]" # Install optional dependencies for generating docs

.. literalinclude:: ../../pyproject.toml
   :language: toml
   :start-at: [project.optional-dependencies.all]
   :end-before: [project.urls.Changelog]

**********************
 Development versions
**********************

To install the most recent development version, install from github:

.. code::

   $ python -m pip install git+https://github.com/ecmwf/anemoi-core.git#subdirectory=graphs

*********
 Testing
*********

To run the test suite after installing anemoi-graphs, install (via pypi)
`py.test <https://pytest.org>`__ and run ``pytest`` in the ``graphs``
directory of the anemoi-core repository.

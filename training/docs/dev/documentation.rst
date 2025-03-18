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
.. _cli-introduction:

=============
Introduction
=============

When you install the `anemoi-graphs` package, this will also install command line tool
called ``anemoi-graphs`` which can be used to design and inspect weather graphs.

The tool can provide help with the ``--help`` options:

.. code-block:: bash

    % anemoi-graphs --help

The commands are:

.. toctree::
    :maxdepth: 1

    create
    describe
    inspect

.. argparse::
    :module: anemoi.graphs.__main__
    :func: create_parser
    :prog: anemoi-graphs
    :nosubcommands:

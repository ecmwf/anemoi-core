.. _cli-describe:

========
describe
========

Use this command to describe a graph stored in your filesystem. It will print graph information to the console.

The syntax of the recipe file is described in :doc:`building graphs <../graphs/introduction>`.

.. argparse::
    :module: anemoi.graphs.__main__
    :func: create_parser
    :prog: anemoi-graphs
    :path: describe

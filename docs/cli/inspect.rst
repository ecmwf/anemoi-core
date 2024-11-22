.. _cli-inspect:

========
inspect
========

Use this command to inspect a graph stored in your filesystem. A set of interactive and static visualisations
are generated to allow visual inspection of the graph design.

The syntax of the recipe file is described in :doc:`building graphs <../graphs/introduction>`.

.. argparse::
    :module: anemoi.graphs.__main__
    :func: create_parser
    :prog: anemoi-graphs
    :path: inspect

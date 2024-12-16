################################################
 Multi-scale connections at refined icosahedron
################################################

The multi-scale connections can only be defined with the same source and
target nodes. Edges of different scales are defined based on the
refinement level of an icosahedron. The higher the refinement level, the
shorther the length of the edges. By default, all possible refinements
levels are considered.

To use this method to build your connections, you can use the following
YAML configuration:

.. code:: yaml

   edges:
     -  source_name: source
        target_name: source
        edge_builders:
        - _target_: anemoi.graphs.edges.MultiScaleEdges
          x_hops: 1

where `x_hops` is the number of hops between two nodes of the same
refinement level to be considered neighbours, and then connected.

.. note::

   This method is used by data-driven weather models like GraphCast to
   process the latent/hidden state.

.. csv-table:: Triangular refinements specifications (x_hops=1)
   :file: ./tri_refined_edges.csv
   :header-rows: 1

.. warning::

   This connection method is only support for building the connections
   within a set of nodes defined with the ``TriNodes`` or ``HexNodes``
   classes.

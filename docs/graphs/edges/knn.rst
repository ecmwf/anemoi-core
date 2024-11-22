######################
 K-Nearest Neighbours
######################

The knn method is a method for establishing connections between two sets
of nodes. Given two sets of nodes, (`source`, `target`), the knn method
connects all target nodes, to their ``num_nearest_neighbours`` nearest
source nodes.

To use this method to build your connections, you can use the following
YAML configuration:

.. code:: yaml

   edges:
     -  source_name: source
        target_name: target
        edge_builders:
        - _target_: anemoi.graphs.edges.KNNEdges
          num_nearest_neighbours: 3

.. note::

   The KNNEdges method is recommended for the decoder edges, to connect
   all target nodes with the surrounding source nodes.

#####################
 K-Nearest Neighbors
#####################

The knn method is a method for establishing connections between two sets
of nodes. Given two sets of nodes, (`source`, `target`), the knn method
connects all destination nodes, to their ``num_nearest_neighbours``
nearest source nodes.

To use this method to build your connections, you can use the following
YAML configuration:

.. code:: yaml

   edges:
     -  source_name: source
        target_name: destination
        edge_builder:
          _target_: anemoi.graphs.edges.KNNEdges
          num_nearest_neighbours: 3

.. note::

   The knn method is recommended for the decoder edges, to connect all
   data nodes with the surrounding hidden nodes.

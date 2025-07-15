.. _knn:

######################
 K-Nearest Neighbours
######################

The k-nearest neighbours (KNN) method is a method for establishing
connections between two sets of nodes. Given two sets of nodes,
(`source`, `target`), the KNN method connects all target nodes to their
``num_nearest_neighbours`` nearest source nodes.

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

   The ``KNNEdges`` method is recommended for the decoder edges, to
   connect all target nodes with the surrounding source nodes.

There is also a reversed KNN edges variant that will connect each target
node to its k nearest source nodes, but the resulting edge direction is
still from the source nodes to the target nodes.

.. code:: yaml

   edges:
     -  source_name: source
        target_name: target
        edge_builders:
        - _target_: anemoi.graphs.edges.ReversedKNNEdges
          num_nearest_neighbours: 3 :contentReference[oaicite:5]{index=5}

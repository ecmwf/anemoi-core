.. _knn:

######################
 K-Nearest Neighbours
######################

The k-nearest neighbours (KNN) method is a method for establishing
connections between two sets of nodes. Given two sets of nodes,
(`source`, `target`), the KNN method connects all target nodes to their
``num_nearest_neighbours`` nearest source nodes.

When ``source`` and ``target`` refer to the same node set, you can set
``exclude_self_edges: true`` to drop self edges from the resulting
graph.

To use this method to build your connections, you can use the following
YAML configuration:

.. code:: yaml

   edges:
     -  source_name: source
        target_name: target
        edge_builders:
        - _target_: anemoi.graphs.edges.KNNEdges
          num_nearest_neighbours: 3

To exclude self edges when connecting a node set to itself, use:

.. code:: yaml

   edges:
     -  source_name: source
        target_name: source
        edge_builders:
        - _target_: anemoi.graphs.edges.KNNEdges
          num_nearest_neighbours: 3
          exclude_self_edges: true

.. note::

   When ``exclude_self_edges: true`` is enabled, self edges are removed
   after the KNN graph is built. This means some query nodes may end up
   with fewer than ``num_nearest_neighbours`` edges.

.. note::

   The ``KNNEdges`` method is recommended for the decoder edges, to
   connect all target nodes with the surrounding source nodes.

###############################
 Reversed K-Nearest Neighbours
###############################

The reversed k-nearest neighbours (``ReversedKNNEdges``) method is
similar to the standard KNN method, but instead establishes connections
based on the nearest neighbours of each source node. Given two sets of
nodes, (`source`, `target`), the ``ReversedKNNEdges`` method connects
all source nodes to their ``num_nearest_neighbours`` nearest target
nodes. The ``exclude_self_edges`` option is available when
``source`` and ``target`` are the same node set.

To use this method to build your connections, you can use the following
YAML configuration:

.. code:: yaml

   edges:
     -  source_name: source
        target_name: target
        edge_builders:
        - _target_: anemoi.graphs.edges.ReversedKNNEdges
          num_nearest_neighbours: 3

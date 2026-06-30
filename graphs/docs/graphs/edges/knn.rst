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

###############################
 Reversed K-Nearest Neighbours
###############################

The reversed k-nearest neighbours (``ReversedKNNEdges``) method is
similar to the standard KNN method, but instead establishes connections
based on the nearest neighbours of each source node. Given two sets of
nodes, (`source`, `target`), the ``ReversedKNNEdges`` method connects
all source nodes to their ``num_nearest_neighbours`` nearest target
nodes.

To use this method to build your connections, you can use the following
YAML configuration:

.. code:: yaml

   edges:
     -  source_name: source
        target_name: target
        edge_builders:
        - _target_: anemoi.graphs.edges.ReversedKNNEdges
          num_nearest_neighbours: 3

##############################
 Mutual K-Nearest Neighbours
##############################

The mutual k-nearest neighbours (``MutualKNNEdges``) method keeps an edge
between a source node and a target node only when the target is among the
source node's ``reversed_num_nearest_neighbours`` nearest target nodes **and**
the source is among the target node's ``num_nearest_neighbours`` nearest source
nodes. It is the intersection of the ``KNNEdges`` and ``ReversedKNNEdges`` edge
sets.

This is useful across a resolution transition — such as the global/regional
boundary of a stretched grid — where plain ``KNNEdges`` or ``CutOffEdges``
produce lopsided, one-sided coupling. Mutual edges taper connectivity
symmetrically and bound node degree on both sides. ``num_nearest_neighbours``
and ``reversed_num_nearest_neighbours`` may differ to control fan-in and
fan-out independently; when ``reversed_num_nearest_neighbours`` is omitted it
defaults to ``num_nearest_neighbours``.

To use this method to build your connections, you can use the following
YAML configuration:

.. code:: yaml

   edges:
     -  source_name: source
        target_name: target
        edge_builders:
        - _target_: anemoi.graphs.edges.MutualKNNEdges
          num_nearest_neighbours: 3
          reversed_num_nearest_neighbours: 3

Combined with the ``source_mask_attr_name`` / ``target_mask_attr_name``
arguments, it can be confined to the nodes crossing the global/regional
boundary of a stretched grid. A global data node deep in the global interior
is never a mutual neighbour of a regional node, so a single masked builder
produces only the genuine boundary edges:

.. code:: yaml

   edges:
     -  source_name: data
        target_name: hidden
        edge_builders:
        - _target_: anemoi.graphs.edges.MutualKNNEdges
          num_nearest_neighbours: 8
          reversed_num_nearest_neighbours: 2
          source_mask_attr_name: global_data_mask
          target_mask_attr_name: regional_hidden_mask

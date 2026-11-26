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

#######################
 Lookup-table search
#######################

By default the nearest neighbours are found with a brute-force search
(pyg-lib if installed, scikit-learn otherwise). ``KNNEdges``,
``ReversedKNNEdges`` and ``MutualKNNEdges`` accept ``backend: lookup``
to search through a lookup table instead:

.. code:: yaml

   edges:
     -  source_name: hidden
        target_name: data
        edge_builders:
        - _target_: anemoi.graphs.edges.KNNEdges
          num_nearest_neighbours: 3
          backend: lookup

The searched nodes are indexed once on a latitude/longitude grid of
cells. Each cell stores every node within
:math:`d_k + 2r` of its centre, where :math:`d_k` is the distance from
the centre to its k-th nearest node and :math:`r` the largest distance
from the centre to a point of the cell. By the triangle inequality this
contains the k nearest nodes of any point in the cell, so the result is
exact (ties may be broken differently). Each query then reads its
cell's handful of candidates, in a Triton kernel on CUDA.

The cell size is half the mean node spacing :math:`\sqrt{4\pi/N}`, so no
grid-specific formula is needed. The table is meant for node sets that
cover the globe (O-, N- and icosahedral grids, HEALPix, stretched
grids, ...); for regional node sets it raises an error, and the
``brute_force`` backend should be used instead.

Tables are cached by the name of the searched node set (and its mask,
if one is applied) together with ``num_nearest_neighbours``, so that
builders searching the same node set for the same number of neighbours
share one table. Each number of neighbours gets its own table: a cell
stores the candidates needed for the k it was built for, and every query
scans all of them, so serving a 3-neighbour search from a 24-neighbour
table would make it several times slower. If the node set under a cached
name changes, the table is rebuilt with a
warning. The backend is for node sets that stay fixed, such as a hidden
mesh searched by a graph that is rebuilt every batch; it checks the size
of the node set on every use, and its coordinates once per table, so a
name given to a node set that keeps changing is reported on its first
reuse rather than on every one.

Building the table is itself a brute-force search from every cell
centre, with a few cells per searched node, so its cost grows with the
square of the number of searched nodes. It pays off when there are many
more queries than searched nodes (for example millions of data or
observation points against a hidden mesh) or when the same node set is
searched repeatedly. A build expected to take more than about 30
seconds is announced with a warning.

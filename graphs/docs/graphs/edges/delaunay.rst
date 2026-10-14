.. _delaunay:

#################
 Delaunay Edges
#################

The ``DelaunayEdges`` method builds the connections within a single set
of nodes from a spherical Delaunay triangulation. It is the counterpart
of the multi-scale edge builders for meshes that have no regular
structure, such as the ones produced by :ref:`kmeans_clusters`.

The triangulation is computed as the convex hull of the nodes on the
unit sphere, and each triangle edge becomes a pair of edges, one in each
direction.

To use this method to build your connections, you can use the following
YAML configuration:

.. code:: yaml

   edges:
     -  source_name: hidden
        target_name: hidden
        edge_builders:
        - _target_: anemoi.graphs.edges.DelaunayEdges

.. note::

   This builder connects a set of nodes to itself, so ``source_name``
   and ``target_name`` must be the same. Use ``KNNEdges`` or
   ``CutOffEdges`` to connect two different sets of nodes.

The triangulation covers the whole domain spanned by the nodes,
including the parts of it that fall outside the area of interest. For an
irregular domain, such as the ocean, the triangulation will therefore
connect nodes across land barriers. Those edges can be removed
afterwards with the ``RemoveEdgesCrossingArea`` post-processor, and the
longest ones with ``RestrictEdgeLength``. See
:ref:`graphs-post_processor`.

.. code:: yaml

   edges:
     -  source_name: hidden
        target_name: hidden
        edge_builders:
        - _target_: anemoi.graphs.edges.DelaunayEdges

   post_processors:
   - _target_: anemoi.graphs.processors.RemoveEdgesCrossingArea
     source_name: hidden
     target_name: hidden
     reference_nodes_name: data
     reference_attribute_name: land_mask

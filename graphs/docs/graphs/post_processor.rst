.. _graphs-post_processor:

#################
 Post-processors
#################

The anemoi-graphs package provides an API to implement post-processors,
which are optional methods applied after a graph is constructed. These
post-processors allow users to modify or refine the graph to suit
specific use cases. They can be configured in the recipe file to enable
flexible and automated post-processing workflows.

************************
 RemoveUnconnectedNodes
************************

The ``RemoveUnconnectedNodes`` post-processor is designed to prune
unconnected nodes from a graph. This is particularly useful in scenarios
where disconnected nodes do not contribute to the analysis or where the
focus is limited to a specific subset of the graph.

One notable application of ``RemoveUnconnectedNodes`` is in Limited Area
Modeling (LAM), where a global dataset is often specified as a forcing
boundary, but the analysis is only concerned with nodes near the limited
area boundary. By pruning unconnected nodes, this post-processor ensures
the resulting graph is focused on the region of interest, making it more
efficient during training.

The ``RemoveUnconnectedNodes`` post-processor also provides
functionality to store the indices of the pruned nodes (mask). This
feature is particularly valuable for workflows involving training or
inference, as it enables users to repeat the same masking operation
consistently across different stages of analysis. To enable this
feature, the user can specify the ``save_indices_mask_attr`` parameter.
This parameter takes a string that represents the name of the new node
attribute where the masking indices will be stored.

.. code:: yaml

   nodes: ...
   edges: ...

   post_processors:
   - _target_: anemoi.graphs.processors.RemoveUnconnectedNodes
      nodes_name: data
      save_mask_indices_to_attr: indices_connected_nodes # optional

The ``RemoveUnconnectedNodes`` post-processor also supports an
``ignore`` argument, which is optional but highly convenient in certain
use cases. This argument corresponds to the name of a node attribute
used as a mask to prevent certain nodes from being dropped, even if they
are unconnected. For example, in LAM workflows, it may be necessary to
retain data nodes from the regional dataset that remain unconnected.

By specifying the ignore argument, users can ensure that such nodes are
preserved. For example:

.. code:: yaml

   nodes: ...
   edges: ...

   post_processors:
   - _target_: anemoi.graphs.processors.RemoveUnconnectedNodes
      nodes_name: data
      ignore: important_nodes
      save_mask_indices_to_attr: indices_connected_nodes # optional

In this configuration, any node with the attribute `important_nodes` set
will not be pruned, regardless of its connectivity status.

*******************
 SubsetNodesInArea
*******************

The ``SubsetNodesInArea`` post-processor is used to focus the graph on a
specific geographic region. It removes all nodes that fall outside the
user-defined area, as well as any edges connected to those nodes. This
is useful for restricting analysis or training to a particular spatial
domain, ensuring that only relevant nodes and their relationships are
retained in the graph.

.. code:: yaml

   nodes: ...
   edges: ...

   post_processors:
   - _target_: anemoi.graphs.processors.SubsetNodesInArea
      nodes_name: [data, hidden]
      area: [40, 10, 20, 30] # (north, west, south, east)

The area is defined by four values representing the northern, western,
southern, and eastern boundaries, in that order: `(north, west, south,
east)`. Only nodes within these boundaries will be retained in the
graph.

********************
 RestrictEdgeLength
********************

The ``RestrictEdgeLength`` post-processor will remove edges longer than
a certain treshold (set in km). This can be useful when one or multiple
edge builders create edges of various lenghts, some of which are
undesirable. For example when using ``KNNEdges`` applied to all of the
hidden mesh but only a subset of the data nodes (e.g. those in a LAM
region) also connections are made to hidden mesh nodes very far away
from the restricted set of data nodes. With this post-processor one can
remove such edges, effectively providing a ``KNNedges`` algorithm
applied only that part of the data mesh within a certain distance to the
restricted set of data nodes.

After the long edges are removed the edge attributes are recomputed,
since the removal of a large number of edges can change their
distribution.

.. code:: yaml

   nodes: ...
   edges: ...

   post_processors:
   - _target_: anemoi.graphs.processors.RestrictEdgeLength
     source_name: data                #source nodes of the edges to be processed
     target_name: hidden              #target nodes of the edges to be processed
     max_length_km: 20                #edges longer than the threshold of 20 km will be removed

The ``RestrictEdgeLength`` post-processor also supports the
``source_mask_attr_name`` and ``target_mask_attr_name`` arguments. These
are optional but allow to refer to a Boolean attribute of the
source/target nodes and only those edges whose source/target is ``True``
under this Boolean mask will be postprocessed. This can be useful if one
wants to exclude a subset of edges that are allowed to be longer than
the threshold. An example usage:

.. code:: yaml

   nodes: ...
      attributes:
         cutout:
            _target_: anemoi.graphs.nodes.attributes.CutOutMask
   edges: ...
   postprocessors:
   - _target_: anemoi.graphs.processors.RestrictEdgeLength
     source_name: data                #source nodes of the edges to be processed
     target_name: hidden              #target nodes of the edges to be processed
     max_length_km: 20                    #edges longer than this threshold (in km) will be removed
     source_mask_attr_name: cutout    #optional

With this configuration only edges whose source is in the cutout region
will be post-processed, i.e. those edges with source node outside the
cutout region will be preserved regardless of their length.

************************************
 Edge Index Sorting Post-processors
************************************

The anemoi-graphs package provides two post-processors for sorting edge
indices: ``SortEdgeIndexBySourceNodes`` and
``SortEdgeIndexByTargetNodes``. These processors help organize the edge
indices in a consistent order, which can be useful for deterministic
behavior and improved performance in certain operations.

SortEdgeIndexBySourceNodes
==========================

This post-processor sorts all edge indices based on the source nodes. It
can be configured to sort in either ascending or descending order:

.. code:: yaml

   post_processors:
   - _target_: anemoi.graphs.processors.SortEdgeIndexBySourceNodes
      descending: True  # optional, defaults to true

SortEdgeIndexByTargetNodes
==========================

Similar to the source node sorter, this post-processor sorts edge
indices based on the target nodes:

.. code:: yaml

   post_processors:
   - _target_: anemoi.graphs.processors.SortEdgeIndexByTargetNodes
      descending: True  # optional, defaults to true

Both processors maintain the consistency of all edge attributes while
sorting, ensuring that the relationship between edge indices and their
corresponding attributes remains intact.

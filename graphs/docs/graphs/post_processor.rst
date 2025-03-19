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
remove such edges, effectively providing a `KNNedges` algorithm applied
only that part of the data mesh within a certain distance to the
restricted set of data nodes.

The ``RestrictEdgeLength`` post-processor also provides functionality to
(re)compute edge attributes, since the removal of a large number of
edges can affect those, e.g. their normalisation.

.. code:: yaml

   nodes: ...
   edges: ...

   post_processors:
   - _target_: anemoi.graphs.processors.RestrictEdgeLength
    source_name: data                #source nodes of the edges to be processed
    target_name: hidden              #target nodes of the edges to be processed
    threshold: 20                    #edges longer than the threshold of 20 km will be removed
    update_attributes:               #update edge attributes after postprocessing (optional)
      edge_length:
        _target_: anemoi.graphs.edges.attributes.EdgeLength
        norm: unit-max

The configuration above asks to (re)compute the edge length attribute
after removal of edges longer than 20 km, something which can be
relevant since removing such longer edges will affect the normalisation
of the edge length attribute.

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
   threshold: 20                    #edges longer than this threshold (in km) will be removed
   source_mask_attr_name: cutout    #optional

With this configuration only edges whose source is in the cutout region
will be post-processed, i.e. those edges with source node outside the
cutout region will be preserved regardless of their length.

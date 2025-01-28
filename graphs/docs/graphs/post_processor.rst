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

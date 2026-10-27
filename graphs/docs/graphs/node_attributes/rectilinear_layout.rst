#########################
 Rectilinear grid layout
#########################

The `RectilinearLayoutIndex` node attribute builder gives the row and
column of each node in its rectilinear grid, as a tensor of shape
`(num_nodes, 2)` of integers.

.. code:: yaml

   nodes:
     hidden:
       node_builder:
         _target_: anemoi.graphs.nodes.LimitedAreaRectilinearNodes
         resolution: 0.25
         reference_node_name: data
         crop_to_margin: True
       attributes:
         layout_index:
           _target_: anemoi.graphs.nodes.attributes.RectilinearLayoutIndex

The size of the full grid is in the hidden node attribute
``_layout_shape``. Some grid points may not be nodes, for example with
``crop_to_margin``. A field on the nodes can be put back on the full
grid, with NaN at those grid points:

.. code:: python

   index = graph["hidden"]["layout_index"]
   grid = torch.full(graph["hidden"]["_layout_shape"], float("nan"))
   grid[index[:, 0], index[:, 1]] = values

The row and column stay correct when post-processors remove nodes.

.. note::

   This node attribute builder is only supported for nodes created with
   ``RectilinearNodes`` or ``LimitedAreaRectilinearNodes``.

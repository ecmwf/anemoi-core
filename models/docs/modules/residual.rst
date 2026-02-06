.. _residual-connections:

######################
 Residual connections
######################

The configurable residual connections link input data to output data.

The type of residual connection used in a model is specified
under the `residual` key in the model configuration YAML. This modular
approach allows users to select and customize the residual strategy best
suited for their forecasting task, whether it be a standard skip
connection, no connection, or a truncated connection.

The following classes implement the available residual connection types
in Anemoi.

*****************
 Skip Connection
*****************

.. autoclass:: anemoi.models.layers.residual.SkipConnection
   :members:
   :no-undoc-members:
   :show-inheritance:

**********************
 Truncated Connection
**********************

.. autoclass:: anemoi.models.layers.residual.TruncatedConnection
   :members:
   :no-undoc-members:
   :show-inheritance:

****************
 Configuration
****************

``TruncatedConnection`` supports two configuration styles:

- **Graph-based**: reference projection edges defined in
  ``config.graph.projections`` and build sparse matrices at runtime.
- **File-based**: load precomputed ``.npz`` matrices from disk.
Choose one style; do not mix graph-based and file-based settings in the
same config.

Graph-based example:

.. code:: yaml

   model:
     residual:
       _target_: anemoi.models.layers.residual.TruncatedConnection
       truncation_down_edges_name: ${graph.projections.residual.down_edges_name}
       truncation_up_edges_name: ${graph.projections.residual.up_edges_name}
       edge_weight_attribute: ${graph.projections.residual.edge_weight_attribute}

Gaussian distance weights computed with ``norm: l1`` should be used as
``edge_weight_attribute`` (commonly ``gauss_weight`` in the projection
graph). ``src_node_weight_attribute`` is optional and can be omitted for
truncation graphs.

File-based example:

.. code:: yaml

   model:
     residual:
       _target_: anemoi.models.layers.residual.TruncatedConnection
       truncation_down_file_path: ${system.input.truncation}
       truncation_up_file_path: ${system.input.truncation_inv}
       row_normalize: false

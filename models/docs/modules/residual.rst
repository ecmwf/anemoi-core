######################
 Residual connections
######################

Residual connections are a key architectural feature in Anemoi's
encoder-processor-decoder models, enabling more effective information
flow and gradient propagation across network layers. Residual
connections help mitigate issues such as vanishing gradients and support
the training of deeper, and more expressive models.

In Anemoi, the type of residual connection used in a model is specified
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

Use file-based matrices:

.. code-block:: yaml

   residual:
     _target_: anemoi.models.layers.residual.TruncatedConnection
     truncation_down_file_path: o96_to_o32.npz
     truncation_up_file_path: o32_to_o96.npz
     truncation_matrices_path: /path/to/matrices

Use a graph-based truncation definition:

.. code-block:: yaml

   residual:
     _target_: anemoi.models.layers.residual.TruncatedConnection
     truncation_graph:
       graph_config:
         nodes:
           data: ...
           trunc: ...
         edges:
           - source_name: data
             target_name: trunc
             edge_builders: [...]
             attributes:
               gauss_weight: ...
           - source_name: trunc
             target_name: data
             edge_builders: [...]
             attributes:
               gauss_weight: ...
         post_processors: []
       down_edges_name: [data, to, trunc]
       up_edges_name: [trunc, to, data]
       edge_weight_attribute: gauss_weight


.. autoclass:: anemoi.models.layers.residual.TruncatedConnection
   :members:
   :no-undoc-members:
   :show-inheritance:

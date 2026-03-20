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

.. autoclass:: anemoi.models.layers.residual.TruncatedConnection
   :members:
   :no-undoc-members:
   :show-inheritance:

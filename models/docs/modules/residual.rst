.. _residual-connections:

######################
 Residual connections
######################

Residual connections are a key architectural feature in Anemoi's
encoder-processor-decoder models, enabling more effective information
flow and gradient propagation across network layers. Residual
connections help mitigate issues such as vanishing gradients and support
the training of deeper, and more expressive models.

The configurable residual connections link input data to output data.
The type of residual connection used in a model is specified under the
``residual`` key in the model configuration YAML. This modular approach
allows users to select and customize the residual strategy best suited
for their forecasting task, whether it be a standard skip connection or
a truncated connection.

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

Both connection types are configured under the ``residual`` key in the
model config. ``TruncatedConnection`` accepts sibling-class kwargs such
as ``step`` transparently, so switching between connection types requires
only changing ``_target_``.

``TruncatedConnection`` supports two modes, both via the
``truncation_config`` key:

- **On-the-fly**: the truncation subgraph is built at runtime from the
  main graph using a coarser ``grid`` specification.
- **File-based**: precomputed ``.npz`` projection matrices are loaded
  from disk.

Choose one mode per config; do not mix the two within the same
``truncation_config`` block.

On-the-fly example:

.. code:: yaml

   model:
     residual:
       _target_: anemoi.models.layers.residual.TruncatedConnection
       truncation_config:
         grid: o32
         num_nearest_neighbours: 3
         sigma: 1.0

File-based example:

.. code:: yaml

   model:
     residual:
       _target_: anemoi.models.layers.residual.TruncatedConnection
       truncation_config:
         truncation_down_file_path: /path/to/O96-O32-grid-box-average.mat.npz
         truncation_up_file_path: /path/to/O32-O96-grid-box-average.mat.npz
         row_normalize: false

.. note::

   The top-level ``truncation_up_file_path`` and
   ``truncation_down_file_path`` kwargs are still accepted for backward
   compatibility, but the recommended approach is to move them inside ``truncation_config``.

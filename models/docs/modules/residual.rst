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

*****************************
 Standard residual (default)
*****************************

The standard residual formulation used in most models is:

.. math::

   x(t+1) = x(t) + f_\theta(x(t))

where :math:`f_\theta` is the learned model increment. This preserves
the full input state and adds a correction.

*****************
 Skip Connection
*****************

Returns the most recent timestep unchanged:

.. math::

   \text{residual}(x) = x_t

This is the default residual and corresponds to the standard formulation
above (the model output is added externally by the architecture).

.. autoclass:: anemoi.models.layers.residual.SkipConnection
   :members:
   :no-undoc-members:
   :show-inheritance:

**********************
 Truncated Connection
**********************

Projects the input to a coarser grid and back, removing high-frequency
content from the skip connection via sparse spatial projections:

.. math::

   \text{residual}(x) = P_{\text{up}} \, P_{\text{down}} \, x_t

where :math:`P_{\text{down}}` maps to the coarse grid and
:math:`P_{\text{up}}` maps back to the original resolution.

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

*****************************
 Learnable residual (Ornstein)
*****************************

Learnable residual connections introduce a trainable scaling parameter
:math:`\alpha` on the residual connection, giving a formulation
equivalent to a discretized Ornstein--Uhlenbeck process:

.. math::

   x(t+1) = \alpha \cdot x(t) + f_\theta(x(t))

With :math:`\alpha` trainable and :math:`\alpha < 1`, errors in the
state are contracted at each step rather than perfectly preserved. This
bounds error growth during autoregressive integration.

Two variants are available, offering increasing degrees of spatial
structure in the learnable parameters.

*****************************
 Scalar Ornstein Connection
*****************************

A single learnable scalar :math:`\alpha_v` per prognostic variable
:math:`v`:

.. math::

   \text{residual}(x)_v = (1 - \alpha_v) \cdot x_{t,v}

where :math:`\alpha_v \in (\alpha_{\text{buff}}, 1)` is parameterized
via a sigmoid. This is the simplest Ornstein variant -- no spatial
structure, just a per-variable damping factor.

.. autoclass:: anemoi.models.layers.residual.ScalarOrnsteinConnection
   :members:
   :no-undoc-members:
   :show-inheritance:

*******************************
 Spectral Ornstein Connection
*******************************

Spatially-varying :math:`\alpha` and bias :math:`\mu`, defined as smooth
functions on the sphere via spherical harmonic (SH) coefficients:

.. math::

   \text{residual}(x)_v = \bigl(1 - \alpha_v(s)\bigr) \cdot x_{t,v}
   + \mu_v(s) + \sum_i \beta_{i,v}(s) \cdot f_i

where :math:`s` denotes the spatial location, :math:`\alpha_v(s)`,
:math:`\mu_v(s)`, and :math:`\beta_{i,v}(s)` are reconstructed from
low-order SH coefficients (controlled by ``lmax``), and :math:`f_i` are
optional forcing regressors.

When ``truncate=True``, a learnable spectral low-pass filter is applied
to the input fields *before* computing the residual. This removes
high-frequency content from the skip connection, forcing the model to
reconstruct fine-scale detail from scratch. An optional anti-aliasing
blend (``anti_aliasing=True``) smoothly mixes the filtered and
unfiltered fields.

.. autoclass:: anemoi.models.layers.residual.SpectralOrnsteinConnection
   :members:
   :no-undoc-members:
   :show-inheritance:

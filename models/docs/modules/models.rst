########
 Models
########

The models module provides several neural network architectures that
work with graph input data and follow an encoder-processor-decoder
structure.

*********************************
 Encoder-Processor-Decoder Model
*********************************

The model defines a network architecture with configurable encoder,
processor, and decoder components (`Lang et al. (2024a)
<https://arxiv.org/abs/2406.01465>`_).

.. autoclass:: anemoi.models.models.encoder_processor_decoder.AnemoiModelEncProcDec
   :members:
   :no-undoc-members:
   :show-inheritance:

******************************************
 Ensemble Encoder-Processor-Decoder Model
******************************************

The ensemble model architecture implementing the AIFS-CRPS approach
`Lang et al. (2024b) <https://arxiv.org/abs/2412.15832>`_.

Key features:

#. Based on the base encoder-processor-decoder architecture
#. Injects noise in the processor for each ensemble member using
   :class:`anemoi.models.layers.normalization.ConditionalLayerNorm`

.. autoclass:: anemoi.models.models.ens_encoder_processor_decoder.AnemoiEnsModelEncProcDec
   :members:
   :no-undoc-members:
   :show-inheritance:

**********************************************
 Hierarchical Encoder-Processor-Decoder Model
**********************************************

This model extends the standard encoder-processor-decoder architecture
by introducing a **hierarchical processor**.

Key features:

#. Requires a predefined list of hidden nodes, `[hidden_1, ...,
   hidden_n]`

#. Nodes must be sorted to match the expected flow of information `data
   -> hidden_1 -> ... -> hidden_n -> ... -> hidden_1 -> data`

#. Supports hierarchical level processing through the
   `enable_hierarchical_level_processing` configuration. This argument
   determines whether a processor is added at each hierarchy level or
   only at the final level.

#. Channel scaling: `2^n * config.num_channels` where `n` is the
   hierarchy level

By default, the number of channels for the mappers is defined as `2^n *
config.num_channels`, where `n` represents the hierarchy level. This
scaling ensures that the processing capacity grows proportionally with
the depth of the hierarchy, enabling efficient handling of data.

.. autoclass:: anemoi.models.models.hierarchical.AnemoiModelEncProcDecHierarchical
   :members:
   :no-undoc-members:
   :show-inheritance:

*****************************************************
 Disentangled Encoder-Processor-Decoder Model
*****************************************************

A multi-dataset architecture where each dataset is encoded independently,
per timestep, and all resulting latent representations are blended before
processing.

Key features:

#. Each dataset uses its own encoder; timesteps are encoded one at a time
   (no stacking in the input dimension)

#. All encoded latents — across datasets and timesteps — are accumulated
   and concatenated, then fed to a learned ``latent_blender`` mapper that
   projects them onto the hidden graph

#. On ``rollout_step > 0`` (latent rollout), the latent buffer is shifted
   and the previous processor output fills the last slot, avoiding
   re-encoding from data space on every autoregressive step

#. Decoding is performed independently per dataset

This model is used together with the
:class:`~anemoi.training.train.tasks.forecaster.LatentGraphForecaster`
task, which passes the ``rollout_step`` index to the model at each
autoregressive step.

.. autoclass:: anemoi.models.models.disentangled_encprocdec.AnemoiModelDisentangledEncProcDec
   :members:
   :no-undoc-members:
   :show-inheritance:

*************************
 Time Interpolator Model
*************************

A specialized architecture for time interpolation tasks.

Key features:

   #. Ability to select time indices for forcing and predictions
   #. Allows for provision of t0 and t6 and predictions of t1->5

.. autoclass:: anemoi.models.models.interpolator.AnemoiModelEncProcDecMultiOutInterpolator
   :members:
   :no-undoc-members:
   :show-inheritance:

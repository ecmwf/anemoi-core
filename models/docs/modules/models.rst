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

Residual connections (including graph-based truncation) are configured in
the model config; see :ref:`residual-connections` for details.

This base model also encodes and decodes multiple datasets; see
:ref:`usage-multi-dataset` for the ``encoders``/``decoders``,
``latent_aggregator`` and decoder ``target_node_features`` options.

Model stages
============

``AnemoiModelEncProcDec`` is built and run in three stages, and every
other model in this module is a specialisation of one or more of them:

-  **encode**: ``_build_encoders`` / ``_encode``. Each input dataset is
   assembled by ``_assemble_input`` and mapped onto the first hidden
   level; the per-dataset latents are combined by the latent aggregator.
-  **process**: ``_build_processor`` / ``_process``. The latent is
   transformed on the deepest hidden level. ``_prepare_processor_input``
   is a smaller hook for models that alter the latent right before the
   processor (the ensemble model injects noise there); the hierarchical
   model wraps the whole stage in a U-Net over the hidden levels.
-  **decode**: ``_build_decoders`` / ``_decode``. The processed latent is
   mapped back onto every target dataset and finalised by
   ``_assemble_output``.

Per-call state (batch and ensemble sizes, sharding, the hidden latents,
the encoder by-products and any extra keyword arguments for the mappers
and the processor, such as the ``cond`` of the transport models) travels
in a ``ForwardContext`` created by ``_init_forward_context``. Models
that need extra forward inputs extend this dataclass.

Because the stages are orthogonal, variants combine through multiple
inheritance, for example
``class TransportHierarchical(AnemoiTransportModelEncProcDec, AnemoiModelEncProcDecHierarchical)``.
Note that the inter-level mappers and level processors of the
hierarchical model receive no conditioning, so their layer kernels must
be unconditional in such a combination.

Reproducing the ``AnemoiModelAutoEncoder`` (deprecated)
========================================================

The dedicated ``AnemoiModelAutoEncoder`` has been removed, it is now a
configuration of ``AnemoiModelEncProcDec``. The autoencoder reconstructs
its output from the input **forcings and coordinates** instead of the
encoded latent. Reproduce it with two changes:

#. In the **data** config, declare every variable as a *forcing* and/or a
   *diagnostic* — leave nothing prognostic. With no prognostic
   variables, the residual skip connection has no state to carry over
   and is effectively a no-op, so no special residual class is needed.
#. In the **model** config, set the decoder ``target_node_features`` to
   ``[forcings, coordinates]`` (the base model default is
   ``[encoded_data]``).

.. code:: yaml

   decoders:
     global:
       datasets: [era5]
       # AutoEncoder behaviour: reconstruct from forcings + coordinates.
       target_node_features: [forcings, coordinates]
       mapper:
         _target_: anemoi.models.layers.mapper.GraphTransformerBackwardMapper
         # ... mapper configuration

See :ref:`usage-multi-dataset` for the full list of target features.

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

For the training-side CRPS setup, including loss, truncation, and
ensemble-specific configuration changes, see
:ref:`anemoi-training:ensemble-crps-training`.

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

The transitions between hierarchy levels are configured with two
dedicated mappers, ``upscale_mapper`` and ``downscale_mapper``, in
addition to the ``encoders`` / ``decoders`` that map between the data
nodes and the first hidden level:

-  ``upscale_mapper``: maps from a lower level to a higher level in the
   hierarchy (a forward mapper).
-  ``downscale_mapper``: maps from a higher level back to a lower level
   (a backward mapper).

.. code:: yaml

   model:
     model:
       _target_: anemoi.models.models.AnemoiModelEncProcDecHierarchical
       hidden_nodes_name: [hidden_1, hidden_2, hidden_3]
     enable_hierarchical_level_processing: True
     level_process_num_layers: 2

     upscale_mapper:
       _target_: anemoi.models.layers.mapper.GraphTransformerForwardMapper
       # ... mapper configuration
     downscale_mapper:
       _target_: anemoi.models.layers.mapper.GraphTransformerForwardMapper
       # ... mapper configuration

.. autoclass:: anemoi.models.models.hierarchical.AnemoiModelEncProcDecHierarchical
   :members:
   :no-undoc-members:
   :show-inheritance:

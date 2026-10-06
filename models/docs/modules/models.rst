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

Paper-structured multidomain downscaling
=======================================

The dedicated transport branch also provides the checkpoint-compatible
models used by the multidomain analog downscaling campaign:

* ``anemoi.models.models.multidomain_transport.PaperStructuredMultidomainTransport``
  is a velocity network on a shared 100,000-node residual-factor mesh. It
  uses ``GraphTransformerProcessor``, domain graph providers, current-weather
  context and conditional flow-time normalization. Atmospheric and transformed
  precipitation models share this implementation with different factor counts.
* ``anemoi.models.models.multidomain_decoder.PaperMultidomainDecoder`` lifts
  factors through ``GraphTransformerBackwardMapper`` to a native regional grid.
  It retains the PCA skip, high-resolution static context, native convolutional
  refiners, isolated precipitation specialization and temperature-coupled
  saturation-humidity coordinate.

These are model components with explicit tensor/graph constructor contracts,
not drop-in ``AnemoiModelEncProcDec`` replacements or registered training
methods. Dataset/bank/PCA preparation, staged training orchestration and
ensemble inference assembly remain in the campaign repository. Publishing
these classes does not register the full campaign with the standard training
CLI. They use a fixed semantic channel union, not the arbitrary-variable
metadata query head. Flow time is integration time, not a forecast lead.

State-dict keys and tensor shapes are retained for existing checkpoints;
``load_state_dict(..., strict=True)`` does not require key remapping.
Precipitation inputs use the campaign's continuous log coordinate; the
decoder does not silently change its physical inverse. Humidity pairing and
scales must be derived from training data only.

.. autoclass:: anemoi.models.models.multidomain_transport.PaperStructuredMultidomainTransport
   :members:

.. autoclass:: anemoi.models.models.multidomain_decoder.PaperMultidomainDecoder
   :members:

.. autoclass:: anemoi.models.models.multidomain_decoder.SaturationHumidityCoordinate
   :members:

Residual connections (including graph-based truncation) are configured in
the model config; see :ref:`residual-connections` for details.

This base model also encodes and decodes multiple datasets; see
:ref:`usage-multi-dataset` for the ``encoders``/``decoders``,
``latent_aggregator`` and decoder ``target_node_features`` options.

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

.. _usage-multi-dataset:

##############################
 Multi-dataset model configs
##############################

The encoder-processor-decoder models can encode and decode **several
datasets at once**. Instead of a single ``encoder`` and ``decoder``, the
model configuration declares *named groups* of encoders and decoders,
each responsible for one or more datasets. The per-dataset latent
representations are combined by a **latent aggregator** before the
processor, and each decoder reconstructs its datasets from a
configurable set of **target features**.

*********************
 Data flow overview
*********************

.. code:: text

   dataset_1 ─┐                                            ┌─► dataset_a
   dataset_2 ─┤► encoder group A ─┐                 ┌──────┤
   dataset_3 ─┘                   │                 │      └─► dataset_b
                                  ├─► aggregator ─► processor ─► ...
   dataset_4 ───► encoder group B ┘                 └──────────► dataset_c

#. Each **encoder group** maps its datasets into the latent (hidden)
   space.
#. The **latent aggregator** merges the per-dataset latents into a
   single latent tensor fed to the processor.
#. The **processor** operates on the aggregated latent.
#. Each **decoder group** reconstructs its datasets from the processed
   latent plus its configured target features.

***********************
 Encoders and decoders
***********************

``encoders`` and ``decoders`` are dictionaries keyed by a
**user-defined group name**. That name is arbitrary but meaningful: it
appears in the model ``state_dict`` (and therefore in checkpoints), so
choose stable, descriptive names.

.. code:: yaml

   encoders:
     global:                 # user-defined group name (appears in the state-dict)
       datasets: [ "era5", "ifs"]  # datasets encoded by this group
       dataset_fusing_strategy: "not_supported"
       mapper:
         _target_: anemoi.models.layers.mapper.GraphTransformerForwardMapper
         num_channels: 1024
         # ... mapper configuration
     regional:
       datasets: [ "cerra" ]
       dataset_fusing_strategy: "not_supported"
       mapper:
         _target_: anemoi.models.layers.mapper.GraphTransformerForwardMapper
         num_channels: 1024
         # ... mapper configuration

   decoders:
     global:
       datasets: [ "era5" ]
       input_target_features: [encoded_data]
       mapper:
         _target_: anemoi.models.layers.mapper.GraphTransformerBackwardMapper
         num_channels: 1024
         # ... mapper configuration
     regional:
       datasets: [ "cerra" ]
       input_target_features: [encoded_data]
       mapper:
         _target_: anemoi.models.layers.mapper.GraphTransformerBackwardMapper
         num_channels: 1024
         # ... mapper configuration

``datasets``
   The list of dataset names handled by this group. Datasets sharing a
   group share the same mapper weights.

``dataset_fusing_strategy``
   How multiple datasets within a single group are combined if passed during the
   same forward pass. Currently only ``"not_supported"`` is available; it is a
   placeholder for future fusing strategies.

``num_channels``
   Note that ``num_channels`` is configured **per mapper** (and on the
   processor), not once at the top level of the model schema.

.. note::

   All datasets encoded by the same encoder group must produce latents
   with consistent shapes (see the target-feature validation below).

*******************
 Latent aggregator
*******************

The ``latent_aggregator`` merges the per-dataset latent tensors produced
by the encoders into a single latent tensor for the processor.

.. code:: yaml

   latent_aggregator:
     _target_: anemoi.models.layers.aggregator.SumAggregator

Available aggregators (in ``anemoi.models.layers.aggregator``):

``SumAggregator``
   Element-wise sum of the latents. Requires all latents to share the
   same channel dimension. Zero-parameter; a single dataset is passed
   through unchanged.

``MeanAggregator``
   Element-wise mean of the latents. Also requires a common channel
   dimension.

``ConcatAggregator``
   Concatenation along the channel dimension. The aggregated channel
   dimension is the **sum** of the per-dataset channel dimensions, so
   the processor ``num_channels`` must be sized accordingly.

*************************
 Decoder target features
*************************

Each decoder group builds its input from an ordered list of
``input_target_features``. This controls what the decoder receives in
addition to (or instead of) the processed latent, and is the mechanism
that replaces bespoke model subclasses such as the former autoencoder.

.. code:: yaml

   decoders:
     global:
       datasets: [ "era5" ]
       input_target_features: [ "encoded_data" ]  # default

Valid features:

``encoded_data`` *(default)*
   The encoder-updated data tensor for the dataset (the encoded latent
   on the data nodes). Requires the dataset to have an encoder. It can
   be that the encoder doesn't update the data nodes, in which case the
   feature is simply the input data.

``coordinates``
   Sin/cos encoded lat-lon coordinates of the output nodes.

``forcings``
   Forcing variables over the input timestep window.

``prognostics``
   Prognostic variables over the input timestep window.

``trainable_parameters``
   Learnable per-node parameters. Requires
   ``trainable_parameters.data > 0`` for the dataset.

Features are concatenated in the listed order. All datasets handled by
the same decoder group must yield the same target features.
Mismatches are reported at model-initialisation time rather than on the
first batch.

.. tip::

   New target features can be registered from user code with the
   :func:`anemoi.models.models.target_features.register_target_feature`
   decorator on a :class:`~anemoi.models.models.target_features.DecodingTargetFeature`
   subclass.

********************************
 Per-dataset bounding and masks
********************************

``bounding``, ``output_mask`` and ``residual`` are configured
**per dataset** under a ``datasets`` key:

.. code:: yaml

   residual:
     datasets:
       era5:
         _target_: anemoi.models.layers.residual.SkipConnection
         step: -1
       cerra:
         _target_: anemoi.models.layers.residual.SkipConnection
         step: -1

   output_mask:
     datasets:
       era5:
         _target_: anemoi.training.utils.masks.NoOutputMask
       cerra:
         _target_: anemoi.training.utils.masks.NoOutputMask

   bounding:
     datasets:
       era5:
         - _target_: anemoi.models.layers.bounding.ReluBounding
           variables: [tp]

.. warning::

   If a bounding layer references a variable that is not present in the
   dataset's ``name_to_index`` mapping, a ``KeyError`` is raised at
   construction time.


A complete example is available in
``training/src/anemoi/training/config/model/graphtransformer_multi.yaml``.

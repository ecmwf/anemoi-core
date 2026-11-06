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
       source_datasets: [ "era5", "ifs"]  # datasets encoded by this group
       dataset_fusing_strategy: "sequential"
       node_embedder:
         _target_: anemoi.models.layers.node_embedder.LinearNodeEmbedder
         num_channels: 512   # input channels of this group's mapper
       mapper:
         _target_: anemoi.models.layers.mapper.GraphTransformerForwardMapper
         num_channels: 1024
         # ... mapper configuration
     regional:
       source_datasets: [ "cerra" ]
       dataset_fusing_strategy: "none"
       node_embedder:                # pass-through: the mapper takes cerra's input as is
         _target_: anemoi.models.layers.node_embedder.NoOpNodeEmbedder
       mapper:
         _target_: anemoi.models.layers.mapper.GraphTransformerForwardMapper
         num_channels: 1024
         # ... mapper configuration

   decoders:
     global:
       target_datasets: [ "era5" ]
       target_node_features: [ "encoded_data" ]
       mapper:
         _target_: anemoi.models.layers.mapper.GraphTransformerBackwardMapper
         num_channels: 1024
         # ... mapper configuration
     regional:
       target_datasets: [ "cerra" ]
       target_node_features: [ "encoded_data" ]
       mapper:
         _target_: anemoi.models.layers.mapper.GraphTransformerBackwardMapper
         num_channels: 1024
         # ... mapper configuration

``{source|target}_datasets``
   The list of dataset names handled by this group. Datasets sharing a
   group share the same mapper weights.

``dataset_fusing_strategy``
   How the source datasets of an encoder group are combined (see
   :ref:`usage-multi-dataset-shared-groups`):

   - ``"none"`` (default): no fusion; the node embedder is the pass-through
     ``NoOpNodeEmbedder``. The mapper takes the assembled input of a source dataset directly, so its input channels are
     that dataset's assembled input channels. Normally the group lists a single dataset. Several are
     allowed only if they have the same number of input channels and each batch holds at most
     one of them (multi-domain training). Training checks this against the
     data loading policy before it builds the model, and the model raises if a
     batch ever holds two.
   - ``"sequential"``: a projecting ``node_embedder`` maps every source dataset onto
     ``node_embedder.num_channels``, the input channels of the mapper. Each dataset is
     then encoded in its own pass with the shared mapper weights, in the order
     listed in ``source_datasets``, and the latent aggregator combines the
     latents. The datasets may differ in their number of variables. This also works with a single
     source dataset. The node embedder aligns node features only: the encoder edges of
     all source datasets must still carry the same number of edge features
     (see :ref:`usage-multi-dataset-edge-features`).
   - ``"joint"``: the ``node_embedder`` maps every source dataset onto the mapper's
     input channels, and all source datasets are encoded in a single mapper pass
     over the union of their nodes, so each hidden node attends to every source
     dataset at once. ``NoOpNodeEmbedder`` is accepted if the datasets share their
     number of input channels; otherwise use a projecting one such as
     ``LinearNodeEmbedder``. See :ref:`usage-multi-dataset-shared-groups`.

``node_embedder``
   The per-dataset input embedding applied before the encoder mapper, named by
   ``_target_``. Two node embedders are available in
   ``anemoi.models.layers.node_embedder``:

   - ``NoOpNodeEmbedder`` passes each source dataset's input through unchanged,
     much like ``NoOpProcessor`` stands in for a processor. It is the node embedder
     of ``"none"`` groups and the default. ``null`` is read as
     ``NoOpNodeEmbedder`` too, and any other node embedder set on a ``"none"``
     group is ignored with a warning. It has no parameters.
   - ``LinearNodeEmbedder`` applies one linear layer per source dataset onto a
     positive ``num_channels``. ``"sequential"`` groups need it (or another
     projecting node embedder); ``NoOpNodeEmbedder`` is rejected there.
     ``"joint"`` groups take either. It uses
     the mapper's ``layer_kernels`` unless it sets its own. Its weights appear in
     the state dict as ``encoder_src_projection.<group>.<dataset>.*``.

``num_channels``
   Note that ``num_channels`` is configured **per mapper** (and on the
   processor), not once at the top level of the model schema.

.. note::

   All datasets encoded by the same encoder group must produce latents
   with consistent shapes (see the target-feature validation below).

.. _usage-multi-dataset-shared-groups:

*************************************************************
 Sharing an encoder or decoder between datasets
*************************************************************

Several datasets can be listed in the same encoder (``source_datasets``) or
decoder (``target_datasets``) group. They then share the mapper weights, while
each dataset keeps its own graph edges.

With ``"sequential"``, when several datasets of a group are present in the
same batch, each is projected by the ``node_embedder``, encoded separately with the
shared weights, and the latents are merged by the
:ref:`latent aggregator <usage-latent-aggregator>`.

With ``"none"`` the node embedder passes the input through (``NoOpNodeEmbedder``), and a batch may hold only one of the group's
datasets. Training rejects a ``"none"`` group whose datasets the loading policy
puts in the same batch (``SynchronizedLoadingPolicy`` puts every dataset in every
batch) before it builds the model. Together with ``MultiDomainLoadingPolicy`` (one domain per batch) this
lets domains with different grids, but the same variables, be trained with one
set of weights. Use ``"sequential"`` instead if the domains differ in their number of variables.

With ``"joint"`` the per-dataset encoder edges are merged into one bipartite
graph and all source datasets are encoded in a single pass; the encoder
contributes one latent, keyed by the encoder name, to the latent aggregator.
Gridded and tabular sources can be mixed. The encoder graphs must have edges
(edge-less and sparse-projection mappers are not supported), and an encoder
with ``shard_strategy: heads`` needs the same merged node count in every batch
element, so ragged (tabular) sources need ``shard_strategy: edges``. With
``MultiDomainLoadingPolicy`` only the datasets of the batch's domain are
encoded, so a ``"joint"`` group merges sources on one hidden mesh at a time.

.. _usage-multi-dataset-edge-features:

Edge features must match
========================

The datasets of a shared encoder group are encoded by one mapper, whose
edge-embedding weights expect a fixed number of edge features. The
``<dataset> -> hidden`` encoder edges of every source dataset in the group must
therefore carry the same number of edge features. With ``"sequential"`` this
holds even though the datasets' node features may differ: the ``node_embedder``
projects node features only, and not edge features. The same applies to a
``"none"`` group with several datasets.

The number of edge features comes from the graph, not from the model:

- a gridded dataset (static graph) contributes the columns of the edge
  attributes named in the mapper's ``sub_graph_edge_attributes``, plus the
  mapper's ``trainable_size``;
- an observation dataset (dynamic graph) contributes the columns of every
  attribute declared on its edges in the graph config, and no trainable
  parameters.

For example, ``edge_length`` (1 column) and ``edge_dirs`` (2 columns) give 3
features. If the mapper sets ``trainable_size: 8``, a gridded dataset gets 11
and an observation dataset 3, so the two cannot share an encoder. To fuse them,
set the mapper's ``trainable_size`` to 0 and declare the same attributes on
each dataset's encoder edges, or give the datasets separate encoder groups.
The model checks this when it is built and names the edge feature count of
each source dataset if they differ.

.. _usage-multi-dataset-hidden-nodes:

********************
 Hidden node names
********************

``model.hidden_nodes_name`` defines the hidden mesh(es) of the model and
accepts three forms:

``str``
   One hidden mesh used by all datasets.

``list[str]``
   The hidden meshes of the levels of a hierarchical model.

``dict[str, str]``
   A hidden mesh per dataset. Several datasets may point to the same mesh. Only
   valid for multi-domain models, where a different hidden mesh is considered
   at each step.

   .. code:: yaml

      model:
        model:
          hidden_nodes_name:
            era5_o96: hidden
            cerra: hidden_cerra

The following rules apply to the mapping form:

- Every dataset of the model must be listed, and no other names may appear.
- All datasets present in one batch must resolve to the same hidden mesh, so a
  batch may not mix datasets with different meshes (use
  ``MultiDomainLoadingPolicy``, which keeps every batch to one domain).
- All hidden meshes must have the same node attribute dimension, because the
  processor weights are shared.
- Hierarchical models do not support the mapping.

The processor graph provider of flat models holds one entry per hidden mesh, so the hidden
mesh names appear in the model ``state_dict`` (and therefore in checkpoints), in
the same way as the encoder and decoder group names.

.. _usage-latent-aggregator:

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
   the processor ``num_channels`` must be sized accordingly. Channels are
   concatenated in the configured dataset order, independently of
   the order of the mapping passed to the model. Every configured dataset
   must be active because omitting one would change the output width.

``PointwiseCrossAttentionAggregator``
   Pointwise cross-attention over the active dataset latents. The
   initial hidden-node attributes provide the query and residual, while
   each dataset latent is projected to ``num_channels`` and used as a
   named key/value source. Encoder widths may differ and active dataset
   subsets are supported. The output width is ``num_channels`` and must
   match the processor width.

   .. code:: yaml

      latent_aggregator:
        _target_: anemoi.models.layers.aggregator.PointwiseCrossAttentionAggregator
        num_channels: ${model.num_channels}
        num_heads: 16
        attn_channels: null
        dropout_p: 0.0
        qkv_bias: false
        qk_norm: false
        gradient_checkpointing: true
        layer_kernels: ${model.layer_kernels}

*********************************
  Decoder target node features
*********************************

Each decoder group builds its input from an ordered list of
``target_node_features``. These are features defined on the decoder's
target (output) grid and supplied as inputs to the decoder. They control
what the decoder receives in addition to the processed latent representation.

.. code:: yaml

   decoders:
     global:
       datasets: [ "era5" ]
       target_node_features: [ "encoded_data" ]  # default

.. note::
   Here, __"target"__ refers to the target grid of the decoder, i.e., the grid on which the decoder produces its output.
   It does not refer to the targets used to compute the loss of the ML model.

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
   :func:`anemoi.models.layers.target_features.register_target_feature`
   decorator on a :class:`~anemoi.models.layers.target_features.DecodingTargetFeature`
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

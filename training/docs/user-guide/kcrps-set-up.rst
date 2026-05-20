##############################
 Ensemble CRPS-based training
##############################

This guide is intended for users who want to train an ensemble
CRPS-based model and are already familiar with the basic training
configurations.

It focuses on the changes relative to deterministic training. Detailed
reference material for graph-based truncation, multiscale loss
configuration, and residual projections lives in
:ref:`Field Truncation <usage-field_truncation>`,
:ref:`multiscale-loss-functions`, and
:ref:`anemoi-models:residual-connections`.

The CRPS training requires the following changes to the deterministic
training:

.. list-table:: Comparison of components between deterministic and CRPS training.
   :widths: 30 35 35
   :header-rows: 1

   -  -  Component
      -  Deterministic
      -  CRPS

   -  -  Forecaster
      -  :class:`GraphForecaster`
      -  :class:`GraphEnsForecaster`

   -  -  Strategy
      -  :class:`DDPGroupStrategy`
      -  :class:`DDPEnsGroupStrategy`

   -  -  Training loss
      -  :class:`MSELoss`
      -  :class:`AlmostFairKernelCRPS`

   -  -  Model
      -  :class:`AnemoiModelEncProcDec`
      -  :class:`AnemoiEnsModelEncProcDec`

   -  -  Datamodule
      -  :class:`AnemoiDatasetsDataModule`
      -  :class:`AnemoiDatasetsDataModule`

**************************
 Changes in System config
**************************

.. literalinclude:: yaml/example_crps_config.yaml
   :language: yaml
   :start-after: # Changes in system
   :end-before: num_gpus_per_ensemble:

The `truncation` and `truncation_inv` can be used in the deterministic
or CRPS training. As described in :ref:`Field Truncation
<usage-field_truncation>`, truncation smooths the skipped connection and
can also be reused for multiscale loss computation.

.. literalinclude:: yaml/example_crps_config.yaml
   :language: yaml
   :start-after: truncation_inv:
   :end-before: # Changes in datamodule

Graph-based truncation is also supported via
``graph.projections.truncation`` together with
``model.residual: TruncatedConnection``. The canonical graph-based and
file-based examples are documented in
:ref:`Field Truncation <usage-field_truncation>` and
:ref:`anemoi-models:residual-connections`.

The CRPS training uses a different DDP strategy which requires to
specify the number of GPUs per ensemble.

******************************
 Changes in datamodule config
******************************

.. literalinclude:: yaml/example_crps_config.yaml
   :language: yaml
   :start-after: # Changes in datamodule
   :end-before: data:

The `datamodule` needs to be set to
:class:`AnemoiEnsDatasetsDataModule`.
:class:`AnemoiEnsDatasetsDataModule` can be used with a single initial
condition for all ensembles or with perturbed initial conditions. The
perturbed initial conditions need to be part of your dataset.

*************************
 Changes in model config
*************************

The config group for the model is set to `transformer_ens.yaml`, which
specifies the :class:`AnemoiEnsModelEncProcDec` class with the Graph
Transformer encoder/decoder and a transformer processor.

Changes in `transformer_ens.yaml` with respect to `transformer.yaml`
are:

.. code:: yaml

   model:
      model:
         _target_: anemoi.models.models.ens_encoder_processor_decoder.AnemoiEnsModelEncProcDec

A different model class is used for CRPS training.

.. code:: yaml

   noise_injector:
      _target_: anemoi.models.layers.ensemble.NoiseConditioning
      noise_std: 1
      noise_channels_dim: 4
      noise_mlp_hidden_dim: 32
      inject_noise: True

Each ensemble member samples random noise at every time step. The noise
is embedded and injected into the latent space of the processor using a
conditional layer norm.

Noise can optionally be projected before conditioning:

- ``noise_matrix`` loads a precomputed sparse matrix from disk.
- ``noise_edges_name`` points to a graph edge type that maps a custom
  source node set to the hidden grid.

In the graph-based form, define the source nodes and corresponding
source-to-hidden edge in the graph config, then point the noise injector
to that edge:

.. code:: yaml

   graph:
      nodes:
         noise:
            node_builder:
               _target_: anemoi.graphs.nodes.ReducedGaussianGridNodes
               grid: o32
      edges:
         - source_name: noise
           target_name: hidden
           edge_builders:
              - _target_: anemoi.graphs.edges.KNNEdges
                num_nearest_neighbours: 32
           attributes:
              gauss_weight:
                 _target_: anemoi.graphs.edges.attributes.GaussianDistanceWeights
                 norm: l1
                 sigma: 0.1

   model:
      noise_injector:
         _target_: anemoi.models.layers.ensemble.NoiseConditioning
         noise_std: 1
         noise_channels_dim: 4
         noise_mlp_hidden_dim: 32
         noise_edges_name: [noise, to, hidden]
         edge_weight_attribute: gauss_weight

In order to condition the latent space on the noise, we need to use a
different layer norm in the processor, here the
:class:`anemoi.models.layers.normalization.ConditionalLayerNorm`.
See :doc:`anemoi-models:modules/models` for the ensemble model
architecture and :doc:`anemoi-models:modules/normalization` for the
normalization layers.

****************************
 Changes in training config
****************************

.. literalinclude:: yaml/example_crps_config.yaml
   :language: yaml
   :start-after: # Changes in training
   :end-before: # Changes in strategy

The model task is set to
:class:`anemoi.training.train.tasks.GraphEnsForecaster` for CRPS
training to deal with the ensemble members. The number of ensemble
members per device needs to be specified.

.. note::

   The total number of ensemble members is the product of the
   `ensemble_size_per_device` and the ratio of `num_gpus_per_ensemble` to `num_gpus_per_model` .

.. literalinclude:: yaml/example_crps_config.yaml
   :language: yaml
   :start-after: # Changes in strategy
   :end-before: # Changes in training loss

The CRPS training uses a different :ref:`Strategy` which allows to
parallelise the training over the ensemble members and shard the model.

.. literalinclude:: yaml/example_crps_config.yaml
   :language: yaml
   :start-after: # Changes in training loss
   :end-before: # Changes in validation metrics

We need to specify the loss function for the CRPS training. Here, we use
the :class:`anemoi.training.losses.kcrps.AlmostFairKernelCRPS` loss
function (`Lang et al. (2024b) <https://arxiv.org/abs/2412.15832>`_):

.. math::

   \text{afCRPS}_\alpha := \alpha\text{fCRPS} + (1-\alpha)\text{CRPS}

The `alpha` parameter is a trade-off parameter between the CRPS and the
fair CRPS.

If you want multiscale CRPS training, the reference documentation for
``MultiscaleLossWrapper`` and the two supported ``loss_matrices_graph``
forms is in :ref:`multiscale-loss-functions`.

.. literalinclude:: yaml/example_crps_config.yaml
   :language: yaml
   :start-after: # Changes in validation metrics
   :end-before: diagnostics:

Typically, the validation metrics are the same as the training loss, but
different validation metrics can be added here (see :ref:`Losses`).

****************
 Example config
****************

A typical config file for CRPS training is:

.. literalinclude:: yaml/example_crps_config.yaml
   :language: yaml

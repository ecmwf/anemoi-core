.. _training-methods target:

###################
 Training Methods
###################

The **training method** is the PyTorch Lightning module that implements
the forward pass, loss computation, and metric calculation. It is
separate from the task: the task says *what* time steps to load; the
method says *how* to train on them. Methods are configured via Hydra
under ``training.training_method``.

All methods inherit from
:class:`~anemoi.training.train.methods.base.BaseTrainingModule`, which
provides distributed training, loss scaling, normalization, and
validation metric hooks.

The three built-in methods are:

``SingleTraining`` (``anemoi.training.train.methods.single``)
   Deterministic single-member training. Suitable for ``Forecaster``,
   ``TemporalDownscaler``, and ``Autoencoder`` tasks. Uses
   ``DDPGroupStrategy`` for distributed execution.

``EnsembleTraining`` (``anemoi.training.train.methods.ensemble``)
   Ensemble (multi-member) training. Generates ``ensemble_size_per_device``
   members per device during training. Uses ``DDPEnsGroupStrategy`` for
   distributed execution.

   .. code:: yaml

      training:
        ensemble_size_per_device: 4

``TransportTraining`` (``anemoi.training.train.methods.transport``)
   Configurable transport training for EDM diffusion and flow-matching
   probabilistic forecasters. Use ``training.transport.prediction_mode``
   to select state or tendency targets, and
   ``training.transport.objective`` to select ``edm_diffusion`` or
   ``flow_matching``.

.. note::

   ``EnsembleTraining`` and transport objectives such as EDM diffusion or
   flow matching require the GNN model type to be replaced with a
   compatible architecture (e.g. GraphTransformer). The plain GNN
   processor is not supported for these methods.


.. _ensemble-crps-training:

******************************
 Ensemble CRPS-based training
******************************

This section is intended for users who want to train an ensemble
CRPS-based model and are already familiar with the basic training
configurations.

The CRPS training requires the following changes to the deterministic
training:

.. list-table:: Comparison of components between deterministic and CRPS training.
   :widths: 30 35 35
   :header-rows: 1

   -  -  Component
      -  Deterministic
      -  CRPS

   -  -  Training method
      -  :class:`SingleTraining`
      -  :class:`EnsembleTraining`

   -  -  Strategy
      -  :class:`DDPGroupStrategy`
      -  :class:`DDPEnsGroupStrategy`

   -  -  Training loss
      -  :class:`MSELoss`
      -  :class:`CRPS`

   -  -  Model
      -  :class:`AnemoiModelEncProcDec`
      -  :class:`AnemoiEnsModelEncProcDec`


Changes in System config
========================

.. literalinclude:: yaml/example_crps_config.yaml
   :language: yaml
   :start-after: # Changes in system
   :end-before: num_gpus_per_ensemble:

The ``truncation`` and ``truncation_inv`` entries are optional and are not
required for CRPS training. They are used by the file-based form of
:class:`anemoi.models.layers.residual.TruncatedConnection`; the graph-based
form builds its projections at startup. See
:ref:`Field Truncation <usage-field_truncation>` and
:ref:`anemoi-models:residual-connections` for the available truncation forms.

For a multiscale training objective, prefer
:ref:`MultiscaleLossWrapper <multiscale-loss-functions>`.

.. literalinclude:: yaml/example_crps_config.yaml
   :language: yaml
   :start-after: truncation_inv:
   :end-before: data:

The CRPS training uses a different DDP strategy which requires to
specify the number of GPUs per ensemble.


Changes in model config
=======================

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

Optionally, noise can be generated on a coarser grid and projected to
the processor grid before conditioning. Use one of these configuration
forms:

- ``noise_matrix`` loads a precomputed sparse matrix from a ``.npz``
  file created with ``anemoi-graphs export_to_sparse`` (see
  :ref:`usage-create_sparse_matrices`).
- ``noise_edges_name`` identifies a graph edge type that maps a custom
  source node set to the hidden grid.

For graph-based projection, define the source nodes and their edge to
the hidden grid, then configure the noise injector to use that edge:

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

``row_normalize_noise_matrix`` and ``autocast`` control how either
projection form is applied.

.. code:: yaml

   layer_kernels:
      processor:
         LayerNorm:
            _target_: anemoi.models.layers.normalization.ConditionalLayerNorm
            normalized_shape: ${model.num_channels}
            condition_shape: ${model.noise_injector.noise_channels_dim}
            zero_init: True
            autocast: false
         ...

In order to condition the latent space on the noise, we need to use a
different layer norm in the processor, here the
:class:`anemoi.models.layers.normalization.ConditionalLayerNorm`.

Changes in training config
==========================

.. literalinclude:: yaml/example_crps_config.yaml
   :language: yaml
   :start-after: # Changes in training
   :end-before: # Changes in strategy

The training method is set to
:class:`anemoi.training.train.methods.EnsembleTraining` for CRPS
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
the :class:`anemoi.training.losses.CRPS` loss function (`Lang et
al. (2024b) <https://arxiv.org/abs/2412.15832>`_):

.. math::

   \text{afCRPS}_\alpha := \alpha\text{fCRPS} + (1-\alpha)\text{CRPS}

The `alpha` parameter is a trade-off parameter between the CRPS and the
fair CRPS.
``alpha=0`` gives standard CRPS, ``alpha=1`` gives fair CRPS, and values
between 0 and 1 give the almost fair CRPS formulation. By default,
``alpha: 0.95`` gives a 5% standard CRPS and 95% fair CRPS combination.
The ``backend`` parameter selects how the score is computed:

- ``naive``: simple loop over unordered ensemble-member pairs, avoiding
  materialization of the full pairwise tensor.
- ``stable``: materializes pairwise tensors and uses the numerically stable
  all-pairs formulation.

.. literalinclude:: yaml/example_crps_config.yaml
   :language: yaml
   :start-after: # Changes in validation metrics

Typically, the validation metrics are the same as the training loss, but
different validation metrics can be added here (see :ref:`Losses`).

CRPS example config
===================

A typical config file for CRPS training is:

.. literalinclude:: yaml/example_crps_config.yaml
   :language: yaml


.. _diffusion-training:

*****************************
 Transport objective training
*****************************

Transport training covers probabilistic objectives that corrupt an
endpoint and train a model to recover either the clean endpoint or the
transport vector field. The supported objectives are ``edm_diffusion`` and
``flow_matching``.

Use :class:`~anemoi.training.train.methods.transport.TransportTraining`
with ``prediction_mode: state`` for state-space targets or
``prediction_mode: tendency`` for tendency-space targets. The model must
use :class:`AnemoiTransportModelEncProcDec` or
:class:`AnemoiTransportTendModelEncProcDec`.

.. warning::

   The plain GNN model is not supported for transport objective training.

Top-level configs
=================

The transport entry points are:

-  ``transport_edm_diffusion.yaml``
-  ``transport_edm_diffusion_tendency.yaml``
-  ``transport_flow_matching.yaml``
-  ``transport_flow_matching_tendency.yaml``

and, for two datasets (``era5`` and ``cerra``) trained together:

-  ``multi_transport_edm_diffusion.yaml``
-  ``multi_transport_edm_diffusion_tendency.yaml``
-  ``multi_transport_flow_matching.yaml``
-  ``multi_transport_flow_matching_tendency.yaml``

These configs select ``training: edm_diffusion`` or
``training: flow_matching`` (``multi_edm_diffusion`` or
``multi_flow_matching`` for two datasets), which set the corresponding
``training.transport.objective``. Tendency variants additionally select
``training.transport.prediction_mode: tendency`` and a tendency model
config.

The entry points select objective-specific model configs such as
``graphtransformer_transport_edm``, ``graphtransformer_transport_fm``,
``graphtransformer_transport_tendency_edm``, and
``graphtransformer_transport_tendency_fm``. Transformer variants use the
same ``_edm`` and ``_fm`` suffixes.

Model configuration
===================

Objective-specific model configs put transport settings under
``model.model.transport``:

.. code:: yaml

   model:
     _target_: anemoi.models.models.AnemoiTransportModelEncProcDec
     transport:
       objective: edm_diffusion
       sigma_data: 1.0
       sigma_max: 100.0
       sigma_min: 0.02
       rho: 7.0
       training_condition:
         distribution: karras
         sigma_max: ${model.model.transport.sigma_max}
         sigma_min: ${model.model.transport.sigma_min}
         rho: ${model.model.transport.rho}
       noise_embedder:
         _target_: anemoi.models.layers.diffusion.SinusoidalEmbeddings
         num_channels: ${model.model.transport.noise_channels}
         max_period: 1000

Both objectives start from Gaussian noise.

For flow matching, set ``objective: flow_matching``:

.. code:: yaml

   model:
     model:
       transport:
         objective: flow_matching
         noise_embedder:
           scale: 1000
         training_condition:
           distribution: uniform_time

Flow-matching time runs from 0 to 1, so the ``*_fm`` configs set
``scale: 1000``: the embedding frequencies then run from 1000 down to about
1.5 radians per unit of time.

Flow matching
=============

Flow matching follows a straight path from the Gaussian source to the
training target:

.. math::

   x_s = (1 - s) x_0 + s x_1

where ``x_0`` is the source, ``x_1`` is the training target and ``s`` is
drawn uniformly between 0 and 1. The model learns the velocity
``x_1 - x_0``.

Sampling integrates that velocity from ``s = 0`` to ``s = 1`` with an ODE
sampler such as ``euler`` or ``heun``. The sampler itself adds no noise:
ensemble members differ because each one starts from its own Gaussian
source.

Validation metrics estimate the target in one step from the predicted
velocity, ``x_s + (1 - s) v``, at the time drawn for that batch.

Training configuration
======================

The shared training base is ``training/transport.yaml``. Objective
configs specialize it:

.. code:: yaml

   # training/edm_diffusion.yaml
   defaults:
     - transport
     - _self_

   transport:
     objective: edm_diffusion

   training_loss:
     datasets:
       data:
         _target_: anemoi.training.losses.WeightedMSELoss

.. code:: yaml

   # training/flow_matching.yaml
   defaults:
     - transport
     - _self_

   transport:
     objective: flow_matching

EDM diffusion uses a weighted clean-endpoint objective. Flow matching
trains the velocity between the source field and the target with the
``MSELoss`` that ``training/transport.yaml`` selects by default.

Sampling defaults
=================

Default sampler settings live under
``model.model.transport.inference_defaults``:

.. code:: yaml

   inference_defaults:
     sampling_schedule:
       schedule_type: karras
       sigma_max: 100.0
       sigma_min: 0.02
       rho: 7.0
       num_steps: 50
     sampler:
       sampler: heun

These defaults can be overridden at inference time with
``schedule_params`` and ``sampler_params`` passed to ``predict_step``.
For flow matching, use the same structure with
``sampling_schedule.schedule_type: unit_time`` and a vector-field sampler
such as ``heun`` or ``euler``.

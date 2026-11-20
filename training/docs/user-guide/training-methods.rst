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
   probabilistic forecasters. The model config sets the objective
   (``model.model.transport.objective``: ``edm_diffusion`` or
   ``flow_matching``), and the model class sets state or tendency targets.

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
with :class:`AnemoiTransportModelEncProcDec`, which learns future states, or
:class:`AnemoiTransportTendModelEncProcDec`, which learns tendencies from the
latest input state. Each model class declares this as its
``prediction_mode``, and training prepares its targets to match.

.. warning::

   The plain GNN model is not supported for transport objective training.

Top-level configs
=================

The transport entry points are:

-  ``transport.yaml``: one dataset, EDM diffusion;
-  ``multi_transport.yaml``: two datasets (``era5`` and ``cerra``) trained
   together, EDM diffusion;
-  ``transport_flow_matching.yaml``: one dataset, flow matching.

They select the state model, ``graphtransformer_transport`` (or
``graphtransformer_multi_transport``). The tendency model is chosen with an
override:

.. code:: bash

   anemoi-training train --config-name=transport model=graphtransformer_transport_tendency
   anemoi-training train --config-name=multi_transport model=graphtransformer_multi_transport_tendency

Each entry config sets everything that depends on the objective, in the
model config and in the training config:

.. code:: yaml

   # transport.yaml (after its defaults list)
   model:
     model:
       transport:
         objective: edm_diffusion
         inference_defaults:
           sampling_schedule:
             schedule_type: karras
             sigma_max: 100.0
             sigma_min: 0.02
             rho: 7.0
             num_steps: 50
           sampler:
             sampler: heun

   training:
     transport:
       training_condition:
         distribution: karras
         sigma_max: 100.0
         sigma_min: 0.02
         rho: 7.0

The flow-matching entry config sets ``objective: flow_matching``, a
``unit_time`` sampling schedule and ``training_condition.distribution:
uniform_time``. Flow-matching time runs from 0 to 1, so it also sets the
noise embedder's ``scale: 1000``: the embedding frequencies then run from
1000 down to about 1.5 radians per unit of time. Config validation rejects
a ``training_condition`` that does not suit the model's objective.

Model configuration
===================

The model configs are the same for both objectives and put the transport
settings under ``model.model.transport``:

.. code:: yaml

   model:
     _target_: anemoi.models.models.AnemoiTransportModelEncProcDec
     transport:
       sigma_data: 1.0
       noise_channels: 32
       noise_cond_dim: 16
       noise_embedder:
         _target_: anemoi.models.layers.diffusion.SinusoidalEmbeddings
         num_channels: ${model.model.transport.noise_channels}
         max_period: 1000

``graphtransformer_transport_tendency`` loads ``graphtransformer_transport``
and switches to :class:`AnemoiTransportTendModelEncProcDec`, with a skip
connection from the latest input state. Both objectives start from Gaussian
noise.

``model.state_variables`` lists outputs of a predicted dataset that the
tendency model learns as states instead of tendencies. For a dataset that
is its own reference, only prognostic variables can be listed (its
diagnostics are learned as states anyway):

.. code:: yaml

   model:
     state_variables:
       data: [swvl1]

.. _tendencies-against-another-dataset:

Tendencies against another dataset
----------------------------------

By default the tendency model learns each predicted dataset's change
against its own latest input. ``model.tendency_reference`` takes the
reference from another dataset instead, for example a low-resolution
dataset projected onto the high-resolution grid by its
``spatial_processor``:

.. code:: yaml

   model:
     tendency_reference:
       hres: lres
     state_variables:
       hres: [tp]
     residual:
       datasets:
         lres:
           _target_: anemoi.models.layers.residual.SkipConnection
           step: matching

   data:
     datasets:
       hres:
         tendency_statistics_file: /path/to/tendency_statistics.npz

-  The outputs of ``hres`` that ``lres`` has as inputs under the same name
   are learned as tendencies: the ``hres`` value minus the ``lres`` input.
   The outputs listed in ``model.state_variables`` are learned as states
   instead. Every output of ``hres`` must be one or the other, otherwise
   the model raises.
-  ``hres`` cannot have prognostic variables: its inputs are valid at the
   output times, so a prognostic variable would give the model the target.
   Declare its inputs as ``forcing`` and its outputs as ``diagnostic``
   (a variable in neither list is prognostic). Declare every variable of
   ``lres`` as ``forcing``, so that it is read only.
-  With ``step: matching``, output step i is taken against ``lres`` input
   step i, so several snapshots (``SpatialDownscaler`` with several
   ``offsets``) are each taken against the ``lres`` snapshot at the same
   time. This needs as many ``lres`` input steps as ``hres`` output steps.
   With one output step, ``step: -1`` (the latest ``lres`` input) works
   too; with several, ``step: matching`` is required.
-  The reference is built by the residual connection of the reference
   dataset (``model.residual.datasets.lres``) and converted to physical
   units with that dataset's processors, so the two datasets can be
   normalised differently. A residual connection for ``hres`` would not be
   used and is rejected.
-  The tendencies are normalised with the statistics in
   ``tendency_statistics_file``: an ``.npz`` file with ``mean``, ``minimum``,
   ``maximum`` and ``stdev`` entries, each a ``{variable: value}`` mapping.
   They are used for every output step. A dataset with a reference from
   another dataset needs this file. Only the variables learned as
   tendencies need entries.
-  Remappings such as ``sqrt`` and output constraints act on states only,
   so they do not act on tendencies, and a tendency is not kept positive.
   To bound a field such as precipitation at 0, learn it as a state and
   add a ``relu`` post-processor.
-  ``condition_on_residual`` cannot be combined with a reference from
   another dataset.

``spatial_downscaler.yaml`` sets up a spatial downscaler this way: ``lres``
is read only (all forcings) and projected onto the ``hres`` grid by its
``spatial_processor``, ``hres`` has the forcings as inputs and the fields
to downscale as outputs only, and one ``joint`` encoder joins ``lres`` and
``hres`` node by node. ``2t``, ``10u`` and ``10v`` are learned as
tendencies; ``tp`` is learned as a state, normalised with ``std`` and
bounded at 0 by a ``relu`` post-processor, as in the forecaster. Its plots
show each snapshot of ``hres`` with the reference as input, so the
increment panel shows the predicted tendency and the persistence error
panel the true tendency.

Flow matching
=============

Flow matching follows a straight path from Gaussian noise to the training
target. The noisy target the model sees at time ``t`` is

.. math::

   x_t = (1 - t) x_0 + t x_1

where ``x_0`` is the noise, ``x_1`` is the training target and ``t`` is
drawn uniformly between 0 and 1. The model learns the velocity
``x_1 - x_0``.

Sampling integrates that velocity from ``t = 0`` to ``t = 1`` with an ODE
sampler such as ``euler`` or ``heun``. The sampler itself adds no noise:
ensemble members differ because each one starts from its own Gaussian
noise.

Validation metrics estimate the target in one step from the predicted
velocity, ``x_t + (1 - t) v``, at the time drawn for that batch.

Training configuration
======================

The settings shared by all transport training (precision, gradient
clipping, optimiser, training method) are in
``training/transport_settings.yaml``. ``training/transport.yaml`` puts them
on top of ``single.yaml`` for one dataset and ``training/multi_transport.yaml``
on top of ``multi.yaml`` for several. The entry config adds
``training.transport.training_condition``, the distribution of the noise
level or time drawn during training (see above).

Both objectives train with the ``MSELoss`` of ``single.yaml`` and
``multi.yaml``. EDM diffusion weights each sample's loss by its noise level,
so its training loss must apply these weights; ``MSELoss`` and the other
pointwise losses do, and a loss that would leave them unused is rejected when
training starts. Flow matching trains the velocity between the noise and the
target.

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
An override of the same ``schedule_type`` or ``sampler`` (or one that names
none) changes only the settings it gives. An override that names another
type replaces the defaults and must give that type's own settings, for
example ``{"schedule_type": "linear", "sigma_max": 100.0, "sigma_min": 0.02,
"num_steps": 20}``; a setting the new type does not take is an error.
For flow matching, use the same structure with
``sampling_schedule.schedule_type: unit_time`` and a vector-field sampler
such as ``heun`` or ``euler``.

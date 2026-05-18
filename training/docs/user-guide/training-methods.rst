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
   Configurable transport training for EDM diffusion and stochastic-interpolant
   probabilistic forecasters. Use ``training.prediction_mode`` to select
   state or tendency targets, and ``training.transport_objective`` to
   select ``edm_diffusion`` or ``stochastic_interpolant``.

.. note::

   ``EnsembleTraining`` and transport objectives such as EDM diffusion or
   stochastic interpolants require the GNN model type to be replaced with a
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

The `truncation` and `truncation_inv` can be used in the deterministic
or CRPS training. As described in :ref:`Field Truncation`, it transforms
the input to the model.

.. literalinclude:: yaml/example_crps_config.yaml
   :language: yaml
   :start-after: truncation_inv:
   :end-before: # Changes in datamodule

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
the processor grid using a sparse projection matrix. This is configured
via the ``noise_matrix`` parameter, which should point to a ``.npz``
file created with ``anemoi-graphs export_to_sparse`` (see
:ref:`usage-create_sparse_matrices`). Additional options
``row_normalize_noise_matrix`` and ``autocast`` control how the
projection matrix is applied.

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
   :end-before: diagnostics:

Typically, the validation metrics are the same as the training loss, but
different validation metrics can be added here (see :ref:`Losses`).

CRPS example config
===================

A typical config file for CRPS training is:

.. literalinclude:: yaml/example_crps_config.yaml
   :language: yaml


.. _diffusion-training:

****************************
 Transport objective training
****************************

Transport training covers probabilistic objectives that corrupt an
endpoint and train a model to recover either the clean endpoint or the
transport vector field. The supported objectives are ``edm_diffusion`` and
``stochastic_interpolant``.

Use :class:`~anemoi.training.train.methods.transport.TransportTraining`
with ``prediction_mode: state`` for state-space targets or
``prediction_mode: tendency`` for tendency-space targets. The model must
use :class:`AnemoiTransportModelEncProcDec` or
:class:`AnemoiTransportTendModelEncProcDec`; the plain GNN model is not
supported.

Top-level configs
=================

The transport entry points are:

-  ``transport_edm_diffusion.yaml``
-  ``transport_edm_diffusion_tendency.yaml``
-  ``transport_stochastic_interpolant.yaml``
-  ``transport_stochastic_interpolant_tendency.yaml``

These configs select ``training: edm_diffusion`` or
``training: stochastic_interpolant`` and set the corresponding
``training.transport_objective``. Tendency variants additionally select
``prediction_mode: tendency`` and a tendency model config.

Model configuration
===================

Transport model configs put all objective-specific settings under
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
       noise_embedder:
         _target_: anemoi.models.layers.diffusion.SinusoidalEmbeddings
         num_channels: ${model.model.transport.noise_channels}
         max_period: 1000
       source:
         kind: default
         scale: 1.0
         noise_scale: 0.0

For ``objective: edm_diffusion``, ``source.kind: default`` resolves to
``gaussian``. This is the recommended source for standard EDM sampling
and training.

For stochastic interpolants, set ``objective: stochastic_interpolant``
and configure the interpolant schedules:

.. code:: yaml

   model:
     model:
       transport:
         objective: stochastic_interpolant
         si_alpha_schedule: linear
         si_beta_schedule: linear
         si_sigma_schedule: brownian_bridge
         si_noise_scale: 1.0
         source:
           kind: gaussian

The available source kinds are ``zero``, ``gaussian``, and
``reference_state``. ``default`` resolves to the objective's default
source at training or sampling time. ``scale`` multiplies the source, and
``noise_scale`` adds additional additive Gaussian noise after the source
is built.

Stochastic-interpolant parameters
=================================

The stochastic-interpolant bridge combines a source endpoint, target
endpoint, and optional bridge noise:

.. math::

   x_s = \alpha(s) x_0 + \beta(s) x_1 + \sigma(s) \epsilon

where ``x_0`` is the selected ``source``, ``x_1`` is the training
target, and ``epsilon`` is standard Gaussian bridge noise.

-  ``si_alpha_schedule`` controls the source coefficient. Currently,
   ``linear`` gives ``alpha(s) = 1 - s``.
-  ``si_beta_schedule`` controls the target coefficient. ``linear``
   gives ``beta(s) = s`` and ``quadratic`` gives ``beta(s) = s^2``.
-  ``si_sigma_schedule`` controls the bridge-noise coefficient.
   ``brownian_bridge`` gives
   ``sigma(s) = si_noise_scale * sqrt(2 * s * (1 - s))``.
   The option ``quadratic_bridge`` gives
   ``sigma(s) = si_noise_scale * s * (1 - s)``, which is zero at
   both endpoints and has a finite derivative there.
-  ``si_noise_scale`` scales the stochastic bridge noise. Set it to
   ``0.0`` for a deterministic bridge.
-  ``source.noise_scale`` is separate from bridge noise. It adds
   additional additive Gaussian noise to the source endpoint before the
   bridge is built.

Flow-matching-like setup
===============================================

A flow-matching-like training can be set up as a
stochastic interpolant with a Gaussian source, linear endpoint schedules,
and no bridge noise:

.. code:: yaml

   training:
     transport_objective: stochastic_interpolant

   model:
     model:
       transport:
         objective: stochastic_interpolant
         si_alpha_schedule: linear
         si_beta_schedule: linear
         si_sigma_schedule: brownian_bridge
         si_noise_scale: 0.0
         source:
           kind: gaussian
           scale: 1.0
           noise_scale: 0.0

Stochastic-interpolant training learns the bridge velocity field. Use
ODE samplers such as ``euler`` or ``heun`` for sampling. Score-corrected SDE
sampling is currently not part of this objective.

Training configuration
======================

The shared training base is ``training/transport.yaml``. Objective
configs specialize it:

.. code:: yaml

   # training/edm_diffusion.yaml
   defaults:
     - transport
     - _self_

   transport_objective: edm_diffusion

   training_loss:
     datasets:
       data:
         _target_: anemoi.training.losses.WeightedMSELoss

.. code:: yaml

   # training/stochastic_interpolant.yaml
   defaults:
     - transport
     - _self_

   transport_objective: stochastic_interpolant

   training_loss:
     datasets:
       data:
         _target_: anemoi.training.losses.MSELoss

EDM diffusion uses a weighted clean-endpoint objective. Stochastic
interpolants train the drift/vector field between the selected source
and target endpoints. With a Gaussian source and ``si_noise_scale: 0``,
the stochastic-interpolant objective is the deterministic bridge case
commonly sampled with ODE solvers such as Euler or Heun.

Sampling defaults
=================

Default sampler settings live under
``model.model.transport.inference_defaults``:

.. code:: yaml

   inference_defaults:
     noise_scheduler:
       schedule_type: karras
       sigma_max: 100.0
       sigma_min: 0.02
       rho: 7.0
       num_steps: 50
     edm_diffusion_sampler:
       sampler: heun
     stochastic_interpolant_sampler:
       sampler: heun
       num_steps: 50

These defaults can be overridden at inference time with
``noise_scheduler_params`` and ``sampler_params`` passed to
``predict_step``.

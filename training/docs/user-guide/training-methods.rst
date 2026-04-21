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

``DiffusionTraining`` (``anemoi.training.train.methods.diffusion``)
   Base class for diffusion-based probabilistic forecasters. Applies
   stepwise pre/post-processors and handles the noise-conditioned
   forward pass.

.. note::

   ``EnsembleTraining`` and ``DiffusionTraining`` require the GNN
   model type to be replaced with a compatible architecture (e.g.
   GraphTransformer). The plain GNN processor is not supported for
   these methods.


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
      -  :class:`AlmostFairKernelCRPS`

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
the :class:`anemoi.training.losses.kcrps.AlmostFairKernelCRPS` loss
function (`Lang et al. (2024b) <https://arxiv.org/abs/2412.15832>`_):

.. math::

   \text{afCRPS}_\alpha := \alpha\text{fCRPS} + (1-\alpha)\text{CRPS}

The `alpha` parameter is a trade-off parameter between the CRPS and the
fair CRPS.

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

**************************
 Diffusion-based training
**************************

This section is intended for users who want to train a diffusion-based
model and are already familiar with the basic training configurations.

The diffusion training requires the following changes to the
deterministic training:

**Differences from deterministic training:**

-  **Forecaster class**: Use :class:`GraphDiffusionForecaster` (or
   :class:`GraphDiffusionTendForecaster` for tendency prediction)
   instead of :class:`GraphForecaster`

-  **Model config**: Use `graphtransformer_diffusion.yaml` or
   `transformer_diffusion.yaml` (or their `_diffusiontend` variants)
   instead of the standard configs

-  **Training config**: Use `diffusion.yaml` instead of `default.yaml`

-  **Model class**: Uses :class:`AnemoiDiffusionModelEncProcDec` (or
   :class:`AnemoiDiffusionTendModelEncProcDec`) instead of
   :class:`AnemoiModelEncProcDec`

-  **Loss computation**: WeightedMSELoss is recommended for diffusion
   training as it properly handles weighting according to the noise
   level.

Changes in model config
=======================

The config group for the model is set to
`graphtransformer_diffusion.yaml` or `transformer_diffusion.yaml`, which
specifies the :class:`AnemoiDiffusionModelEncProcDec` class with
diffusion-specific components.

Changes in the diffusion model configs:

.. code:: yaml

   model:
     _target_: anemoi.models.models.AnemoiDiffusionModelEncProcDec
     # Diffusion parameters
     diffusion:
       sigma_data: 1.0
       noise_channels: 32
       noise_cond_dim: 16
       sigma_max: 100.0
       sigma_min: 0.02
       rho: 7.0
       noise_embedder:
         _target_: anemoi.models.layers.diffusion.SinusoidalEmbeddings
         num_channels: ${model.model.diffusion.noise_channels}
         max_period: 1000
       inference_defaults:
         noise_scheduler:
           schedule_type: "karras"
           sigma_max: 100.0
           sigma_min: 0.02
           rho: 7.0
           num_steps: 50
         diffusion_sampler:
           sampler: "heun"
           S_churn: 0.0
           S_min: 0.0
           S_max: .inf
           S_noise: 1.0

The diffusion configuration includes:

-  `sigma_data`: Data standard deviation for preconditioning
-  `noise_channels`: Number of noise channels to inject
-  `noise_cond_dim`: Dimension of noise conditioning
-  `sigma_max` / `sigma_min`: Maximum and minimum noise levels
-  `rho`: Controls the noise schedule distribution
-  `noise_embedder`: Sinusoidal embeddings for noise conditioning
-  `inference_defaults`: Default parameters for noise scheduler and
   sampler, these are not used during training.

.. code:: yaml

   layer_kernels:
     LayerNorm:
       _target_: anemoi.models.layers.normalization.ConditionalLayerNorm
       normalized_shape: ${model.num_channels}
       condition_shape: 16
       zero_init: True
       autocast: false

The diffusion model uses conditional layer normalization to condition
the latent space on the noise level, enabling the model to denoise
appropriately at different noise scales.

Inference configuration
=======================

The `inference_defaults` block specifies default parameters for
sampling:

.. code:: yaml

   inference_defaults:
     noise_scheduler:
       schedule_type: "karras"  # Noise schedule type
       num_steps: 50           # Number of sampling steps
       sigma_max: 100.0        # Maximum noise level
       sigma_min: 0.02         # Minimum noise level
       rho: 7.0               # Schedule distribution parameter
     diffusion_sampler:
       sampler: "heun"         # Sampling algorithm
       S_churn: 0.0           # Stochasticity parameters
       S_min: 0.0
       S_max: .inf
       S_noise: 1.0

These defaults can be overridden at inference time by passing
`noise_scheduler_params` and `sampler_params` to the `predict_step`
method.

Here is an example of how to modify inference settings for a diffusion
model in your configuration:

.. code:: yaml

   checkpoint: /path/to/your/checkpoint
   date: 20250101T00:00:00
   predict_kwargs:
     noise_scheduler_params:
       num_steps: 20
       sigma_max: 90.0
       sigma_min: 0.03
       rho: 7.0
     sampler_params:
       sampler: "heun"
       S_churn: 2.5
       S_min: 0.75
       S_max: 90
       S_noise: 1.05

Changes in training config
==========================

The training configuration for diffusion models requires changes:

.. code:: yaml

   # Select diffusion model task
   # For standard diffusion:
   training_method: anemoi.training.train.methods.DiffusionTraining

   # For tendency-based diffusion:
   training_method: anemoi.training.train.methods.DiffusionTendencyTraining

   # Standard training configuration remains similar
   multistep_input: 2
   rollout:
     start: 1
     max: 1

The training method must be set to the appropriate diffusion training class
to handle the diffusion-specific forward pass with preconditioning and
noise injection.

Changes in loss computation
===========================

The diffusion training uses WeightedMSELoss which handles noise weights
properly:

.. code:: yaml

   training_loss:
      datasets:
          your_dataset_name:
              _target_: anemoi.training.losses.WeightedMSELoss

During training, the :class:`GraphDiffusionForecaster` automatically
passes the required `weights` based on the noise level to the loss
function.

Diffusion model variants
=========================

There are two variants of diffusion models available:

**Standard Diffusion**
----------------------

Uses `graphtransformer_diffusion.yaml` or `transformer_diffusion.yaml`:

-  Predicts the denoised state directly
-  Applies noise to the target state during training
-  Model class: :class:`AnemoiDiffusionModelEncProcDec`
-  Training method: :class:`DiffusionTraining`
-  Use single-step rollout (`rollout.max: 1`)

**Tendency-based Diffusion**
----------------------------

Uses `graphtransformer_diffusiontend.yaml` or
`transformer_diffusiontend.yaml`:

-  Predicts the tendency (change) between timesteps
-  Applies noise to the tendency rather than the state
-  Model class: :class:`AnemoiDiffusionTendModelEncProcDec`
-  Training method: :class:`DiffusionTendencyTraining`
-  Requires `statistics_tendencies` for normalization
-  Use single-step rollout (`rollout.max: 1`)

Choose the variant based on your specific use case.

Diffusion example config
=========================

A minimal config file for standard diffusion training:

.. code:: yaml

   defaults:
   - data: zarr
   - dataloader: native_grid
   - diagnostics: evaluation
   - system: example
   - graph: multi_scale
   - model: graphtransformer_diffusion  # Use diffusion model
   - task: forecaster
   - training: diffusion                 # Use diffusion training config
   - _self_

   # Select training method for diffusion
   training:
     training_method: anemoi.training.train.methods.DiffusionTraining

   config_validation: True

For tendency-based diffusion, change the model config and model task:

.. code:: yaml

   defaults:
   - data: zarr
   - dataloader: native_grid
   - diagnostics: evaluation
   - system: example
   - graph: multi_scale
   - model: graphtransformer_diffusiontend  # Use tendency diffusion model
   - task: forecaster
   - training: diffusion                     # Same training config
   - _self_

   # Select training method for tendency-based diffusion
   training:
     training_method: anemoi.training.train.methods.DiffusionTendencyTraining

   # Ensure statistics_tendencies are available
   config_validation: True

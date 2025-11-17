##########################
 Diffusion-based training
##########################

This guide is intended for users who want to train a diffusion-based
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

*************************
 Changes in model config
*************************

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
       w_one_bias_zero_init: True
       autocast: false

The diffusion model uses conditional layer normalization to condition
the latent space on the noise level, enabling the model to denoise
appropriately at different noise scales.

*************************
 Inference configuration
*************************

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

****************************
 Changes in training config
****************************

The training configuration for diffusion models requires changes:

.. code:: yaml

   # Select diffusion model task
   # For standard diffusion:
   model_task: anemoi.training.train.tasks.GraphDiffusionForecaster

   # For tendency-based diffusion:
   model_task: anemoi.training.train.tasks.GraphDiffusionTendForecaster

   # Standard training configuration remains similar
   multistep_input: 2
   rollout:
     start: 1
     max: 1

The model task must be set to the appropriate diffusion forecaster class
to handle the diffusion-specific forward pass with preconditioning and
noise injection.

*****************************
 Changes in loss computation
*****************************

The diffusion training uses WeightedMSELoss which handles noise weights
properly:

.. code:: yaml

   training_loss:
     _target_: anemoi.training.losses.WeightedMSELoss

During training, the :class:`GraphDiffusionForecaster` automatically
passes the required `weights` based on the noise level to the loss
function.

**************************
 Diffusion model variants
**************************

There are two variants of diffusion models available:

**Standard Diffusion**
======================

Uses `graphtransformer_diffusion.yaml` or `transformer_diffusion.yaml`:

-  Predicts the denoised state directly
-  Applies noise to the target state during training
-  Model class: :class:`AnemoiDiffusionModelEncProcDec`
-  Forecaster: :class:`GraphDiffusionForecaster`
-  Use single-step rollout (`rollout.max: 1`)

**Tendency-based Diffusion**
============================

Uses `graphtransformer_diffusiontend.yaml` or
`transformer_diffusiontend.yaml`:

-  Predicts the tendency (change) between timesteps
-  Applies noise to the tendency rather than the state
-  Model class: :class:`AnemoiDiffusionTendModelEncProcDec`
-  Forecaster: :class:`GraphDiffusionTendForecaster`
-  Requires `statistics_tendencies` for normalization
-  Use single-step rollout (`rollout.max: 1`)

Choose the variant based on your specific use case.

****************
 Example config
****************

A minimal config file for standard diffusion training:

.. code:: yaml

   defaults:
   - data: zarr
   - dataloader: native_grid
   - diagnostics: evaluation
   - hardware: example
   - graph: multi_scale
   - model: graphtransformer_diffusion  # Use diffusion model
   - training: diffusion                 # Use diffusion training config
   - _self_

   # Select model task for diffusion
   training:
     model_task: anemoi.training.train.tasks.GraphDiffusionForecaster

   config_validation: True

For tendency-based diffusion, change the model config and model task:

.. code:: yaml

   defaults:
   - data: zarr
   - dataloader: native_grid
   - diagnostics: evaluation
   - hardware: example
   - graph: multi_scale
   - model: graphtransformer_diffusiontend  # Use tendency diffusion model
   - training: diffusion                     # Same training config
   - _self_

   # Select model task for tendency-based diffusion
   training:
     model_task: anemoi.training.train.tasks.GraphDiffusionTendForecaster

   # Ensure statistics_tendencies are available
   config_validation: True

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

-  **Loss computation**: Standard MSE loss (noise-based weighting
   applied automatically during training)

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
     # Diffusion noise parameters
     diffusion:
       noise:
         rho: 7
         sigma_max: 100.0
         sigma_min: 0.02
         sigma_data: 1.0
         noise_channels: 32
         noise_cond_dim: 16

The diffusion configuration includes noise scheduling parameters:

-  `rho`: Controls the noise schedule distribution
-  `sigma_max` / `sigma_min`: Maximum and minimum noise levels
-  `sigma_data`: Data standard deviation for preconditioning
-  `noise_channels`: Number of noise channels to inject
-  `noise_cond_dim`: Dimension of noise conditioning

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

****************************
 Changes in training config
****************************

The training configuration for diffusion models requires changes:

.. code:: yaml

   # Select diffusion model task
   # For standard diffusion:
   model_task: anemoi.training.train.forecaster.GraphDiffusionForecaster

   # For tendency-based diffusion:
   model_task: anemoi.training.train.forecaster.GraphDiffusionTendForecaster

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

The diffusion training uses the same MSE loss class but requires noise
weights during computation:

.. code:: yaml

   training_loss:
     _target_: anemoi.training.losses.MSELoss

During training, the :class:`GraphDiffusionForecaster` automatically
applies the required `weights` based on the noise level before the loss
computation.

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
-  Suitable for most weather forecasting applications

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
   - datamodule: single
   - hardware: example
   - graph: multi_scale
   - model: graphtransformer_diffusion  # Use diffusion model
   - training: diffusion                 # Use diffusion training config
   - _self_

   # Select model task for diffusion
   training:
     model_task: anemoi.training.train.forecaster.GraphDiffusionForecaster

   config_validation: True

For tendency-based diffusion, change the model config and model task:

.. code:: yaml

   defaults:
   - data: zarr
   - dataloader: native_grid
   - diagnostics: evaluation
   - datamodule: single
   - hardware: example
   - graph: multi_scale
   - model: graphtransformer_diffusiontend  # Use tendency diffusion model
   - training: diffusion                     # Same training config
   - _self_

   # Select model task for tendency-based diffusion
   training:
     model_task: anemoi.training.train.forecaster.GraphDiffusionTendForecaster

   # Ensure statistics_tendencies are available
   config_validation: True

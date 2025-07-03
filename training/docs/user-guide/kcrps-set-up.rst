##############################
 Ensemble CRPS-based training
##############################

This guide is intended for users who want to train an ensemble
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

   -  -  Forecaster
      -  :class:`GraphForecaster`
      -  :class:`GraphEnsForecaster`

   -  -  Strategy
      -  :class:`DDPGroupStrategy`
      -  :class:`DDPEnsGroupStrategy`

   -  -  Training loss
      -  :class:`WeightedMSELoss`
      -  :class:`AlmostFairKernelCRPS`

   -  -  Model
      -  :class:`AnemoiModelEncProcDec`
      -  :class:`AnemoiEnsModelEncProcDec`

   -  -  Datamodule
      -  :class:`AnemoiDatasetsDataModule`
      -  :class:`AnemoiEnsDatasetsDataModule`

****************************
 Changes in hardware config
****************************

.. literalinclude:: yaml/example_crps_config.yaml
   :language: yaml
   :start-after: # Changes in hardware
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
is embedded and added to the latent space of the processor using a
conditional layer norm.

.. code:: yaml

   layer_kernels:
      processor:
         LayerNorm:
            _target_: anemoi.models.layers.normalization.ConditionalLayerNorm
            normalized_shape: ${model.num_channels}
            condition_shape: ${model.noise_injector.noise_channels_dim}
            w_one_bias_zero_init: True
            autocast: false
         ...

In order to condition the latent space on the noise, we need to use a
different layer norm in the processor, here the
:class:`anemoi.models.layers.normalization.ConditionalLayerNorm`.

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
   `ensemble_size_per_device` and the `num_gpus_per_ensemble`.

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

****************
 Example config
****************

A typical config file for CRPS training is:

.. literalinclude:: yaml/example_crps_config.yaml
   :language: yaml

############
 Optimizers
############

Optimizers are responsible for updating the model parameters during
training based on the computed gradients from the loss function. In
``anemoi-training``, optimizers are configured in the training
configuration file under ``config.training.optimizer``. By default,
optimizers are instantiated using a Hydra-style ``_target_`` entry,
allowing full flexibility to specify both standard PyTorch optimizers
and custom implementations.

The optimizer configuration is handled internally by the
``BaseGraphModule`` class through its ``_create_optimizer_from_config``
method, which reads the provided configuration and creates the
corresponding optimizer object. Additional settings, such as learning
rate schedulers and warm-up phases, are also defined and managed within
the same module.

**************************
 Configuring an Optimizer
**************************

An optimizer can be defined in the training configuration file using its
Python import path as the ``_target_``. For example, to use the standard
Adam optimizer:

.. code:: yaml

   optimizer:
      _target_: torch.optim.Adam
      betas: [0.9, 0.95]
      weight_decay: 0.1

The ``BaseGraphModule`` automatically injects the learning rate from
``config.training.lr``. The optimizer configuration can therefore focus
on algorithm-specific parameters.

**************************
 Learning Rate Schedulers
**************************

Learning rate schedulers can be attached to any optimizer to control the
evolution of the learning rate during training. By default,
``anemoi-training`` uses a ``CosineLRScheduler`` from ``timm.scheduler``
with optional warm-up steps and minimum learning rate.

The scheduler is created by ``BaseGraphModule._create_scheduler`` and
returned to the trainer together with the optimizer in
``configure_optimizers``. Currently, the scheduler is hard-coded to
``CosineLRScheduler``, but in the future this will be made more flexible
to allow configurable schedulers.

The scheduler is returned in a dictionary of the form:

.. code:: python

   {
       "optimizer": optimizer,
       "lr_scheduler": {
           "scheduler": scheduler,
           "interval": "step",
       },
   }

********************
 AdEMAMix Optimizer
********************

``AdEMAMix`` is a custom optimizer implemented in
``anemoi.training.optimizers.AdEMAMix.py`` and taken from the `Apple ML
AdEMAMix project <https://github.com/apple/ml-ademamix>`_. It combines
elements of Adam and exponential moving average (EMA) mixing for
improved stability and generalization.

The optimizer maintains **three exponential moving averages (EMAs)** of
the gradients. See <https://arxiv.org/abs/2409.03137> for more details.

***********************
 Configuration in YAML
***********************

An example configuration for using ``AdEMAMix`` is shown below:

.. code:: yaml

   optimizer:
      _target_: anemoi.training.optimizers.AdEMAMix.AdEMAMix
      betas: [0.9, 0.999, 0.9999]
      alpha: 2.0
      weight_decay: 0.01
      beta3_warmup: 1000
      alpha_warmup: 1000

**************************
 Implementation Reference
**************************

.. automodule:: anemoi.training.optimizers.AdEMAMix
   :members:
   :no-undoc-members:
   :show-inheritance:

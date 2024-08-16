###############
 Configuration
###############

Anemoi Training uses Hydra for configuration management, allowing for
flexible and modular configuration of the training pipeline. This guide
explains how to use Hydra effectively in the project.

**************
 Hydra Basics
**************

Hydra is a framework for elegantly configuring complex applications. It
allows for:

#. Hierarchical configuration
#. Configuration composition
#. Dynamic object instantiation

*********************************
 Object Instantiation with Hydra
*********************************

Hydra provides powerful tools for instantiating objects directly from
configuration files:

-  `hydra.utils.instantiate()`: Creates object instances
-  `hydra.utils.call()`: Calls functions with configured parameters

Example: Instantiating an Optimizer
===================================

Consider the following Python class:

.. code:: python

   class Optimizer:
       def __init__(self, algorithm: str, learning_rate: float) -> None:
           self.opt_algorithm = algorithm
           self.lr = learning_rate

Configuration in YAML:

.. code:: yaml

   optimizer:
     _target_: my_code.Optimizer
     algorithm: SGD
     learning_rate: 0.01

Instantiating in code:

.. code:: python

   from hydra.utils import instantiate

   optimizer = instantiate(config.optimizer)

********************************************
 Configurable Components in Anemoi Training
********************************************

Anemoi Training uses Hydra's instantiation feature for various
components, including:

#. Model architectures
#. Pressure level scalers
#. Graph definitions

And there are plans to extend these to other areas, such as:

#. Loss functions
#. Callbacks
#. Data loaders

Example: Configuring a Pressure Level Scaler
============================================

In `config.training.pressure_level_scaler`, users can define custom
scaling behavior:

.. code:: yaml

   pressure_level_scaler:
       _target_: anemoi.training.losses.scalers.ReLUPressureLevelScaler
       min_weight: 0.2

****************************************
 Best Practices for Hydra Configuration
****************************************

#. Use configuration groups for logically related settings.
#. Leverage Hydra's composition feature to combine configurations.
#. Use interpolation to reduce redundancy in configurations.
#. Provide default values for all configurable parameters.
#. Use type hints in your classes to ensure correct instantiation.

*************************
 Advanced Hydra Features
*************************

1. Config Groups
================

Organize related configurations into groups for easier management and
overriding.

2. Multi-run
============

Hydra supports running multiple configurations in a single execution:

.. code:: bash

   python train.py --multirun optimizer.learning_rate=0.001,0.01,0.1

3. Sweeps
=========

Define parameter sweeps for hyperparameter tuning, a powerful feature,
but usually only required when the model development is relatively
mature:

.. code:: yaml

   # config.yaml
   defaults:
     - override hydra/sweeper: optuna

   hydra:
     sweeper:
       sampler:
         _target_: optuna.samplers.TPESampler
       direction: minimize
       n_trials: 20
       params:
         optimizer.learning_rate: range(0.0001, 0.1, log=true)

Run the sweep:

.. code:: bash

   python train.py --multirun

By leveraging these Hydra features, you can create flexible,
maintainable, and powerful configurations for Anemoi Training.

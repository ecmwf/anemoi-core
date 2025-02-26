###############
 Configuration
###############

Anemoi Training uses Hydra and Pydantic for configuration management,
allowing for flexible and modular configuration of the training pipeline
while provide robustness through validation. This guide explains how to
use Hydra and Pydantic effectively in the project.

***************************************
 Pydantic and Configuration Validation
***************************************

Pydantic is a package designed for data validation and settings
management. It provides a simple way to define schemas which can be used
to validate configuration files. For example, the following schema can
be used to validate a training configuration:

.. code:: python

   from pydantic import BaseModel, Field, PositiveFloat, Literal

   class TrainingSchema(BaseModel):
      model: Literal{"AlexNet", "ResNet", "VGG"} = Field(default="AlexNet")
      """Model architecture to use for training."""
      learning_rate: PositiveFloat = Field(default=0.01)
      """Learning rate."""
      loss: str = Field(default="mse")
      """Loss function."""

To allow more complex configurations, Pydantic also supports nested
schemas. For example, the following schema can be used to validate a
configuration with a configurable model:

.. code:: python

   from pydantic import BaseModel, Field, PositiveFloat, Literal

   from enum import StrEnum

   class ActivationFunctions(StrEnum):
       relu = "relu"
       sigmoid = "sigmoid"
       tanh = "tanh"

   class ModelSchema(BaseModel):
       num_layers: PositiveInt = Field(default=3)
       """Number of layers in the model."""
       activation: ActivationFunctions = Field(default="relu")
       """Activation function to use."""

   class TrainingSchema(BaseModel):
       model: ModelSchema
       """Model configuration."""
       learning_rate: PositiveFloat = Field(default=0.01)
       """Learning rate."""
       loss: str = Field(default="mse")
       """Loss function."""

If your new feature requires a new configuration parameter, you should
add it to the appropriate schemas and update the configuration files
accordingly.

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

Pydantic schema:

.. code:: python

   from pydantic import BaseModel

   class OptimizerSchema(BaseModel):
       algorithm: str
       learning_rate: float

Instantiating in code:

.. code:: python

   from hydra.utils import instantiate

   optimizer = instantiate(config.optimizer.model_dump())

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

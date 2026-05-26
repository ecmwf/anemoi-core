################
 Model Training
################

Anemoi provides a modular and extensible training framework for Graph
Neural Networks (GNNs), designed for tasks such as forecasting,
temporal downscaling, and ensemble learning. The training setup is structured
around three key components:

-  ``BaseTrainingModule``: The abstract base class for all task-specific
   models, encapsulating shared logic for training, evaluation, and
   distributed execution.

-  **Tasks**: Task-specific subclasses that implement models for
   forecasting, temporal downscaling, autoencoding, etc.

-  ``AnemoiTrainer``: The training orchestrator responsible for running
   and managing the training and validation loops.

To train a model, users typically subclass one of the pre-implemented
graph modules or create a new one by extending ``BaseTrainingModule``.

*****************
 BaseTrainingModule
*****************

All training methods subclass :class:`~anemoi.training.train.methods.base.BaseTrainingModule`,
which itself inherits from PyTorch Lightning's
:class:`~pytorch_lightning.LightningModule`. This base class defines the
standard interface for all models in Anemoi and implements the core
logic required for training, validation, and distributed inference.

Key responsibilities include:

-  Support for sharded and distributed training
-  Node-based weighting and custom loss scaling
-  Normalization and inverse-scaling of output variables
-  Validation metric computation with customizable subsets
-  Input/output masking to support variable or region-specific
   processing

``BaseTrainingModule`` is not intended to be instantiated directly.
Instead, use one of the concrete training methods or subclass it to
implement a new one by overriding the :meth:`_step` method.

**Core Parameters:**

-  ``config``: A structured configuration (usually a dataclass) defining
   model architecture and training settings.
-  ``graph_data``: A :class:`~torch_geometric.data.HeteroData` object
   with static node and edge features.
-  ``statistics`` / ``statistics_tendencies``: Mean and std dev for
   normalization of variables.
-  ``data_indices``: Index mappings between variable names and tensor
   positions.
-  ``supporting_arrays``: Optional maps like topography or land-sea
   masks.

**Subclasses must implement:**

-  :meth:`_step`: Defines how a batch is processed and losses are
   computed.

Additional features include optional sharding of input batches across
devices (to reduce communication overhead), dynamic creation of scalers
from statistics.

.. automodule:: anemoi.training.train.methods.base
   :members:
   :no-undoc-members:
   :show-inheritance:

*********************
 Training Methods
*********************

The training method is the PyTorch Lightning module that implements the
forward pass, loss computation, and metric calculation for a given task.
All methods inherit from
:class:`~anemoi.training.train.methods.base.BaseTrainingModule`.

:class:`~anemoi.training.train.methods.single.SingleTraining`
   Deterministic single-member training. Compatible with
   :class:`~anemoi.training.tasks.forecasting.Forecaster`,
   :class:`~anemoi.training.tasks.temporal_downscaling.TemporalDownscaler`,
   and :class:`~anemoi.training.tasks.timeless.Autoencoder`.

:class:`~anemoi.training.train.methods.ensemble.EnsembleTraining`
   Ensemble (multi-member) training. Generates multiple perturbed
   members per device and uses ``DDPEnsGroupStrategy`` for distributed
   execution.

:class:`~anemoi.training.train.methods.diffusion.DiffusionTraining`
   Base class for diffusion-based probabilistic forecasters. Applies
   stepwise pre/post-processors and handles the noise-conditioned
   forward pass.

.. automodule:: anemoi.training.train.methods.single
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: anemoi.training.train.methods.ensemble
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: anemoi.training.train.methods.diffusion
   :members:
   :no-undoc-members:
   :show-inheritance:

*****************
 Available Tasks
*****************

Anemoi supports multiple task-specific implementations that define the
temporal I/O structure for each scientific workflow.

Current supported tasks include:

#. **Forecaster** —
   :class:`~anemoi.training.tasks.forecasting.Forecaster`
#. **Temporal Downscaler** —
   :class:`~anemoi.training.tasks.temporal_downscaling.TemporalDownscaler`
#. **AutoEncoder** —
   :class:`~anemoi.training.tasks.timeless.Autoencoder`

Each task defines which time steps are loaded as inputs and targets and
provides helpers for mapping those offsets to batch positions.

.. seealso::

   :doc:`tasks` — Task classes, ``RolloutConfig``, and batch-index helpers.

Key methods to override when adapting or extending a model:

-  ``__init__``: Customizes the model architecture and internal
   components.
-  ``_step``: Implements the forward pass and loss/metric computation
   for a single batch.

Task ``_step`` return contract
==============================

Task implementations are expected to return a 3-tuple with a consistent
shape across all task types:

- ``loss``: a tensor scalar used for optimization.
- ``metrics``: a mapping of metric names to tensors.
- ``predictions``: a list of per-step dictionaries keyed by dataset
  name.

For single-output tasks (for example autoencoder), the
``predictions`` value is a one-element list. For rollout-based tasks,
the list contains one entry per rollout step. This shared contract keeps
plotting callbacks task-agnostic and avoids task-specific unpacking
logic.

*********************
 Training Controller
*********************

The training process is orchestrated by
:class:`~anemoi.training.train.train.AnemoiTrainer`, which wraps a
PyTorch Lightning Trainer and provides additional logic for:

-  Distributed training and inference
-  Dynamic loss scheduling and learning rate adjustment
-  Logging and profiling via ``profiler.py``
-  Dataset loading
-  Graph loading and creation

.. automodule:: anemoi.training.train.train
   :members:
   :no-undoc-members:
   :show-inheritance:

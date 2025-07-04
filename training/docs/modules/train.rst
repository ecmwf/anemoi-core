################
 Model Training
################

Anemoi provides a modular and extensible training framework for Graph
Neural Networks (GNNs), designed for tasks such as forecasting,
interpolation, and ensemble learning. The training setup is structured
around three key components:

-  ``BaseGraphModule``: The abstract base class for all task-specific
   models, encapsulating shared logic for training, evaluation, and
   distributed execution.

-  **Tasks**: Task-specific subclasses that implement models for
   deterministic forecasting, interpolation, ensemble learning, etc.

-  ``AnemoiTrainer``: The training orchestrator responsible for running
   and managing the training and validation loops.

To train a model, users typically subclass one of the pre-implemented
graph modules or create a new one by extending ``BaseGraphModule``.

*****************
 BaseGraphModule
*****************

All model tasks subclass :class:`~anemoi.graphmodules.BaseGraphModule`,
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

``BaseGraphModule`` is not intended to be instantiated directly.
Instead, model developers should subclass it to implement specific
forecasting or interpolation tasks by overriding the :meth:`_step`
method and optionally customizing the initialization logic in
:meth:`__init__`.

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
-  ``truncation_data``: Masking information to restrict training or
   validation to a subset of time or space.

**Subclasses must implement:**

-  :meth:`_step`: Defines how a batch is processed and losses are
   computed.

Additional features include optional sharding of input batches across
devices (to reduce communication overhead), dynamic creation of scalers
from statistics.

.. autoclass:: anemoi.graphmodules.BaseGraphModule
   :members:
   :undoc-members:
   :show-inheritance:

*****************
 Available Tasks
*****************

Anemoi supports multiple task-specific models, which are high-level
subclasses of :class:`~anemoi.graphmodules.BaseGraphModule` and provide
working implementations for key scientific workflows.

Current supported graphmodules include:

#. **Deterministic Forecasting** —
   :class:`~anemoi.training.train.tasks.forecaster.GraphForecaster`
#. **Ensemble Forecasting** —
   :class:`~anemoi.training.train.tasks.ensforecaster.GraphEnsForecaster`
#. **Time Interpolation** —
   :class:`~anemoi.training.train.tasks.interpolator.GraphInterpolator`

Each of these implements the ``__init__`` and ``_step`` methods to
define task-specific model behavior. They support full Lightning
compatibility and make use of internal utilities like metric tracking,
variable masking, and inverse-scaling.

Key methods to override when adapting or extending a model:

-  ``__init__``: Customizes the model architecture and internal
   components.
-  ``_step``: Implements the forward pass and loss/metric computation
   for a single batch.

.. automodule:: anemoi.training.train.tasks.forecaster
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: anemoi.training.train.tasks.ensforecaster
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: anemoi.training.train.tasks.interpolator
   :members:
   :no-undoc-members:
   :show-inheritance:

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

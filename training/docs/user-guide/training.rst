##########
 Training
##########

The Anemoi Training module is the heart of the framework where machine
learning models for weather forecasting are trained. This section will
guide you through the entire training process, from setting up your data
to configuring your model and executing the training pipeline.

*************
 Setup Steps
*************

Anemoi Training requires two primary components to get started:

Step 1 and 2:
=============

#. **Graph Definition from Anemoi Graphs:** This defines the structure
   of your machine learning model, including the layers, connections,
   and operations that will be used during training.

#. **Dataset from Anemoi Datasets:** This provides the training data
   that will be fed into the model. The dataset should be pre-processed
   and formatted according to the specifications of the Anemoi Datasets
   module.

These 2 steps are outlined in :ref:`prep-training-components`.

Step 3: Configure the Training Process
======================================

Once your graph definition and dataset are ready, you can configure the
training process. Anemoi Training allows you to adjust various
parameters such as learning rate, batch size, number of epochs, and
other hyperparameters that control the training behavior.

To configure the training:

-  Specify the training parameters in your configuration file or through
   the command line interface.
-  Replace all "missing" values in config `???` with the appropriate
   values for your training setup.
-  Choose the model task and model type from :ref:`Models <Models>`.
-  Optionally, customize additional components like the normaliser or
   optimization strategies to enhance model performance.

*****************
 Parallelization
*****************

Anemoi Training supports different parallelization strategies based on
the training task (see :ref:`Strategy <strategy target>`):

-  **DDPGroupStrategy**: Used for deterministic training tasks
-  **DDPEnsGroupStrategy**: Used for ensemble training tasks

These strategies have to be set depending on the model task specified in
the configuration.

Step 4: Set Up Experiment Tracking (Optional)
=============================================

Experiment tracking is an essential aspect of machine learning
development, allowing you to keep track of various runs, compare model
performances, and reproduce results. Anemoi Training can be easily
integrated with popular experiment tracking tools like **TensorBoard**,
**MLflow** or **Weights & Biases (W&B)**.

These different tools provide various features such as visualizing
training metrics, logging hyperparameters, and storing model
checkpoints. You can choose the tool that best fits your workflow and
set it up to track your training experiments.

To set up experiment tracking:

#. Install the desired experiment tracking tool (e.g., TensorBoard,
   MLflow, or W&B).
#. Configure the tool in your training configuration file or through the
   command line interface.
#. Start the experiment tracking server and monitor your training runs
   in real-time.

Step 5: Execute Training
========================

With everything set up, you can now execute the training process. Anemoi
Training will use the graph definition and dataset to train your model
according to the specified configuration.

To execute training:

-  Run the training command given below, ensuring that all paths to the
   graph definition and dataset are correctly specified.
-  Monitor the training process, adjusting parameters as needed to
   optimize model performance.
-  Upon completion, the trained model will be registered and stored for
   further use.

Make sure you have a GPU available and simply call:

.. code:: bash

   anemoi-training train

.. _restart target:

**************
 Data Routing
**************

Anemoi Training uses the Anemoi Datasets module to load the data.

Anemoi training implements data routing, in which you can specify which
variables are used as ``forcings``; used as input only, and which
variables are as ``diagnostics``; appear as output only and to be
predicted by the model. All remaining variables will be treated as
``prognostic``, i.e. they appear as both inputs and outputs.

``Forcings`` are variables such as solar insolation or land-sea-mask.
These would make little sense to predict as they are external to the
model. These can be static (like the land-sea-mask) or dynamic (like
solar insolation). Note within anemoi, forcing does not have the
classical NWP meaning of external variables which impact the model, such
as wind forcing applied to an ocean model. Instead, forcing here refers
to any variable which is an input only. In some cases this includes
'traditional forcing', alongside other variables.

``Diagnostics`` includes the variables like precipitation that we want
to predict, but which may not be available in forecast step zero due to
technical limitations. These can aso include derived quantities which we
wish to train the model to predict directly, but do not want to use as
inputs.

``Prognostic`` variables are the variables like temperature or humidity
that we want to predict and appear as both inputs and outputs.

The user can specify the routing of the data by setting the
``config.data.forcings`` and ``config.data.diagnostics``. These are
named strings, as Anemoi datasets enables us to address variables by
name. Any variable in the dataset which is not listed as either forcing
or diagnostic (or dropped, see :ref:`Dataloader <Dataloader>` below),
will be classed as a prognostic variable.

.. code:: yaml

   data:
      forcings:
         - solar_insolation
         - land_sea_mask
      diagnostics:
         - total_precipitation

**************
 Data Modules
**************

Anemoi Training provides different data modules to handle various model
tasks:

-  **AnemoiDatasetDataModule**: Standard data module for deterministic
   training

-  **AnemoiEnsDatasetsDataModule**: Specialized data module for ensemble
   training. It also allows for training with perturbed initial
   conditions.

The choice of data module depends on your training task and input data
requirements.

************
 Dataloader
************

The dataloader file contains information on how many workers are used,
and the batch size. ``num_workers`` relates to model parallelisation.

.. code:: yaml

   num_workers:
      training: 8
      validation: 8
      test: 8
   batch_size:
      training: 2
      validation: 4
      test: 4

   limit_batches:
      training: null
      validation: null
      test: 20

The grid points being modelled are also defined. In many cases this will
be the full grid. For limited area modelling, you may want to define a
set of target indices which mask/remove some grid points, leaving only
the area being modelled.

.. code:: yaml

   # set a custom mask for grid points.
   # Useful for LAM (dropping unconnected nodes from forcing dataset)
   grid_indices:
      _target_: anemoi.training.data.grid_indices.FullGrid
      nodes_name: ${graph.data}

The dataloader file also describes the files used for training,
validation and testing, and the datasplit For machine learning, we
separate our data into: training data, used to train the model;
validation data, used to assess various version of the model throughout
the model development process; and test data, used to assess a final
version of the model. Best practice is to separate the data in time,
ensuring the validation and test data are suitably independent from the
training data.

We define the start and end time of each section of the data. This can
be given as a full date, or just the year, or year and month, in these
cases the first of the month/first of the year is used.

The dataset used, and the frequency can be set spearately for the
different parts of the dataset, for example, if test data is stored in a
different file.

By default, every variable within the dataset is used. If this is not
desired, variables can be listed within ``drop`` and they won't be used.
Conversely, if only a few variables from the file are needed ``select``
can be used in place of drop, and only the listed variables are used.
The same overall set of variables must be used throughout training,
validation and test. If using different files, which contain different
variables, the items listed in drop/select may vary.

.. literalinclude:: yaml/dataloader.yaml
   :language: yaml

***************
 Normalisation
***************

Machine learning models are sensitive to the scale of the input data. To
ensure that the model can learn effectively, it is important to
normalise the input data, so all variables exhibit a similar range. This
ensures variables have comparable contributions to the loss function,
and enables the model to learn effectively.

The nornmaliser is one of many 'preprocessors' within anemoi, it
implements multiple strategies that can be applied to the data using the
config. Currently, the normaliser supports the following strategies:

-  ``none``: No normalisation is applied.
-  ``mean-std``: Data is normalised by subtracting the mean and dividing
   by the standard deviation
-  ``std``: Data is normalised by dividing by the standard deviation.
-  ``min-max``: Data is normalised by substracting the min value and
   dividing by the range.
-  ``max``: Data is normalised by dividing by the max value.

Values like the land-sea-mask do not require additional normalisation as
they already span a range between 0 and 1. Variables like temperature or
humidity are usually normalised using ``mean-std``. Some variables like
the geopotential height should be max normalised, so the 'zero' point
and the proportional distance from this point is retained,

The user can specify the normalisation strategy by choosing a default
method, and additionally specifying specific cases for certain variables
within ``config.data.normaliser``:

.. code:: yaml

   normaliser:
      default: mean-std
      none:
         - land_sea_mask
      max:
         - geopotential_height

An additional option in the normaliser overwrites statistics of specific
variables onto others. This is primarily used for convective
precipitation (cp), which is a fraction of total precipitation (tp), by
overwriting the cp statistics with the tp statistics, we ensure the
fractional relationship remains intact in the normalised space. Note
that this is a design choice.

.. code:: yaml

   normaliser:
      remap:
        cp: tp

*********
 Imputer
*********

It is important to have no missing values (e.g. NaNs) in the data when
training a model as this will break the backpropagation of gradients and
cause the model to predict only NaNs. For fields which contain missing
values, we provide options to replace these values via an "imputer".
During training NaN values are replaced with the specified value for the
field. The default imputer is "none", which means no imputation is
performed. The user can specify the imputer by setting
``processors.imputer`` under the ``data/zarr.yaml`` file. It is comon to
impute with the mean value, ensuring that the variable value over NaNs
becomes zero after mean-std normalisation. Another option is to impute
with a given constant.

The ``DynamicInputImputer`` can be used for fields where the NaN
locations change in time.

.. code:: yaml

   imputer:
      default: "none"
      mean:
         - 2t

   processors:
   imputer:
      _target_: anemoi.models.preprocessing.imputer.InputImputer
      config: ${data.imputer}

****************
 Loss Functions
****************

Anemoi Training supports various loss functions for different training
tasks and easily allows for custom loss functions to be added.

.. code:: yaml

   training_loss:
      _target_: anemoi.training.losses.mse.WeightedMSELoss
      # class kwargs

The choice of loss function depends on the model task and the desired
properties of the forecast.

For ensemble training, the following loss functions are available:

-  **Kernel CRPS**: Continuous Ranked Probability Score using kernel
   density estimation
-  **AlmostFairKernelCRPS**: A variant of Kernel CRPS which accounts for
   the number of ensemble members used.

.. _loss-function-scaling:

***********************
 Loss function scaling
***********************

It is possible to change the weighting given to each of the variables in
the loss function by changing the default `pressure_level` and
`general_variable` scalers. They are by default applied to the fields
before applying the training loss function and defined in the
configuration `training.scalers`.

While in the `general_variable` scaler each variable is given a
weighting, the `pressure_level` scaler is applied to the pressure levels
variables with respect to the pressure level. For almost all
applications, upper atmosphere pressure levels should be given lower
weighting than the lower atmosphere pressure levels (i.e. pressure
levels nearer to the surface). By default anemoi-training uses a ReLU
Pressure Level scaler with a minimum weighting of 0.2 (i.e. no pressure
level has a weighting less than 0.2), defined in class
`anemoi.training.losses.scalers.ReluVariableLevelScaler`.

.. code:: yaml

   general_variable:
      _target_: anemoi.training.losses.scalers.GeneralVariableLossScaler
      weights:
         default: 1
         t: 6
         z: 12
         10u: 0.1
         10v: 0.1
         2d: 0.5
         tp: 0.025
         cp: 0.0025

.. code:: yaml

   pressure_level:
      # Variable level scaler to be used
      _target_: anemoi.training.losses.scalers.ReluVariableLevelScaler
      group: pl
      y_intercept: 0.2
      slope: 0.001

The loss is also scaled by assigning a weight to each node on the output
grid. These weights are calculated during graph-creation and stored as
an attribute in the graph object. The configuration option
``config.training.node_loss_weights`` is used to specify the node
attribute used as weights in the loss function. By default
anemoi-training uses area weighting, where each node is weighted
according to the size of the geographical area it represents.

It is also possible to rescale the weight of a subset of nodes after
they are loaded from the graph. For instance, for a stretched grid setup
we can rescale the weight of nodes in the limited area such that their
sum equals 0.25 of the sum of all node weights with the following config
setup

.. code:: yaml

   node_loss_weights:
      _target_: anemoi.training.losses.nodeweights.ReweightedGraphNodeAttribute
      target_nodes: data
      scaled_attribute: cutout
      weight_frac_of_total: 0.25

***************
 Learning rate
***************

Anemoi training uses the ``CosineLRScheduler`` from PyTorch as it's
learning rate scheduler. Docs for this scheduler can be found here
https://github.com/huggingface/pytorch-image-models/blob/main/timm/scheduler/cosine_lr.py
The user can configure the maximum learning rate by setting
``config.training.lr.rate``. Note that this learning rate is scaled by
the number of GPUs with:

.. code:: yaml

   global_learning_rate = config.training.lr.rate * num_gpus_per_node * num_nodes / gpus_per_model

The user can also control the rate at which the learning rate decreases
by setting the total number of iterations -
``config.training.lr.iterations`` and the minimum learning rate reached
- ``config.training.lr.min``. Note that the minimum learning rate is not
scaled by the number of GPUs. The user can also control the warmup
period by setting ``config.training.lr.warmup_t``. If the warmup period
is set to 0, the learning rate will start at the maximum learning rate.
If no warmup period is defined, a default warmup period of 1000
iterations is used.

*********
 Rollout
*********

Rollout training is when the model is iterated within the training
process, producing forecasts for many future time steps. The loss is
calculated on every step in the rollout period and averaged, and
gradients backprogogated through the iteration process.

For example, if using ``rollout=3`` and a model with a 6 hour prediction
step-size, when training the model predicts for time t+1, this is used
as inputs to predict time t+2, and this used to predict time t+3. The
loss is calculated as ``1/3 * ( (loss at t+1) + (loss at t+2) + (loss at
t+3) )`` Rollout training has been shown to improve stability for long
auto-regressive inference runs, by making the training objective is
closer to the use case of forecasting arbitrary lead timestep through
autoreggresive iteration of the model.

In most cases, in the first stage of training, the model is trained for
many epochs to perdict only one step (i.e. rollout.max = 1). Once this
is completed, there is a second stage of training, which uses *rollout*
to fine-tune the model error at longer leadtimes. The model begins with
a rollout loss defined by ``rollout.start``, usually 1, and then every n
epochs (defined by rollout.epoch_increment) the rollout value increases
up till ``rollout.max``.

.. code:: yaml

   rollout:
      start: 1
      # increase rollout every n epochs
      epoch_increment: 1
      # maximum rollout to use
      max: 12

This two stage approach requires the model training to be restarted
after stage one, see instructions below. The user should make sure to
set ``config.training.run_id`` equal to the run-id of the first stage of
training.

Note, for many purposes, it may make sense for the rollout stage (stage
two) to performed at the minimum learning rate throughout and for the
number of batches to be reduced (using
``config.dataloader.training.limit_batches``) to prevent overfit to
specific timesteps.

***************************
 Restarting a training run
***************************

It may be necessary at certain points to restart the model training,
i.e. because the training has exceeded the time limit on an HPC system
or because the user wants to fine-tune the model from a specific point
in the training.

This can be done by setting ``config.training.run_id`` in the config
file to be the *run_id* of the run that is being restarted. In this case
the new checkpoints will go in the same folder as the old checkpoints.
If the user does not want this then they can instead set
``config.training.fork_run_id`` in the config file to the *run_id* of
the run that is being restarted. In this case the old run will be
unaffected and the new checkpoints will go in to a new folder with a new
run_id. The user might want to do this if they want to start multiple
new runs from 1 old run.

The above will restart the model training from where the old run
finished training. It's also possible to restart the model training from
a specific checkpoint. This can either be a checkpoint from the same run
or a checkpoint from a different run that you have run in the past or
that you using for transfer learning. To do this, set
``config.hardware.files.warm_start`` to be the checkpoint filename they
want to restart from and ``config.hardware.paths.warm_start`` to be the
path to the checkpoint. See the example below.

.. code:: yaml

   # This is a sample YAML block
   hardware:
      files:
         warm_start: checkpoint_epoch_10.ckpt
      paths:
         warm_start: /path/to/checkpoint/folder/

The above can be adapted depending on the use case and taking advantage
of hydra, you can also reuse ``config.training.run_id`` or
``config.training.fork_run_id`` to define the path to the checkpoint.

*******************
 Transfer Learning
*******************

Transfer learning allows the model to reuse knowledge from a previously
trained checkpoint. This is particularly useful when the new task is
related to the old one, enabling faster convergence and often improving
model performance.

To enable transfer learning, set the config.training.transfer_learning
flag to True in the configuration file.

.. code:: yaml

   training:
      # start the training from a checkpoint of a previous run
      fork_run_id: ...
      load_weights_only: True
      transfer_learning: True

When this flag is active and a checkpoint path is specified in
config.hardware.files.warm_start or self.last_checkpoint, the system
loads the pre-trained weights using the `transfer_learning_loading`
function. This approach ensures only compatible weights are loaded and
mismatched layers are handled appropriately.

For example, transfer learning might be used to adapt a weather
forecasting model trained on one geographic region to another region
with similar characteristics.

****************
 Model Freezing
****************

Model freezing is a technique where specific parts (submodules) of a
model are excluded from training. This is useful when certain parts of
the model have been sufficiently trained or should remain unchanged for
the current task.

To specify which submodules to freeze, use the
config.training.submodules_to_freeze field in the configuration. List
the names of submodules to be frozen. During model initialization, these
submodules will have their parameters frozen, ensuring they are not
updated during training.

For example with the following configuration, the processor will be
frozen and only the encoder and decoder will be trained:

.. code:: yaml

   training:
      # start the training from a checkpoint of a previous run
      fork_run_id: ...
      load_weights_only: True

      submodules_to_freeze:
         - processor

Freezing can be particularly beneficial in scenarios such as fine-tuning
when only specific components (e.g., the encoder, the decoder) need to
adapt to a new task while keeping others (e.g., the processor) fixed.

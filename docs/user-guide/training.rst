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

#. **Graph Definition from Anemoi Graphs:** This defines the structure
   of your machine learning model, including the layers, connections,
   and operations that will be used during training.

#. **Dataset from Anemoi Datasets:** This provides the training data
   that will be fed into the model. The dataset should be pre-processed
   and formatted according to the specifications of the Anemoi Datasets
   module.

These 2 steps are outlined in `the start guide
<start/training-your-first-model>`_.

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
-  Optionally, customize additional components like the normaliser or
   optimization strategies to enhance model performance.

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

-  Run the training command, ensuring that all paths to the graph
   definition and dataset are correctly specified.
-  Monitor the training process, adjusting parameters as needed to
   optimize model performance.
-  Upon completion, the trained model will be registered and stored for
   further use.

Then you make sure you have a GPU available and simply call:

.. code:: bash

   anemoi-training train

.. _restart target:

**************
 Data Routing
**************

Anemoi Training uses the Anemoi Datasets module to load the data. The
dataset contains the entirety of variables we can use for training.
Initial experiments in data-driven weather forecasting have used the
same input variables as output variables.

Anemoi training implements data routing, in which you can specify which
variables are used as ``forcings`` in the input only to inform the
model, and which variables are used as ``diagnostics`` in the output
only to be predicted by the model. All remaining variables will be
treated as ``prognostic`` in the intial and forecast states.

Intuitively, ``forcings`` are the variables like solar insolation or
land-sea-mask. These would make little sense to predict as they are
external to the model. ``Diagnostics`` are the variables like
precipitation that we want to predict, but which may not be available in
forecast step zero due to technical limitations. ``Prognostic``
variables are the variables like temperature or humidity that we want to
predict and are available after data assimilation operationally.

The user can specify the routing of the data by setting the
``config.data.forcings`` and ``config.data.diagnostics``. These are
named strings, as Anemoi datasets enables us to address variables by
name.

This can look like the following:

.. code:: yaml

   data:
      forcings:
         - solar_insolation
         - land_sea_mask
      diagnostics:
         - total_precipitation

***************
 Normalisation
***************

Machine learning models are sensitive to the scale of the input data. To
ensure that the model can learn effectively, it is important to
normalise the input data.

Anemoi training provides preprocessors for different aspects of the
training, with the normaliser being one of them. The normaliser
implements multiple strategies that can be applied to the data using the
config.

Currently, the normaliser supports the following strategies:

-  ``none``: No normalisation is applied.
-  ``mean-std``: Standard normalisation is applied to the data.
-  ``min-max``: Min-max normalisation is applied to the data.
-  ``max``: Max normalisation is applied to the data.

Values like the land-sea-mask do not require additional normalisation.
However, variables like temperature or humidity should be normalised to
ensure the model can learn effectively. Additionally, variables like the
geopotential height should be max normalised to ensure the model can
learn the vertical structure of the atmosphere.

The user can specify the normalisation strategy, including the default
by setting ``config.data.normaliser``, such that:

.. code:: yaml

   normaliser:
      default: mean-std
      none:
         - land_sea_mask
      max:
         - geopotential_height

***********************
 Loss function scaling
***********************

It is possible to change the weighting given to each of the variables in
the loss function by changing
``config.training.loss_scaling.pl.<pressure level variable>`` and
``config.training.loss_scaling.sfc.<surface variable>``.

It is also possible to change the scaling given to the pressure levels
using ``config.training.pressure_level_scaler``. For almost all
applications, upper atmosphere pressure levels should be given lower
weighting than the lower atmosphere pressure levels (i.e. pressure
levels nearer to the surface). By default anemoi-training uses a ReLU
Pressure Level scaler with a minimum weighting of 0.2 (i.e. no pressure
level has a weighting less than 0.2).

***************
 Learning rate
***************

Anemoi training uses the ``CosineLRScheduler`` from PyTorch as it's
learning rate scheduler. The user can configure the maximum learning
rate by setting ``config.training.lr.rate``. Note that this learning
rate is scaled by the number of GPUs where for the `data parallelism
<distributed>`_.

.. code:: yaml

   global_learning_rate = config.training.lr.rate * num_gpus_per_node * num_nodes / gpus_per_model

The user can also control the rate at which the learning rate decreases
by setting the total number of iterations through
``config.training.lr.iterations`` and the minimum learning rate reached
through ``config.training.lr.min``. Note that the minimum learning rate
is not scaled by the number of GPUs.

*********
 Rollout
*********

In the first stage of training, standard practice is to train the model
on a 6 hour interval. Once this is completed, in the second stage of
training, it is advisable to *rollout* and fine-tune the model error at
longer leadtimes too. Generally for medium range forecasts, rollout is
performed on 12 forecast steps (equivalent to 72 hours) incrementally.
In other words, at each epoch another forecast step is added to the
error term.

Rollout requires the model training to be restarted so the user should
make sure to set ``config.training.run_id`` equal to the run-id of the
first stage of training.

Note, in the standard set-up, rollout is performed at the minimum
learning rate and the number of batches used is reduced (using
``config.dataloader.training.limit_batches``) to prevent any overfit to
specific timesteps.

To start rollout set ``config.training.rollout.epoch_increment`` equal
to 1 (thus increasing the rollout step by 1 at every epoch) and set a
maximum rollout by setting ``config.training.rollout.max`` (usually set
to 12).

***************************
 Restarting a training run
***************************

Whether it's because the training has exceeded the time limit on an HPC
system or because the user wants to fine-tune the model from a specific
point in the training, it may be necessary at certain points to restart
the model training.

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
finished training. However if the user wants to restart the model from a
specific point they can do this by setting
``config.hardware.files.warm_start`` to be the checkpoint they want to
restart from..

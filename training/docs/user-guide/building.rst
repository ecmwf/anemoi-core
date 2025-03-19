##############
 Building Configs for Training
##############

Anemoi training is designed so you can adjust key parts of the models
and training process without needing to modify the underlying code.

A basic introduction to the configuration system is provided in the
`getting started <start/hydra-intro>`_ section. This section will go
into more detail on how to configure the training pipeline.

***********************
 Default Config Groups
***********************

A typical config file will start with specifying the default config
settings at the top as follows:

.. code:: yaml

   defaults:
   - data: zarr
   - dataloader: native_grid
   - diagnostics: evaluation
   - hardware: example
   - graph: multi_scale
   - model: gnn
   - training: default
   - _self_

These are group configs for each section. The options after the defaults
are then used to override the configs, by assigning new features and
keywords.

You can also find these defaults in other configs, like the
``hardware``, which implements:

.. code:: yaml

   defaults:
   - paths: example
   - files: example

***************************************
 Data
***************************************

[format: zarr
# Time frequency requested from dataset
frequency: 6h
# Time step of model (must be multiple of frequency)
timestep: 6h
Remapper?]

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


Normalisation
============


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
-  ``std``: Data is normalised by the standard deviation.
-  ``min-max``: Data is normalised by substracting the min value and dividing by the range.
-  ``max``: Data is normalised by the max value.

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

An additional option in the normaliser overwrites statistics of specific variables onto others.
This is primarily used for convective precipitation (cp), which is a fraction of total precipitation (tp),
by overwriting the cp statistics with the tp statistics, we ensure the fractional relationship
remains intact in the normalised space. Note that this is a design choice.

.. code:: yaml

   normaliser:
      remap:
        cp: tp


Imputer
============

It is important to have no missing values (e.g. NaNs) in the data when training a model as this will break the backpropagation
of gradients and cause the model to predict only NaNs. For fields which contain missing values,
we provide options to replace these values via an "imputer". During training NaN values are replaced with the specified value
for the field. The default imputer is "none", which means no imputation is performed. The user can specify the imputer by setting
``processors.imputer`` under the ``data/zarr.yaml`` file. An example of this is shown below:

.. code:: yaml

   imputer:
      default: "none"
      mean:
         - 2t

   processors:
   imputer:
      _target_: anemoi.models.preprocessing.imputer.InputImputer
      _convert_: all
      config: ${data.imputer}

There are other options for the imputer; constant values can by used, or the ``DynamicInputImputer`` can be used for fields where the
NaN locations change in time.

***************************************
 Dataloader
***************************************

The dataloader file points...

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

# set a custom mask for grid points.
# Useful for LAM (dropping unconnected nodes from forcing dataset)
grid_indices:
  _target_: anemoi.training.data.grid_indices.FullGrid
  nodes_name: ${graph.data}

dataset: ${hardware.paths.data}/${hardware.files.dataset}

training:
  dataset: ${dataloader.dataset}
  start: null
  end: 2020
  frequency: ${data.frequency}
  drop:  []

validation_rollout: 1 # number of rollouts to use for validation, must be equal or greater than rollout expected by callbacks

validation:
  dataset: ${dataloader.dataset}
  start: 2021
  end: 2021
  frequency: ${data.frequency}
  drop:  []

test:
  dataset: ${dataloader.dataset}
  start: 2022
  end: null
  frequency: ${data.frequency}
  drop:  []

***************************************
 Diagnostics
***************************************

***************************************
 Graph
***************************************

***************************************
 Hardware
***************************************

***************************************
 Model
***************************************
The user can pick between three different model types, when using
anemoi-training:

#. Graph Neural Network (GNN)
#. Graph Transformer Neural Network
#. Transformer Neural Network

Currently, all models have a Encoder-Processor-Decoder structure, with
physical data being encoded on to a latent space where the processing
takes place.

For a more detailed read on connections in Graph Neural Networks,
`Velickovic (2023) <https://arxiv.org/pdf/2301.08210>`_ is recommended.


 Processors
============

The processor is the part of the model that performs the computation on
the latent space. The processor can be chosen to be a GNN,
GraphTransformer or Transformer with Flash attention.

**GNN**

The GNN structure is similar to that user in Keisler (2022) and Lam et
al. (2023).

The physical data is encoded on to a multi-mesh latent space of
decreasing resolution. This multi-mesh is defined by the graph given in
``config.hardware.files.graph``.

.. figure:: ../images/gnn-encoder-decoder-multimesh.jpg
   :width: 500
   :align: center

   GNN structure

On the processor grid, information passes between the node embeddings
via simultaneous multi-message-passing. The messages received from
neighboring nodes are a function of their embeddings from the previous
layer and are aggregated by summing over the messages received from
neighbours. The data is then decoded by the decoder back to a single
resolution grid.

**Graph Transformer**

The GraphTransformer uses convolutional multi-message passing on the
processor. In this case, instead of the messages from neighbouring nodes
being weighted equally (as in the case for GNNs), the GNN can learn
which node embeddings are important and selectively weight those more
through learning the `attention weight` to give to each embedding.

Note that here, the processor grid is a single resolution whih is
coarser than the resolution of the base data.

**Transformer**

The Transformer uses a multi-head self attention on the processor. Note
that this requires `flash-attention
<https://github.com/Dao-AILab/flash-attention>`__ to be installed.

Thhe attention windows are chosen in such a way that a complete grid
neighbourhood is always included (see Figure below). Like with the
GraphTransformer, the processor grid is a single resolution which is
coarser than the resolution of the base data.

.. figure:: ../images/global-sliding-window-attention.png
   :width: 500
   :align: center

   Attention windows (grid points highlighted in blue) for different grid points (red).


Encoders/Decoders
============

The encoder and decoder can be chosen to be a GNN or a GraphTransformer.
This choice is independent of the processor, but currently the encoder
and decoder must be the same model type otherwise the code will break,

***************************************
 Training
***************************************

Loss function scaling
============

It is possible to change the weighting given to each of the variables in
the loss function by changing
``config.training.variable_loss_scaling.pl.<pressure level variable>``
and ``config.training.variable_loss_scaling.sfc.<surface variable>``.

It is also possible to change the scaling given to the pressure levels
using ``config.training.pressure_level_scaler``. For almost all
applications, upper atmosphere pressure levels should be given lower
weighting than the lower atmosphere pressure levels (i.e. pressure
levels nearer to the surface). By default anemoi-training uses a ReLU
Pressure Level scaler with a minimum weighting of 0.2 (i.e. no pressure
level has a weighting less than 0.2).

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


Learning rate
============

Anemoi training uses the ``CosineLRScheduler`` from PyTorch as it's
learning rate scheduler. Docs for this scheduler can be found here
https://github.com/huggingface/pytorch-image-models/blob/main/timm/scheduler/cosine_lr.py
The user can configure the maximum learning rate by setting
``config.training.lr.rate``. Note that this learning rate is scaled by
the number of GPUs where for the `data parallelism <distributed>`_.

.. code:: yaml

   global_learning_rate = config.training.lr.rate * num_gpus_per_node * num_nodes / gpus_per_model

The user can also control the rate at which the learning rate decreases
by setting the total number of iterations through
``config.training.lr.iterations`` and the minimum learning rate reached
through ``config.training.lr.min``. Note that the minimum learning rate
is not scaled by the number of GPUs. The user can also control the
warmup period by setting ``config.training.lr.warmup_t``. If the warmup
period is set to 0, the learning rate will start at the maximum learning
rate. If no warmup period is defined, a default warmup period of 1000
iterations is used.

Rollout
============

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

Restarting a training run
============

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


Transfer Learning
============

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


Model Freezing
============

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

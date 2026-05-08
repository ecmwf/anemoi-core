##############
 Basic Set-Up
##############

Once Anemoi training is installed, you can test your installation by
running

.. code:: bash

   anemoi-training train

which will use the default model configurations with missing values to
lauch a training run.

The training script will intentionally crash as it does not know where
your data is stored.

These missing values in the configuration are placeholders for the user
to fill in marked with ``???``. You can find the default configurations
in the `Anemoi Training repository
<https://github.com/ecmwf/anemoi-training>`_ under
``src/anemoi/training/config/``. Alternatively, the next section will
show you how to :ref:`generate a user config file <Configuring the
Training>`.

.. _prep-training-components:

*******************************
 Preparing training components
*******************************

Anemoi Training requires two primary components to get started:

#. **Graph Definition** from `Anemoi Graphs
   <https://anemoi-graphs.readthedocs.org>`_: This defines the structure
   of your machine learning model, including the layers, connections,
   and operations that will be used during training.

#. **Dataset** from `Anemoi Datasets
   <https://anemoi-datasets.readthedocs.org>`_ : This provides the
   training data that will be fed into the model. The dataset should be
   pre-processed and formatted according to the specifications of the
   Anemoi Datasets module.

In many cases, your organisation may already have some of these
prepared, but if not, you can follow the steps below to prepare them.

Step 1: Prepare Your Graph Definition
=====================================

The first step before training your model is to select or create a graph
definition. Anemoi Graphs provides a flexible way to define the
architecture of your machine learning model. You can either use a
predefined graph that suits common forecasting tasks or customize your
own to better fit your specific requirements.

The graph defines the connectivity between your datapoints and the
operations that will be performed on them during training. The data flow
and "learnable parameters" are defined in the model itself, so the graph
is a high-level description of the "connecitivity" within the model
architecture.

To prepare your graph:

-  Choose an existing graph definition from Anemoi Graphs or define a
   new one that matches your forecasting needs.
-  Ensure that the graph is compatible with the input data format and
   the type of weather forecasting problem you are addressing.

Step 2: Prepare Your Dataset
============================

The next step is to prepare your dataset using Anemoi Datasets. This
dataset will be used to train the model defined by your graph.

To prepare your dataset:

-  Select a dataset from Anemoi Datasets or create a new dataset that
   includes all relevant weather variables and historical data points.
-  Pre-process the data to ensure it is in the correct format and
   structure required by the graph definition.

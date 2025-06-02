##########
 Overview
##########

Anemoi is a comprehensive framework designed to streamline the
development of machine learning models for data-driven weather
forecasting. The anemoi-models package lies at the core of this
framework, providing core components to train graph neural networks
(GNNs), including graph transformers for data-driven weather
forecasting.

The anemoi-models package has the following dependencies in the code:

.. figure:: ../_static/anemoi-models_schematic.png
   :alt: Dependencies and initial structure of anemoi models
   :align: center

Below is a detailed breakdown of these main components within the
anemoi-models package:

***********
 Interface
***********

The `AnemoiModelInterface` is designed to provide an interface between
the training and the model itself. This code is used for making
predictions with an underlying machine learning model. It implements the
interface for pre-processing input data, performing inference using the
model, and post-processing the output.

These components can be extended and switched out with the config files.

This modular approach allows for easy customization and extension,
facilitating the use of different models and processors as needed.

***************
 Preprocessing
***************

The `preprocessing` module provides a set of tools and utilities for
preprocessing input data for the model. This includes normalizing the
input data and different imputation methods. The preprocessing module is
designed to be flexible and modular, allowing for easy customization and
extension.

This is achieved through a config that can provide a list of
preprocessing steps to be applied to the input data.

Currently the package includes the following preprocessing steps:

-  Normalization
-  Imputation

*******
 Model
*******

The `models` module is the core component of the anemoi-models package,
defining various model architectures to work with graph data. The models
are designed to be flexible and modular, allowing for easy customization
and extension through configuration.

The package currently includes the following model architectures:

-  **AnemoiModelEncProcDec**: The base encoder-processor-decoder
   architecture, e.g. AIFS-single
-  **AnemoiModelEncProcDecHierarchical**: A hierarchical variant of the
   base architecture
-  **AnemoiEnsModelEncProcDec**: The CRPS-optimized ensemble version
   that injects noise in the processor, e.g. AIFS-CRPS
-  **AnemoiModelEncProcDecInterpolator**: A specialized architecture for
   time interpolation

All models support flexible layer kernel configuration, allowing for
customization of linear and normalization layers in different parts of
the model.

********
 Layers
********

The `layers` module provides the core building blocks for the neural
network in `Models`. This includes graph transformers, graph
convolutions, and transformers, as well as, other layers that are used
to process the input data.

The layers are designed as extensible classes to allow for easy
experimentation and switching out of components.

Mappers
=======

The layers implement `Mappers`, which maps data between the input grid
and the internal hidden grid. The `Mappers` are used as encoder and
decoder. The `Mappers` use the `Blocks` to process the data, which are
the building blocks of the graph neural network.

Processors
==========

Additionally, the layers implement `Processors` which are used to
process the data on the hidden grid. The `Processors` use a chunking
strategy with `Chunks` that pass a subset of layers to `Blocks` to allow
for more efficient processing of the data.

**************
 Data Indices
**************

Throughout *anemoi models* and *anemoi training* we use the concept of
data indices to refer to the indices of the data and provide of the full
training data.

Specifically, this enables data routing of variables that are only used
as `forcing` variables in the input to the model, `diagnostic` which is
only an output of the model, and `prognostic` variables that are both
present in the input and the output.

*************
 Distributed
*************

The `distributed` module provides utilities for distributed training of
the model. This includes includes the splitting and gathering of the
model and its tensors / parameters across multiple GPUs. This process is
also known as "model shardings".

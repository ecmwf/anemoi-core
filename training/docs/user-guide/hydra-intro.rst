.. _hydra-intro:

######################
 Configuration Basics
######################

Anemoi training is set up in a way that you should be able to modify key
components of both the models and training without changing the code.

This configuration is achieved by using the `Hydra
<https://hydra.cc/>`__ config system.

Hydra allows for the creation of a structured configuration that can be
overridden from the command line. This allows for the creation of
configurable models and training pipelines. All while keeping the code
clean and easy to read.

Additionally, Hydra allows us to keep track of config changes and
command line overrides. This is useful for debugging and reproducing
results.

Without even generating a config file, you can try and run the training
script with the default settings:

.. code:: bash

   anemoi-training train

This will run the training script with the default settings. These
settings contain some missing values, which will intentionally crash, as
we don't know where your data is stored. This is where the config file
comes in.

******************************
 Generating User Config Files
******************************

Anemoi training provides a command line interface to generate a user
config file. This can be done by running:

.. code:: bash

   anemoi-training config generate

This will create a new config file in the current directory. The user
can then modify this file to suit their needs.

These config files are YAML files, which can be easily read and
modified.

They are split across files based on topic. For example, the hardware
config is in the hardware folder. The model config is in the model
folder.

You will need to specify the location of your anemoi dataset in the
hardware paths and files. These contain ``???`` as placeholders.

Anemoi training provides two default configurations ``config.yaml`` and
``debug.yaml``. The first is a generic config file, while the second is
used for debugging purposes, with a smaller run and fewer epochs.

In order to use the debug config file ``--config-name=debug`` should be
added to the training command like so:

.. code:: bash

   anemoi-training train --config-name=debug

****************************
 Important Config Overrides
****************************

The following missing config options which must be overridden by users:

-  ``hardware.paths.data``: Location of base directory where datasets
   are stored

-  ``hardware.paths.graph``: Location of graph directory

-  ``hardware.paths.output``: Location of output directory

-  ``hardware.files.dataset``: Filename(s) of datasets used for training

-  ``hardware.files.graph``: If you have pre-computed a specific graph,
   specify its filename here. Otherwise, a new graph will be constructed
   on the fly and written to the filename given.

*********************************
 Validation of the Configuration
*********************************

The configuration is validated using `Pydantic` before a training run
starts. To turn this off, you can use the `--no-validation` flag in your
top-level config. For more information on validation, see
:ref:`config-validation`.

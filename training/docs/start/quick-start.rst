########################
 Train your first model
########################

Once Anemoi training is installed, run the following in command to
generate config files.

Anemoi training provides a command line interface to generate a user
config file. This can be done by running:

.. code:: bash

   anemoi-training config generate

This will create a new config file in the current directory. The user
can then modify this file to suit their needs.

   you can run your first model with

.. code:: bash

   anemoi-training train --config-name=firstmodel.yaml

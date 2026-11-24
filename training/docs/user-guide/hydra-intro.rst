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

They are split across files based on topic. For example, the system
config includes the sections related to hardware specification, inputs
(such as datasets used for training) and outputs (where to store logs,
plots, checkpoints, etc.). The model config is in the model folder.

You will need to specify the location of your anemoi dataset in the
system input section. These contain ``???`` as placeholders.

Anemoi training provides two default configurations ``config.yaml`` and
``debug.yaml``. The first is a generic config file, while the second is
used for debugging purposes, with a smaller run and fewer epochs.

In order to use the debug config file ``--config-name=debug`` should be
added to the training command like so:

.. code:: bash

   anemoi-training train --config-name=debug

Generating Only the Top-Level Configs
=====================================

If you would prefer to keep all of your changes in a single file, you can
generate only the top-level configs using the ``--top-level-only`` flag:

.. code:: bash

   anemoi-training config generate --top-level-only

This writes the top-level YAML files, such as ``config.yaml`` and
``lam.yaml``, but none of the config group directories (``data/``,
``model/``, ``training/``, ...).

When composing a config, Hydra looks for each file first in the
directory you run from (or the directory given with ``--config-path``),
and then in the configs shipped with the installed ``anemoi-training``
package. As the group directories are not present locally, every group
listed in the ``defaults`` of your top-level config is loaded from the
package. You can customise them from the top-level file, either by
selecting a different option in the ``defaults`` list or by overriding
individual values in the body of the file, e.g.:

.. code:: yaml

   defaults:
   - data: zarr
   - dataloader: native_grid
   - diagnostics: evaluation
   - system: example
   - graph: multi_scale
   - model: transformer   # select a different model config (current default is gnn)
   - task: forecaster
   - training: single
   - _self_

   training:
      max_epochs: 10  # override a single value from training/single.yaml

The available options for each group can be browsed in the
``anemoi/training/config`` directory of the installed package, or in the
`anemoi-core repository
<https://github.com/ecmwf/anemoi-core/tree/main/training/src/anemoi/training/config>`__.

Because the group configs always come from the installed package,
migrating to a new version of ``anemoi-training`` only requires updating
the overrides and ``defaults`` in your top-level file. This also means
that defaults you have not overridden **can change** when you upgrade to
a new version of ``anemoi-training``.

To keep a record of the composed config, write it to a separate file
with ``anemoi-training config dump``:

.. code:: bash

   anemoi-training config dump --config-name config --output config_dump.yaml

Do not dump to ``config.yaml`` in the same directory since it is the default
output path and is overwritten without confirmation. Note that interpolations
such as ``${...}`` are written unresolved.

Some users may prefer to use the normal ``generate`` command alongside the
``--top-level-only`` flag as a reference, to see the full set of default
configs without having to search through where the code is installed.
In this case, you can use the normal ``generate`` command with the
``--output`` (``-o``) flag to write all configs into a separate directory,
e.g.:

.. code:: bash

   cd <directory>
   anemoi-training config generate --top-level-only
   anemoi-training config generate --output=reference_configs

Hydra does not search inside ``reference_configs/``, so its files are
not used for training. Do not name this directory ``config``: if a
``config`` directory exists, Hydra searches it instead of the directory
you run from. Your top-level files are ignored, and the group files
inside ``config/`` are used instead of the packaged ones. Also avoid config
group names (``data``, ``model``, ``training``, ...), as the reference
files would then show up as extra options for that group. Note that the
copy is not updated when you upgrade ``anemoi-training``. To refresh it,
delete the directory and generate it again.

.. note::

   Local config group directories take precedence over the package. If
   group directories already exist where you run from, for example from
   an earlier full ``anemoi-training config generate``, their files are
   used instead of the packaged ones. ``--top-level-only`` does not
   remove existing directories.

   This is decided file by file: with a local ``model/gnn.yaml``,
   ``model: gnn`` uses the local file, while ``model: transformer`` is
   still loaded from the package. A local file replaces the packaged file
   completely; the two are not merged. Remove any leftover group
   directories if you want the packaged configs to be used.

****************************
 Important Config Overrides
****************************

The following missing config options which must be overridden by users:

-  ``system.output.root``: Location of output directory

-  ``system.input.dataset``: Filename(s) of datasets used for training

-  ``system.input.graph``: If you have pre-computed a specific graph,
   specify its filename here. Otherwise, a new graph will be constructed
   on the fly and written to the filename given.

For datsets you can either pass an absolute path or a relative path
(excluding the file format (ex: .zarr)). This functionality requires to
have the settings.toml correctly configured `see anemoi-datasets docs
for more information on how to do so
<https://anemoi.readthedocs.io/projects/datasets/en/latest/using/configuration.html#configuration>`__

Root can be left empty (if set to Null or "") then the current working
directory will be used. Output paths for checkpoints, logs, profiler
outputs and plots are built using the root path as base.

*********************************
 Validation of the Configuration
*********************************

The configuration is validated using `Pydantic` before a training run
starts. To turn this off, you can use the `--no-validation` flag in your
top-level config. For more information on validation, see
:ref:`config-validation`.

##############################
 Basic Training Configuration
##############################

Anemoi training is designed so you can adjust key parts of the models
and training process without needing to modify the underlying code.

A basic introduction to the configuration system is provided in
:ref:`hydra-intro`. This section will go into more detail on how to
configure the training pipeline.

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

*****************************
 YAML-based config overrides
*****************************

The config files are written in YAML format. This allows for easy
overrides of the default settings. For example, to change the model from
the default GNN to a transformer, you can use the following config in
the config groups.:

.. code:: yaml

   model: transformer

This will override the default model config with the transformer model.

You can also override individual settings. For example, to change the
learning rate from the default value of 0.625e-4 to 1e-3, you can add
the following to the config you're using:

.. code:: yaml

   training:
      lr:
         rate: 1e-3

You can also change the GPU count to whatever you have available:

.. code:: yaml

   hardware:
       num_gpus_per_node: 1

This matches the interface of the underlying defaults in Anemoi
training.

Example Config File
===================

Here is an example of a config file that changes the model to a
transformer, the learning rate to 1e-3, and the number of GPUs to 1. We
also need to specify the paths to the data, output, and graph data and
give the names of the files to use. You can get a dataset from the
`Anemoi Datasets catalogue <https://anemoi.ecmwf.int/>`_ or create one
using the `Anemoi Datasets
<https://anemoi-datasets.readthedocs.io/en/latest/>`_ package.

You can create a graph using `Anemoi Graphs
<https://anemoi-graphs.readthedocs.io/en/latest/>`_ or one will be
created for you at runtime. Note that you must specify a filename for
the graph, here we use `first_graph_m320.pt`.

You'll also notice we've specified a resolution for the data, this must
match the dataset you provide.

.. code:: yaml

   defaults:
   - data: zarr
   - dataloader: native_grid
   - diagnostics: evaluation
   - hardware: example
   - graph: multi_scale
   - model: transformer # Change from default group
   - training: default
   - _self_

   config_validation: True
   data:
      resolution: n320

   hardware:
      num_gpus_per_node: 1
      paths:
         output: /home/username/anemoi/training/output
         data: /home/username/anemoi/datasets
         graph: /home/username/anemoi/training/graphs
      files:
         dataset: datset-n320-2019-2021-6h.zarr
         graph: first_graph_n320.pt

   training:
      lr:
         rate: 1e-3

When we save this `example.yaml` file, we can run the training with this
config using:

.. code:: bash

   anemoi-training train --config-name=example.yaml

*******************************
 Command-line config overrides
*******************************

It is also possible to use command line config overrides. We can switch
out group configs using

.. code:: bash

   anemoi-training train model=transformer

or override individual config entries such as

.. code:: bash

   anemoi-training train diagnostics.plot.enabled=False

or combine everything together

.. code:: bash

   anemoi-training train --config-name=debug.yaml model=transformer diagnostics.plot.enabled=False

.. _config-validation:

*******************
 Config validation
*******************

It is possible to validate your configuration before starting a training
run using the following command:

.. code:: bash

   anemoi-training config validate --config-name debug.yaml

This will check that the configuration is valid and that all the
required fields are present. If your config is correctly defined then
the command will show an output similar to:

.. code:: bash

   2025-01-28 09:37:23 INFO Validating configs.
   2025-01-28 09:37:23 INFO Prepending Anemoi Home (/home_path/.config/anemoi/training/config) to the search path.
   2025-01-28 09:37:23 INFO Prepending current user directory (/repos_path/config_anemoi_core) to the search path.
   2025-01-28 09:37:23 INFO Search path is now: [provider=anemoi-cwd-searchpath-plugin, path=/repos_path/config_anemoi_core, provider=anemoi-home-searchpath-plugin, path=/home_path/.config/anemoi/training/config, provider=hydra, path=pkg://hydra.conf, provider=main, path=/repos_path/anemoi-core/training/src/anemoi/training/commands]
   cfg = BaseSchema(**cfg)
   2025-01-28 09:37:23 INFO Config files validated.

Otherwise if there is an issue with some of your configuration fields,
Pydantic will report an error message. If your config is missing the
definition of a required field, then the validation will also fail. This
can be the case if you pull the defaults anemoi configs and do not
replace the empty fields (usually represented by '??') with the actual
values. Similarly if you have a field that is expected to use an
environment variable and you do not have it set, the validation will
fail. To overcome this issue and still be able to validate the config,
you can use the `----mask_env_vars` flag, which will skip the validation
of the environment variables. When using this flag, the validation will
still be performed, but the environment variables will be masked with
the default values. See below an example output where we have 5
environment variables that are not set and are masked with the default
values:

.. code:: bash

   (anemoi_core_venv)[] $ anemoi-training config validate --config-name=debug --mask_env_vars
   2025-02-16 17:48:38 INFO Validating configs.
   2025-02-16 17:48:38 WARNING Note that this command is not taking into account if your config has
   set the config_validation flag to false.So this command will validate the config regardless of the flag.
   2025-01-28 09:37:23 INFO Prepending Anemoi Home (/home_path/.config/anemoi/training/config) to the search path.
   2025-01-28 09:37:23 INFO Prepending current user directory (/repos_path/config_anemoi_core) to the search path.
   2025-01-28 09:37:23 INFO Search path is now: [provider=anemoi-cwd-searchpath-plugin, path=/repos_path/config_anemoi_core, provider=anemoi-home-searchpath-plugin, path=/home_path/.config/anemoi/training/config, provider=hydra, path=pkg://hydra.conf, provider=main, path=/repos_path/anemoi-core/training/src/anemoi/training/commands]
   2025-02-16 17:48:39 WARNING Environment variable EXP_NAME not found, masking with default
   2025-02-16 17:48:39 WARNING Environment variable RUN_NAME not found, masking with default
   2025-02-16 17:48:39 WARNING Environment variable SLURM_GPUS_PER_NODE not found, masking with 0
   2025-02-16 17:48:39 WARNING Environment variable SLURM_NNODES not found, masking with 0
   2025-02-16 17:48:39 WARNING Environment variable LOCAL_LR not found, masking with 0
   2025-02-16 17:48:39 INFO Config files validated.

See example below where we have a `debug.yaml` file with a field not
correctly indented (in this case the `diagnostics.log` field):

.. code:: yaml

   defaults:
   - data: zarr
   - dataloader: native_grid
   - diagnostics: evaluation
   - hardware: example
   - graph: multi_scale
   - model: transformer # Change from default group
   - training: default
   - _self_


   diagnostics:
   log:
   mlflow:
      enabled: True
      offline: True
      experiment_name: 'test'
      project_name: 'AIFS'
      run_name: 'test_anemoi_core'
      tracking_uri: 'https://mlflow-server.int'
      authentication: True
      terminal: True

If we try to validate the above then the validate command will report
the following error:

.. code:: python

   2025-01-28 09:37:23 INFO Validating configs.
   2025-01-28 09:37:23 INFO Prepending Anemoi Home (/home_path/.config/anemoi/training/config) to the search path.
   2025-01-28 09:37:23 INFO Prepending current user directory (/repos_path/config_anemoi_core) to the search path.
   2025-01-28 09:37:23 INFO Search path is now: [provider=anemoi-cwd-searchpath-plugin, path=/repos_path/config_anemoi_core, provider=anemoi-home-searchpath-plugin, path=/home_path/.config/anemoi/training/config, provider=hydra, path=pkg://hydra.conf, provider=main, path=/repos_path/anemoi-core/training/src/anemoi/training/commands]
   pydantic_core._pydantic_core.ValidationError: 1 validation error for BaseSchema
   diagnostics.log
    Input should be a valid dictionary or instance of LoggingSchema [type=model_type, input_value=None, input_type=NoneType]
      For further information visit https://errors.pydantic.dev/2.10/v/model_type
   2025-01-28 09:54:08 ERROR
   💣 1 validation error for BaseSchema
   diagnostics.log
   Input should be a valid dictionary or instance of LoggingSchema [type=model_type, input_value=None, input_type=NoneType]
      For further information visit https://errors.pydantic.dev/2.10/v/model_type
   2025-01-28 09:54:08 ERROR 💣 Exiting

Which indicates that the `diagnostics.log` field is not correctly
defined as it should be a dictionary or instance of `LoggingSchema`.
Please note there might still be cases not captured by the current
schemas, so it is always good to double check the configuration file
before running the training. See below an example of a config with some
typos that might still need to be fixed manually:

.. code:: yaml

   defaults:
   - data: zarr
   - dataloader: native_grid
   - diagnostics: evaluation
   - hardware: example
   - graph: multi_scale
   - model: transformer # Change from default group
   - training: default
   - _self_


   diagnostics:
   log:
      mlflow:
         enabled: True
         ofline: True # this is a typo - should be offline
         experiment_name: 'test'
         project_name: 'AIFS'
         run_name: 'test_anemoi_core'
         tracking_uri: 'https://mlflow-server.int'
         authentication: True
         terminal: True

In the example above, if there is a default already defined for
`offline` under `diagnostics: evaluation` then the validation will be
successful, and in the high-level config (ie `debug`) `ofline` it will
just simply not be used, since it has a typo. Otherwise, if there is no
default for `offline` then the validation will fail, with the following
error:

.. code:: python

   2025-01-28 09:37:23 INFO Validating configs.
   2025-01-28 09:37:23 INFO Prepending Anemoi Home (/home_path/.config/anemoi/training/config) to the search path.
   2025-01-28 09:37:23 INFO Prepending current user directory (/repos_path/config_anemoi_core) to the search path.
   2025-01-28 09:37:23 INFO Search path is now:  [provider=anemoi-cwd-searchpath-plugin, path=/repos_path/config_anemoi_core, provider=anemoi-home-searchpath-plugin, path=/home_path/.config/anemoi/training/config, provider=hydra, path=pkg://hydra.conf, provider=main, path=/repos_path/anemoi-core/training/src/anemoi/training/commands]
   pydantic_core._pydantic_core.ValidationError: 1 validation error for BaseSchema
   diagnostics.log.mlflow.offline
   Field required [type=missing, input_value={'enabled': True, 'authen...onfig'], 'ofline': True}, input_type=DictConfig]
      For further information visit https://errors.pydantic.dev/2.10/v/missing
   2025-01-28 10:14:49 ERROR
   💣 1 validation error for BaseSchema
   diagnostics.log.mlflow.offline
   Field required [type=missing, input_value={'enabled': True, 'authen...onfig'], 'ofline': True}, input_type=DictConfig]
      For further information visit https://errors.pydantic.dev/2.10/v/missing
   2025-01-28 10:14:49 ERROR 💣 Exiting

That will indicate that the `offline` field is required and it is
missing from the configuration file. If you identify any issues with the
schemas or missing functionality, please raise an issue on the `Anemoi
Core repository`.

Another type of error that we can see when working with Pydantic, is
when we have a union of schemas, and then we try to validate using on
those schemas config. For information about Unions, please refer to the
`Pydantic documentation
<https://docs.pydantic.dev/latest/concepts/unions/>`_. For example,
let's say we have a config with a union of schemas like the following:

.. code:: yaml

   defaults:
   - data: zarr
   - dataloader: native_grid
   - diagnostics: evaluation
   - hardware: example
   - graph: multi_scale
   - model: transformer # Change from default group
   - training: default
   - _self_


   graphs:
      attributes:
         nodes:
             area_weight:
               _target_: anemoi.graphs.nodes.attributes.SphericalAreaWeights # options: Area, Uniform
               norm: unit-max # options: l1, l2, unit-max, unit-sum, unit-std

In the example above, Pydantic will try to validate the
SphericalAreaWeights schema using the union NodeAttributeSchemas, which
contains a list of all the possible schemas for graph nodes attributes.

.. code:: python

   NodeAttributeSchemas = Union[
      PlanarAreaWeightSchema
      | SphericalAreaWeightSchema
      | CutOutMaskSchema
      | NonmissingAnemoiDatasetVariableSchema
      | BooleanOperationSchema
   ]

If the schema is not correctly defined, then the validation will fail,
with the following error:

.. code:: python

   2025-01-28 09:37:23 INFO Validating configs.
   2025-01-28 09:37:23 INFO Prepending Anemoi Home (/home_path/.config/anemoi/training/config) to the search path.
   2025-01-28 09:37:23 INFO Prepending current user directory (/repos_path/config_anemoi_core) to the search path.
   2025-01-28 09:37:23 INFO Search path is now:  [provider=anemoi-cwd-searchpath-plugin, path=/repos_path/config_anemoi_core, provider=anemoi-home-searchpath-plugin, path=/home_path/.config/anemoi/training/config, provider=hydra, path=pkg://hydra.conf, provider=main, path=/repos_path/anemoi-core/training/src/anemoi/training/commands]
   pydantic_core._pydantic_core.ValidationError: 1 validation error for BaseSchema
   2025-01-28 10:14:49 ERROR
   💣 14 validation error for BaseSchema
   graph.nodes.data.attributes.area_weight.PlanarAreaWeightSchema._target_
   Input should be 'anemoi.graphs.nodes.attributes.AreaWeights', 'anemoi.graphs.nodes.attributes.PlanarAreaWeights', 'anemoi.graphs.nodes.attributes.CutOutMask' or 'anemoi.graphs.nodes.attributes.UniformWeights' [type=literal_error, input_value='anemoi.graphs.nodes.attr...es.SphericalAreaWeights', input_type=str]
      For further information visit https://errors.pydantic.dev/2.10/v/literal_error
   graph.nodes.data.attributes.area_weight.function-after[convert_centre_to_ndarray(), SphericalAreaWeightSchema].fill_value
   Field required [type=missing, input_value={'_target_': 'anemoi.grap...ts', 'norm': 'unit-max'}, input_type=DictConfig]
      For further information visit https://errors.pydantic.dev/2.10/v/missing
   graph.nodes.data.attributes.area_weight.CutOutMaskSchema._target_
   Input should be 'anemoi.graphs.nodes.attributes.CutOutMask' [type=literal_error, input_value='anemoi.graphs.nodes.attr...es.SphericalAreaWeights', input_type=str]
      For further information visit https://errors.pydantic.dev/2.10/v/literal_error
   graph.nodes.data.attributes.area_weight.NonmissingAnemoiDatasetVariableSchema._target_
   Input should be 'anemoi.graphs.nodes.attributes.NonmissingAnemoiDatasetVariable' [type=literal_error, input_value='anemoi.graphs.nodes.attr...es.SphericalAreaWeights', input_type=str]
      For further information visit https://errors.pydantic.dev/2.10/v/literal_error
   graph.nodes.data.attributes.area_weight.NonmissingAnemoiDatasetVariableSchema.variable
   Field required [type=missing, input_value={'_target_': 'anemoi.grap...ts', 'norm': 'unit-max'}, input_type=DictConfig]
      For further information visit https://errors.pydantic.dev/2.10/v/missing
   graph.nodes.data.attributes.area_weight.BooleanOperationSchema._target_
   Input should be 'anemoi.graphs.nodes.attributes.BooleanNot', 'anemoi.graphs.nodes.attributes.BooleanAndMask' or 'anemoi.graphs.nodes.attributes.BooleanOrMask' [type=literal_error, input_value='anemoi.graphs.nodes.attr...es.SphericalAreaWeights', input_type=str]
      For further information visit https://errors.pydantic.dev/2.10/v/literal_error
   graph.nodes.hidden.attributes.area_weight.PlanarAreaWeightSchema._target_
   Input should be 'anemoi.graphs.nodes.attributes.AreaWeights', 'anemoi.graphs.nodes.attributes.PlanarAreaWeights', 'anemoi.graphs.nodes.attributes.CutOutMask' or 'anemoi.graphs.nodes.attributes.UniformWeights' [type=literal_error, input_value='anemoi.graphs.nodes.attr...es.SphericalAreaWeights', input_type=str]
      For further information visit https://errors.pydantic.dev/2.10/v/literal_error
   graph.nodes.hidden.attributes.area_weight.function-after[convert_centre_to_ndarray(), SphericalAreaWeightSchema].fill_value
   Field required [type=missing, input_value={'_target_': 'anemoi.grap...ts', 'norm': 'unit-max'}, input_type=DictConfig]
      For further information visit https://errors.pydantic.dev/2.10/v/missing
   graph.nodes.hidden.attributes.area_weight.CutOutMaskSchema._target_
   Input should be 'anemoi.graphs.nodes.attributes.CutOutMask' [type=literal_error, input_value='anemoi.graphs.nodes.attr...es.SphericalAreaWeights', input_type=str]
      For further information visit https://errors.pydantic.dev/2.10/v/literal_error
   graph.nodes.hidden.attributes.area_weight.NonmissingAnemoiDatasetVariableSchema._target_
   Input should be 'anemoi.graphs.nodes.attributes.NonmissingAnemoiDatasetVariable' [type=literal_error, input_value='anemoi.graphs.nodes.attr...es.SphericalAreaWeights', input_type=str]
      For further information visit https://errors.pydantic.dev/2.10/v/literal_error
   graph.nodes.hidden.attributes.area_weight.NonmissingAnemoiDatasetVariableSchema.variable
   Field required [type=missing, input_value={'_target_': 'anemoi.grap...ts', 'norm': 'unit-max'}, input_type=DictConfig]
      For further information visit https://errors.pydantic.dev/2.10/v/missing
   graph.nodes.hidden.attributes.area_weight.BooleanOperationSchema._target_
   Input should be 'anemoi.graphs.nodes.attributes.BooleanNot', 'anemoi.graphs.nodes.attributes.BooleanAndMask' or 'anemoi.graphs.nodes.attributes.BooleanOrMask' [type=literal_error, input_value='anemoi.graphs.nodes.attr...es.SphericalAreaWeights', input_type=str]
      For further information visit https://errors.pydantic.dev/2.10/v/literal_error
   training.scale_validation_metrics
   Extra inputs are not permitted [type=extra_forbidden, input_value={'scalars_to_apply': ['va...e'], 'metrics': ['all']}, input_type=DictConfig]
      For further information visit https://errors.pydantic.dev/2.10/v/extra_forbidden
   2025-02-07 16:13:33 ERROR 💣 Exiting

What's happening here, is that Pydantic can't match the config schema
with the defined SphericalAreaWeightSchema (since it's missing the entry
`fill_value: 0`. ) and it then tries to see if any of the other schemas
in the union match our config, going from left to right and throwing an
error for each of the schemas in the union. We understand the current
error reported is not very intuitive and indeed hides the real issue. We
will work on improving this on future releases, but mean time we
recommend to double check the schemas and the config files to make sure
they are correctly defined.

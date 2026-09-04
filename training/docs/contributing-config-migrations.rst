.. _contributing-config-migrations:

##############################
 Contributing Config Migration
##############################

If your PR changes the configuration structure, you will need to provide a configuration
migration.

*********************
 Creating a migration
*********************

Run:

.. code:: bash

    anemoi-training config migration create NAME [--final]

This command will create a new migration script that will look like:

.. code:: bash

    from anemoi.utils.migrations import MigrationMetadata

    from anemoi.training.migrations.config import Config

    # DO NOT CHANGE -->
    metadata = MigrationMetadata(
        versions={
            "migration": "1.0.0",
            "anemoi-training": "%NEXT_ANEMOI_TRAINING_VERSION%",
        },
    )
    # <-- END DO NOT CHANGE


    def migrate(config: Config) -> Config:
        """Migrate the config.

        Parameters
        ----------
        config : Config
            The config object to migrated.

        Returns
        -------
        Config
            The migrated config.
        """
        return config

You will need to edit the ``migrate`` function. You can assume that the config object
that you get as input is a valid configuration prior to the changes in your PR. The
output config should be updated to reflect the changes in your PR.

See below for details abaut the config object.

******************
 The Config Object
******************

When creating a migration, you will have access to an abstraction of the config that
allows high-level changes of the config. This ``Config`` class encapsulate `OmegaConf`_
and `yamlrocks`_. OmegaConf allows manipulating interpolations in the config and yamlrocks
is a yaml parsing library that allows round-tripping editing (the exported output
will correspond to the original content).

.. _OmegaConf: https://omegaconf.readthedocs.io
.. _yamlrocks: https://yaml.rocks/

Getting a value
---------------

.. code:: python

    from anemoi.training.migrations.config import Config
    config = Config.from_path("/path/to/yaml_file.yaml")
    config["data"]["frequency"] # This is a Node object
    config["data"]["frequency"].value # This is the actual (uninterpolated) value
    config["data"]["frequency"].resolved_value # This is the omegaconf resolved value

You can also use the ``select`` method when selecting from a dot-delimited string:

.. code:: python

    # with a dot-delimited string:
    config.select("data.frequency", create_missing=True)
    # or
    config.select(("data", "frequency"), create_missing=True)

``create_missing=True`` will create missing keys as dicts if they don't exist in the tree.
Defaults to ``False``.

Adding a new key
----------------

.. code:: python

    from anemoi.training.migrations.config import Config
    config = Config.from_path("/path/to/yaml_file.yaml")
    config.add_key("data.datasets.foo.bar", "new value")

This will add the whole tree, including missing intermediary nodes:

.. code:: yaml
    data:
      datasets:
        foo:
          bar: new value

Dropping a key
--------------

.. code:: python

    from anemoi.training.migrations.config import Config
    config = Config.from_path("/path/to/yaml_file.yaml")
    config.drop_key("data.datasets.foo.bar", remove_empty=True)

``remove_empty=True`` will also remove the key "foo" because it will be empty after
dropping the "bar" key. Defaults to False.


Renaming/Moving a key
---------------------

.. code:: python

    from anemoi.training.migrations.config import Config
    config = Config.from_path("/path/to/yaml_file.yaml")
    config.rename_key("data.datasets.foo.bar", remove_empty=True)


``remove_empty=True`` will also remove the key "foo" because it will be empty after
moving the "bar" key.

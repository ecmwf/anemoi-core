##################
 Config Migrations
##################

Config migrations allow users to update the structure of their anemoi-training configuration
across version changes. This tool will help users migrate their configuration files so
they can be used in newer anemoi-training versions.

Migrations are only compatible with configuration dumps.

Migrating your configuration dump may change the structure of the configuration, but it
may not change any values. As a result, you cannot use this tool to update default values.
However, migrations can add, remove or move keys in the config.


------
 Usage
------

To update your config dump:

.. code:: bash

    anemoi-training config migration sync PATH [--output OUTPUT] [--no-color]

where ``PATH`` is the path to your config file. You can define where the migrated config
will be written to with ``--output path/to/migrated-config.yaml``. Defaults to printing
in stdout.

This command will also write a diff of the old versus new config to help you quickly see
what has been changed by the migrations.

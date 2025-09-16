.. _migrations:

##################
 Migration System
##################

This serves as general information on how the migration system works.
This can be useful for contributors who need to write a migration
script, users who want to understand how their checkpoint are updated,
or futur contributors to the migration code

The migration system's goal is to allow users to keep a checkpoint
trained on a version of anemoi-models, and use it on newer or older
version, even if it would have lead to a break of the checkpoint.

This is not only convenient for user to avoid having to retrain a full
model just because a layer has been renamed, but also it allowes more
flexibility to contributors for changes that they would not have done
lest breaking existing checkpoints.

******************
 General Overview
******************

Migrations are stored in anemoi-models in ``migrations.scripts`` as an
ordered list of scripts. Each script contains:

-  some metadata information, such as the version of the migration
   system, or the version of anemoi-models,
-  a ``migrate`` function to migrate checkpoints
-  optionnally a ``rollback`` function to reverse the migration function
-  optionnally a ``migrate_setup`` function to fix import issues.

Similarly, the checkpoint contain some migration information that
informs on its migration state:

-  the ``name`` of the migration: corresponds to the filename of the
   script in anemoi-models,

-  the ``metadata``: same as in the migration scripts,

-  the ``signature``: a hash digest of the original migration script.
   This is used to detect whether already executed scripts have changed.
   For now, it only logs a warning, but a more complex behavior could be
   added in the future,

-  the ``rollback`` function: the same as in the script, in case you
   need the checkpoint in an older version,

**********************
 Compatibility groups
**********************

Some changes cannot be migrated. For example, a change in architecture
that adds some trainable weights. When this happens, a "final" migration
script need to be created. The "final" migrations act as separators to
show migrations that are compatible with one another. For example, let's
look at this list of migration in anemoi-models:

+-------------------+-----------+-----------+---------------+-----------+--------------+-----------+
| Name              | migration | migration | final         | migration | final        | migration |
|                   | 1         | 2         | migration     | 3         | migraion     | 4         |
+===================+===========+===========+===============+===========+==============+===========+
| Version           | 0.8.1     | 0.8.3     | 0.9.0         | 0.10.5    | 0.12.0       | 0.12.2    |
+-------------------+-----------+-----------+---------------+-----------+--------------+-----------+
| Compatibility     | 1         | 1         | 2             | 2         | 3            | 3         |
| group             |           |           |               |           |              |           |
+-------------------+-----------+-----------+---------------+-----------+--------------+-----------+

This also shows the ``compatibility groups`` that groups migrations that
are compatible with one-another.

For example, for a checkpoint trained on version 0.8.1, ``migration 1``
is already registered in the checkpoint. This checkpoint can be migrated
to be used with all versions of its compatibility group (group 1) up
until (and excluding) 0.9.0.

Similarly, a checkpoint trained on version 0.12.2 can be downgraded up
until (and including) 0.12.0.

.. note::

   Checkpoints only store migration information of their own
   compatibility group. The "final" migration of a group can also be
   seen as the first migration of the following group. In fact, "final"
   migrations are always the first registered migration of a group, and
   acts as a marker of the compatibility group of the checkpoint. The
   first compatibility group is an exception, and does not start with a
   "final" migration.

**********************
 Resolution algorithm
**********************

The operations to execute are decided by the following resolution
algorithm. To follow along, here is an example:

+----------------+-----------------+
| In             | In the          |
| anemoi-models  | checkpoint      |
+================+=================+
| migration 1    | migration 1     |
+----------------+-----------------+
| migration 2    | migration 2     |
+----------------+-----------------+
| migration 5    | migration 3     |
+----------------+-----------------+
| migration 6    | migration 4     |
+----------------+-----------------+
| migration 7    |                 |
+----------------+-----------------+

-  First, we rollback any additional migrations in the checkpoint,
   starting from the end (here, migration 4 and 3 need to be
   rollbacked).

-  Then, we migrate any missing migrations in the checkpoint, starting
   from the start (here migration 5, 6 and 7).

In the example, it will produce:

-  ROLLBACK migration 4
-  ROLLBACK migration 3
-  MIGRATE migration 5
-  MIGRATE migration 6
-  MIGRATE migration 7

*********************
 Executed migrations
*********************

The whole history of migrations is stored in the metadata of the
checkpoint. It can be accessed through:

.. code::

   >>> history = metadata.get("migrations", {}).get("history", [])
   >>> for executed_migration in history:
   ...    print(executed_migration)
   { "type": "rollback", "name": "migration_name1.py", "signature": "[...]" }
   { "type": "migrate", "name": "migration_name2.py", "signature": "[...]" }

**********
 Migrator
**********

.. automodule:: anemoi.models.migrations.migrator
   :members:
   :no-undoc-members:
   :show-inheritance:

***************
 Setup Context
***************

.. automodule:: anemoi.models.migrations.setup_context
   :members:
   :no-undoc-members:
   :show-inheritance:

.. _create-migrations:

##############################
 Create checkpoint migrations
##############################

.. note::

   For more technical details about migrations, see :ref:`migrations`.

*********
 Migrate
*********

To create a new migration, run:

.. code:: bash

   anemoi-models migration create MIGRATION_NAME

This will create a new migration script at the provided location that
looks like:

.. code:: python

   from anemoi.models.migrations import CkptType
   from anemoi.models.migrations import MigrationMetadata

   metadata = MigrationMetadata(
       versions={
           "migration": "1.0.0",
           "anemoi-models": "0.8.1",
       }
   )


   def migrate(ckpt: CkptType) -> CkptType:
       """
       Migrate the checkpoint.


       Parameters
       ----------
       ckpt : CkptType
           The checkpoint dict.

       Returns
       -------
       CkptType
           The migrated checkpoint dict.
       """
       return ckpt


   def rollback(ckpt: CkptType) -> CkptType:
       """
       Rollback the checkpoint.


       Parameters
       ----------
       ckpt : CkptType
           The checkpoint dict.

       Returns
       -------
       CkptType
           The rollbacked checkpoint dict.
       """

       return ckpt

``migrate`` receives an old checkpoint (made before your changes), and
must return a checkpoint compatible with your changes.

.. note::

   The metadata object is automatically generated. You should not change
   this part of the script.

   In particular, it contains the version of the migration system. This
   is to allow future changes in the API but still support older
   migration scripts.

Migrations are only done for training checkpoints. Users are expected to
re-generate the inference checkpoint once the training checkpoint is
migrated.

If you migration is only related to a specific architecture, you should
add a guard in the migration script. For example, related to a specific
processor class:

.. code:: python

   def migrate(ckpt: CkptType) -> CkptType:
      if ckpt["hyper_parameters"]["config"].model.processor._target_ == "anemoi.models.layers.processor.TransformerProcessor":
          # Do stuff
          ...
      return ckpt

Migration names have a timestamp at the start to specify their order of
execution. The timestamp is decided when creating the migration script.
However, it may happen that a new commit in main contains a migration
script with a later timestamp than one or several of your migration
scripts, which would the correct order.

The unit test ``test_migration_order`` will check whether the correct
order is preserved. If you get an error, you can run ``anemoi-models
migration fix-order`` to update the timestamps of your scripts.

**********
 Rollback
**********

``rollback`` does the opposite operation and receives a checkpoint
compatible with your changes and must return a checkpoint usable before
your change.

.. note::

   We use `cloudpickle <https://github.com/cloudpipe/cloudpickle>`_ to
   pickle the rollback function by value rather than by inference. In
   particular, you should follow the recommandations described `here
   <https://github.com/cloudpipe/cloudpickle/tree/master?tab=readme-ov-file#overriding-pickles-serialization-mechanism-for-importable-constructs>`_.

.. note::

   Rollback functions are not strictly required. However checkpoints
   will not be able to be rollbacked before your migration script if it
   does not have a rollback.

To generate a migration script without a rollback use the
``--no-rollback`` parameter:

.. code:: bash

   anemoi-models migration create migration-name --no-rollback

****************
 Simple example
****************

For example, if you renamed a layer x to y, you can make the following
migration:

.. code:: python

   from anemoi.models.migrations import CkptType
   from anemoi.models.migrations import MigrationMetadata

   metadata = MigrationMetadata(
       versions={
           "migration": "1.0.0",
           "anemoi-models": "0.8.1",
       }
   )


   def migrate(ckpt: CkptType) -> CkptType:
       """
       Migrate the checkpoint.


       Parameters
       ----------
       ckpt : CkptType
           The checkpoint dict.

       Returns
       -------
       CkptType
           The migrated checkpoint dict.
       """
       ckpt["state_dict"]["y"] = ckpt["state_dict"].pop("x")
       return ckpt


   def rollback(ckpt: CkptType) -> CkptType:
       """
       Rollback the checkpoint.


       Parameters
       ----------
       ckpt : CkptType
           The checkpoint dict.

       Returns
       -------
       CkptType
           The rollbacked checkpoint dict.
       """
       ckpt["state_dict"]["x"] = ckpt["state_dict"].pop("y")
       return ckpt

****************
 Setup callback
****************

Python objects are stored by reference in a pickle object. This means
that if you move (or remove) a class, old checkpoints cannot be loaded.

.. note::

   Migration scripts use a special Unpickler that obfuscate these import
   errors to access the migration information in the checkpoint.

The setup callbacks are functions that fix import errors. They are run
before loading the checkpoint. To add a setup callback to your script,
define the ``migrate_setup`` callback:

.. code:: python

   from anemoi.models.migrations import MigrationContext


   def migrate_setup(context: MigrationContext) -> None:
       """
       Migrate setup callback to be run before loading the checkpoint.

       Parameters
       ----------
       context : MigrationContext
          A MigrationContext instance
       """

.. note::

   The setup is only defined for migrate. The setup required for the
   rollback pass is automatically inferred.

To generate your script with the setup callbacks, use the
``--with-setup`` argument:

.. code:: bash

   anemoi-models migration create migration-name --with-setup

The context object provides three methods to fix import errors:

-  ``context.move_attribute(start_path, end_path)`` to indicate that an
   attribute was moved from ``start_path`` to ``end_path``.

-  ``context.move_module(start_path, end_path)`` to indicate that a
   module was moved from ``start_path`` to ``end_path``.

-  ``context.delete_attribute(path)`` to indicate that an attribute was
   removed. You can use the wildcard "*" to delete any attribute in the
   module.

For example, if you renamed the module
``anemoi.models.schemas.data_processor`` to
``anemoi.models.schemas.data``, your migration might look like:

.. code:: python

   from anemoi.models.migrations import CkptType
   from anemoi.models.migrations import MigrationContext
   from anemoi.models.migrations import MigrationMetadata

   metadata = MigrationMetadata(
       versions={
           "migration": "1.0.0",
           "anemoi-models": "0.8.1",
       }
   )


   def migrate_setup(context: MigrationContext) -> None:
       """
       Migrate setup callback to be run before loading the checkpoint.

       Parameters
       ----------
       context : MigrationContext
          A MigrationContext instance
       """
       context.move_module("anemoi.models.schemas.data_processor", "anemoi.models.schemas.data")


   def migrate(ckpt: CkptType) -> CkptType:
       """
       Migrate the checkpoint.


       Parameters
       ----------
       ckpt : CkptType
           The checkpoint dict.

       Returns
       -------
       CkptType
           The migrated checkpoint dict.
       """
       # This is also executed. You can update the checkpoint if you need to.
       return ckpt


   def rollback(ckpt: CkptType) -> CkptType:
       """
       Rollback the checkpoint.


       Parameters
       ----------
       ckpt : CkptType
           The checkpoint dict.

       Returns
       -------
       CkptType
           The rollbacked checkpoint dict.
       """
       return ckpt

Similarly, if you moved the class ``NormalizerSchema`` from
``anemoi.training.schemas.data`` to
``anemoi.models.schemas.data_processor``, the setup callback might look
like:

.. code:: python

   def migrate_setup(context: MigrationContext) -> None:
       """
       Migrate setup callback to be run before loading the checkpoint.

       Parameters
       ----------
       context : MigrationContext
          A MigrationContext instance
       """
       context.move_attribute(
           "anemoi.training.schemas.data.NormalizerSchema", "anemoi.models.schemas.data_processor.NormalizerSchema"
       )

.. note::

   The attribute can also have a different name in the final location.

******************
 Final migrations
******************

If the modifications are too complex, and it is decided that migrating
old checkpoint should not be supported, you can create a "final"
migration with:

.. code:: bash

   anemoi-models migration create --final MIGRATION_NAME

**************
 Full example
**************

Here is a full example of a migration to fix `PR 433
<https://github.com/ecmwf/anemoi-core/pull/433>`_

.. code:: python

   from anemoi.models.migrations import CkptType
   from anemoi.models.migrations import MigrationContext
   from anemoi.models.migrations import MigrationMetadata

   metadata = MigrationMetadata(
       versions={
           "migration": "1.0.0",
           "anemoi-models": "0.9.0",
       }
   )


   def migrate_setup(context: MigrationContext) -> None:
       """
       Migrate setup callback to be run before loading the checkpoint.

       Parameters
       ----------
       context : MigrationContext
          A MigrationContext instance
       """
       context.move_attribute(
           "anemoi.training.schemas.data.NormalizerSchema", "anemoi.models.schemas.data_processor.NormalizerSchema"
       )


   def migrate(ckpt: CkptType) -> CkptType:
       """
       Migrate the checkpoint.


       Parameters
       ----------
       ckpt : CkptType
           The checkpoint dict.

       Returns
       -------
       CkptType
           The migrated checkpoint dict.
       """
       return ckpt


   def rollback_setup(context: MigrationContext) -> None:
       """
       Rollback setup callback to be run before loading the checkpoint.

       Parameters
       ----------
       context : MigrationContext
          A MigrationContext instance
       """
       context.move_attribute(
           "anemoi.models.schemas.data_processor.NormalizerSchema", "anemoi.training.schemas.data.NormalizerSchema"
       )


   def rollback(ckpt: CkptType) -> CkptType:
       """
       Rollback the checkpoint.


       Parameters
       ----------
       ckpt : CkptType
           The checkpoint dict.

       Returns
       -------
       CkptType
           The rollbacked checkpoint dict.
       """
       return ckpt

****************
 Best practices
****************

Here are best practices that will help you create good migration
scripts.

-  Use a `if` guard to only apply scripts to specific architecture.

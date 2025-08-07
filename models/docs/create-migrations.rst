.. _create-migrations:

##############################
 Create checkpoint migrations
##############################

********
 Migrate
********

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
       """Migrate the checkpoint"""
       return ckpt


   def rollback(ckpt: CkptType) -> CkptType:
       """Rollbacks the migration"""
       return ckpt

``migrate`` receives an old checkpoint (made before your changes), and
must return a checkpoint compatible with your changes.

*********
 Rollback
*********

``rollback`` does the opposite operation and receives a checkpoint
compatible with your changes and must return a checkpoint usable before
your change.

.. note::

   We use `cloudpickle <https://github.com/cloudpipe/cloudpickle>`_ to
   pickle the rollback function by value rather than by inference. In
   particular, you should follow the recommandations described `here
   <https://github.com/cloudpipe/cloudpickle/tree/master?tab=readme-ov-file#overriding-pickles-serialization-mechanism-for-importable-constructs>`_.

Rollback are not strictly required for the migration script to function. Note however
that if the migration script does not have a rollback, checkpoints will not be able to
be rollbacked before your migration script.

To generate a migration script without a rollback use the ``--no-rollback`` parameter:

.. code:: bash
    anemoi-models migration create migration-name --no-rollback

***************
 Simple example
***************

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
       """Migrate the checkpoint"""
       ckpt["state_dict"]["y"] = ckpt["state_dict"].pop("x")
       return ckpt


   def rollback(ckpt: CkptType) -> CkptType:
       """Rollbacks the migration"""
       ckpt["state_dict"]["x"] = ckpt["state_dict"].pop("y")
       return ckpt


****************
 Setup callbacks
****************

Python objects are store by reference in a pickle object. This means that if your
changes moves to another module an object stored in checkpoints, it will break the checkpoint
to the point that it cannot be loaded.

.. note::

    Migration scripts use a special Unpickler that obfuscate these import errors to access
    the store migration information.

The setup callbacks is are functions in your migration script that are run before loading the
checkpoint to fix import errors:

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

    def migrate_rollback(context: MigrationContext) -> None:
        """
        Migrate setup callback to be run before loading the checkpoint.

        Parameters
        ----------
        context : MigrationContext
           A MigrationContext instance
        """

To generate your script with the setup callbacks, use the ``--with-setup`` argument:

.. code:: bash

   anemoi-models migration create migration-name --with-setup 

The context object provides two functions that can be used to simplify fixing attributes
that are moved:

* ``context.move_module(start_path, end_path)`` to indicate that this script moved a module
  from ``start_path`` to ``end_path``.
* ``context.move_attribute(start_path, end_path)`` to indicate that an attribute was moved
  from ``start_path`` to ``end_path``.

For example, if you rename the module ``anemoi.models.schemas.data_processor`` to ``anemoi.models.schemas.data``,
your migration might look like:

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
       """Migrate the checkpoint"""
       return ckpt

   def migrate_rollback(context: MigrationContext) -> None:
       """
       Migrate setup callback to be run before loading the checkpoint.

       Parameters
       ----------
       context : MigrationContext
          A MigrationContext instance
       """
       context.move_module("anemoi.models.schemas.data", "anemoi.models.schemas.data_processor")

   def rollback(ckpt: CkptType) -> CkptType:
       """Rollbacks the migration"""
       return ckpt

Similarly, if you moved the class ``NormalizerSchema`` from ``anemoi.training.schemas.data`` to
``anemoi.models.schemas.data_processor``, the setup callback might look like:

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
           "anemoi.training.schemas.data.NormalizerSchema", "anemoi.models.schemas.data_processor.DataSchema"
       )

.. note::

   The attribute can also have a different name in the final location.



*****************
 Final migrations
*****************

If the modifications are too complex, and we decide that we don't
support migrating old checkpoints past this change, you can create a
"final" migration with:

.. code:: bash

   anemoi-models migration create --final MIGRATION_NAME

*************
 Full example
*************

Here is a full example of what a migration could have looked like for `PR 433 <https://github.com/ecmwf/anemoi-core/pull/433>`_

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
        Setup function ran before loading the checkpoint. This can be used to move objects
        around.

        Parameters
        ----------
        context : MigrationContext
           A context object with some utilities
        """
        context.move_attribute("anemoi.models.schemas.data_processor.DataSchema", "anemoi.training.schemas.data.DataSchema")
        context.move_attribute(
            "anemoi.training.schemas.data.NormalizerSchema", "anemoi.models.schemas.data_processor.DataSchema"
        )


    def migrate(ckpt: CkptType) -> CkptType:
        """
        
        
        Parameters
        ----------
        ckpt : CkptType
            
        
        Returns
        -------
        CkptType
            
        """
        """Migrate the checkpoint"""
        return ckpt


    def rollback_setup(context: MigrationContext) -> None:
        """
        Setup function ran before loading the checkpoint. This can be used to move objects
        around.

        Parameters
        ----------
        context : MigrationContext
           A context object with some utilities
        """
        context.move_attribute("anemoi.training.schemas.data.DataSchema", "anemoi.models.schemas.data_processor.DataSchema")
        context.move_attribute(
            "anemoi.models.schemas.data_processor.DataSchema", "anemoi.training.schemas.data.NormalizerSchema"
        )


    def rollback(ckpt: CkptType) -> CkptType:
        """Rollbacks the migration"""
        return ckpt

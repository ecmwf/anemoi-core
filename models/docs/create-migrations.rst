.. _create-migrations:

##############################
 Create checkpoint migrations
##############################

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

``rollback`` does cthe opposite operation and receives a checkpoint
compatible with your changes and must return a checkpoint usable before
your change.

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

If the modifications are too complex, and we decide that we don't
support migrating old checkpoints past this change, you can create a
"final" migration with:

.. code:: bash

   anemoi-models migration create --final MIGRATION_NAME

.. _cli-migration:

##########################
 Migrating your checkpoint
##########################

Anemoi-models provides a way for users to migrate old checkpoints so that they can be
used with recent versions of anemoi-models.

If you want to use your checkpoint with the currently installed version of anemoi-models,
you can use:

.. code:: bash

   anemoi-models migration sync PATH_TO_CKPT


This will update (if possible) your checkpoint so that it is compatible with the current version
of anemoi-models. If your checkpoint is too old and migrating is not supported, you will get a
``IncompatibleCheckpointException``.

Your old checkpoint is still available with the name ``OLD_NAME-v{version}.ckpt``.


********************************
 Migrating to a specific version
********************************
Update your anemoi-models to the desired version and call ``anemoi-models migration sync``.

Note that this should work when updating to a newer version, as well as downgrading to an older
version, as long as your checkpoint is not too old.

*********
 Rollback
*********
If you update to an older version, the checkpoint will be rollbacked to be compatible with this
older version.

*******************
 Manually migrating
*******************
You can decide to manually migrate a certain number of steps with

.. code:: bash

   anemoi-models migration sync PATH_TO_CKPT --steps STEPS

``STEPS`` can be both positive (migrating) or negative (rollback).

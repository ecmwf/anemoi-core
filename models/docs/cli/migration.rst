.. _other-migration:

##########################
 Migrating your checkpoint
##########################

anemoi-models provides a way for users to migrate old checkpoints so that they can be
used with recent versions of anemoi-models.

********
 Inspect
********

To inspect the registered versions in the checkpoint, run:

.. code:: bash

   anemoi-models migration inspect PATH_TO_CKPT

You can remove colors with the ``--no-color`` argument.


********
 Migrate
********

If you want to use your checkpoint with the currently installed version of anemoi-models,
you can use:

.. code:: bash

   anemoi-models migration sync PATH_TO_CKPT


This will update (if possible) your checkpoint so that it is compatible with the current version
of anemoi-models. If your checkpoint is too old and migrating is not supported, you will get a
``IncompatibleCheckpointException``.

Your old checkpoint is still available with the name ``OLD_NAME-v{version}.ckpt``.

You can remove colors with the ``--no-color`` argument.

********
 Dry-run
********

You can check which migration will be executed with a dry-run, without updating your checkpoint:

.. code:: bash

   anemoi-models migration sync --dry-run PATH_TO_CKPT


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

####################
 General guidelines
####################

Thank you for your interest in Anemoi Training! Please follow the
:ref:`general Anemoi contributing guidelines
<anemoi-docs:contributing>`.

These include general guidelines for contributions to Anemoi,
instructions on setting up a development environment, and guidelines on
collaboration on GitHub, writing documentation, testing, and code style.

************
 Unit tests
************

anemoi-models include unit tests that can be executed locally using
pytest. For more information on testing, please refer to the
:ref:`general Anemoi testing guidelines
<anemoi-docs:testing-guidelines>`.

*******************************
 Provide checkpoint Migrations
*******************************

If your changes break existing checkpoints, you must provide a
checkpoint migration that will migrate old checkpoint so that they are
still usable with the newer version.

There is actually a test in CI to check whether your change breaks
existing checkpoints. It tries to restart training from an existing
checkpoint. This test applies all migrations to the checkpoint before
training, so providing a valid migration will fix the CI pipeline.

See :ref:`create checkpoint migrations <create-migrations>` for
information.

Migration names have a timestamp at the start to specify their order of
execution. The timestamp is decided when creating the migration script.
However, it may happen that a new commit in main contains a migration
script with a later timestamp than one or several of your migration
scripts, which would the correct order.

The unit test ``test_migration_order`` will check whether the correct
order is preserved. If you get an error, you can run ``anemoi-models
migration fix-order`` to update the timestamps of your scripts.

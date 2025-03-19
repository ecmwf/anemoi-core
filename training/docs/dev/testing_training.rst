**********************************************
 Integration tests and member state use cases
**********************************************

Integration tests in anemoi-training include both general integration
tests and tests for member state use cases.

Running tests
=============

To run integration tests in anemoi-training, ensure that you have GPU 
available, then from the top-level directory of anemoi-core run:

.. code:: bash

   pytest training/tests/integration --longtests



Configuration handling in integration tests
===========================================

Configuration management is essential to ensure that integration tests
remain reliable and maintainable. Our approach includes:

1. Using Configuration Templates: Always start with a configuration
template from the repository to minimize redundancy and ensure
consistency. We expect the templates to be consistent with the code base
and have integration tests that check for this consistency.

2. Test-specific Modifications: Apply only the necessary
use-case-specific (e.g. dataset) and testing-specific (e.g. batch_size)
modifications to the template. Use a config modification yaml, or hydra
overrides for parametrization of a small number of config values.

3. Reducing Compute Load: Where possible, reduce the number of batches,
epochs, and batch sizes.

4. Debugging and Failures: When integration tests fail, check the config
files in `training/src/anemoi/training/config` for inconsistencies with
the code and update the config files if necessary. Also check if
test-time modifications have introduced unintended changes.

Example of configuration handling
=================================

For an example, see `training/tests/integration/test_training_cycle.py`.
The test uses a configuration based on the template
`training/src/anemoi/training/config/basic.py`, i.e. the basic global
model. It applies testing-specific modifications to reduce batch_size
etc. as detailed in
`training/tests/integration/test_training_cycle.yaml`. It furthermore
applies use-case-specific modifications as detailed in
`training/tests/integration/test_basic.yaml` to provide the location of
our testing dataset compatible with the global model.

Note that we also parametrize the fixture `architecture_config` to
override the default model configuration in order to test different
model architectures.

Adding a member state use case test
===================================

To add a new member test use case, follow these steps:

1. Use an Integration Test Template: To ensure maintainability, we
recommend following the config handling guidelines detailed above in so
far as this makes sense for your use case.

2. Best practices: Follow best practices, such as reducing compute load
and managing configurations via configuration files.

3. Prepare the Data: Ensure the required dataset is uploaded to the EWC
S3 before adding the test. Please get in touch about access.

4. Subfolder Organization: Place your test and config files in a new
subfolder within `training/tests/integration/` for clarity and ease of
maintenance.

5. Handling Test Failures: Complex use cases will likely require more
test-time modifications. Check if these have overwritten expected
configurations or are out-of-date with configuration changes in the
templates.
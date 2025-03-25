#########
 Testing
#########

Comprehensive testing is crucial for maintaining the reliability and
stability of Anemoi Training. This guide outlines our testing strategy
and best practices for contributing tests.

*******************
 Testing Framework
*******************

We use pytest as our primary testing framework. Pytest offers a simple
and powerful way to write and run tests.

***************
 Writing Tests
***************

General Guidelines
==================

#. Write tests for all new features and bug fixes.
#. Aim for high test coverage, especially for critical components.
#. Keep tests simple, focused, and independent of each other.
#. Use descriptive names for test functions, following the pattern
   `test_<functionality>_<scenario>`.

Example Test Structure
======================

.. code:: python

   import pytest
   from anemoi.training import SomeFeature


   def test_some_feature_normal_input():
       feature = SomeFeature()
       result = feature.process(normal_input)
       assert result == expected_output


   def test_some_feature_edge_case():
       feature = SomeFeature()
       with pytest.raises(ValueError):
           feature.process(invalid_input)

Parametrized Tests
==================

Use pytest's parametrize decorator to run the same test with different
inputs:

.. code:: python

   @pytest.mark.parametrize(
       "input,expected",
       [
           (2, 4),
           (3, 9),
           (4, 16),
       ],
   )
   def test_square(input, expected):
       assert square(input) == expected

You can also consider ``hypothesis`` for property-based testing.

Fixtures
========

Use fixtures to set up common test data or objects:

.. code:: python

   @pytest.fixture
   def sample_dataset():
       # Create and return a sample dataset
       pass


   def test_data_loading(sample_dataset):
       # Use the sample_dataset fixture in your test
       pass

****************
 Types of Tests
****************

1. Unit Tests
=============

Test individual components in isolation. These should be the majority of
your tests.

2. Integration Tests
====================

Test how different components work together. These are particularly
important for data processing pipelines and model training workflows.

Integration tests in anemoi-training include both general integration
tests and tests for member state use cases.

3. Functional Tests
===================

Test entire features or workflows from start to finish. These ensure
that the system works as expected from a user's perspective.

***************
 Running Tests
***************

To run all unit tests:

.. code:: bash

   pytest

To run tests in a specific file:

.. code:: bash

   pytest tests/unit/test_specific_feature.py

To run tests with a specific mark:

.. code:: bash

   pytest -m slow

For integration tests, ensure that you have GPU available, then from the
top-level directory of anemoi-core run:

.. code:: bash

   pytest training/tests/integration --longtests

For long-running integration tests, we use the `--longtests` flag to
ensure that they are run only when necessary. This means that you should
add the correspondong marker to these tests:

.. code:: python

   @pytest.mark.longtests
   def test_long():
         pass

**********************************************
 Integration tests and member state use cases
**********************************************

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

***************
 Test Coverage
***************

We use pytest-cov to measure test coverage. To run tests with coverage:

.. code:: bash

   pytest --cov=anemoi_training

Aim for at least 80% coverage for new features, and strive to maintain
or improve overall project coverage.

************************
 Continuous Integration
************************

All tests are run automatically on our CI/CD pipeline for every pull
request. Ensure all tests pass before submitting your PR.

*********************
 Performance Testing
*********************

For performance-critical components:

#. Write benchmarks.
#. Compare performance before and after changes.
#. Set up performance regression tests in CI.

**********************
 Mocking and Patching
**********************

Use unittest.mock or pytest-mock for mocking external dependencies or
complex objects:

.. code:: python

   def test_api_call(mocker):
       mock_response = mocker.Mock()
       mock_response.json.return_value = {"data": "mocked"}
       mocker.patch("requests.get", return_value=mock_response)

       result = my_api_function()
       assert result == "mocked"

****************
 Best Practices
****************

#. Keep tests fast: Optimize slow tests or mark them for separate
   execution.
#. Use appropriate assertions: pytest provides a rich set of assertions.
#. Test edge cases and error conditions, not just the happy path.
#. Regularly review and update tests as the codebase evolves.
#. Document complex test setups or scenarios.

By following these guidelines and continuously improving our test suite,
we can ensure the reliability and maintainability of Anemoi Training.

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

****************
 Types of Tests
****************

1. Unit Tests
=============

-  Test individual components in isolation.
-  Should constitute the majority of test cases.
-  Unit tests reside in `tests/` or, for packages with integration
   tests, in `tests/unit`

2. Integration Tests
====================

-  Test how different components work together.
-  Important for data processing pipelines and model training workflows.
-  Integration tests reside in `tests/integration`.

3. Functional Tests
===================

-  Test entire features or workflows from start to finish.
-  Ensure that the system works as expected from a user's perspective.
-  This is a work in progress.

***************
 Running Tests
***************

Basic commands
==============

To run all unit tests:

.. code:: bash

   pytest

To run tests in a specific file:

.. code:: bash

   pytest tests/unit/test_specific_feature.py

To run tests with a specific mark:

.. code:: bash

   pytest -m slow

Running Integration Tests
=========================

To run all integration tests, including slow-running tests, use the
`--longtests` flag. Follow the package-specific instructions. For
integration tests in anemoi-training, for instance, ensure that you have
GPU available and run:

.. code:: bash

   pytest training/tests/integration/ --longtests

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

Mocking and Patching
====================

Use unittest.mock or pytest-mock for mocking external dependencies or
complex objects:

.. code:: python

   def test_api_call(mocker):
       mock_response = mocker.Mock()
       mock_response.json.return_value = {"data": "mocked"}
       mocker.patch("requests.get", return_value=mock_response)

       result = my_api_function()
       assert result == "mocked"

***************************
 Writing Integration Tests
***************************

Marking Long-Running Tests
==========================

For long-running integration tests, we use the `--longtests` flag to
ensure that they are run only when necessary. This means that you should
add the correspondong marker to these tests:

.. code:: python

   @pytest.mark.longtests
   def test_long():
         pass

Configuration Handling
======================

Integration tests in anemoi-training, anemoi-datasets, etc., rely on
appropriate handling of configuration files. Configuration management is
essential to ensure that the tests remain reliable and maintainable. Our
approach includes:

1. Using Configuration Templates: Always start with a configuration
template from the repository to minimize redundancy and ensure
consistency. We expect the templates to be consistent with the code base
and have integration tests that check for this consistency.

2. Test-specific Modifications: Apply only the necessary
use-case-specific (e.g. related to the dataset) and testing-specific
(e.g. batch_size or restricted date range) modifications to the
template.

3. Reducing Compute Load: Where possible, reduce the number of batches,
epochs, batch sizes, number of dates etc.

4. Debugging and Failures: When integration tests fail, check the config
files (e.g. in `training/src/anemoi/training/config`) for
inconsistencies with the code and update the config files if necessary.
Also check if test-time modifications have introduced unintended
changes.

For more details and package-specific examples, please refer to the
package-level documentation.

***************
 Test Coverage
***************

We use pytest-cov to measure test coverage. To check coverage:

.. code:: bash

   pytest --cov=anemoi_training

Aim for at least 80% coverage for new features, and strive to maintain
or improve overall project coverage.

************************
 Continuous Integration
************************

All unit tests are run automatically on our CI/CD pipeline for every
pull request after the initial review by maintainers. Ensure all tests
pass before submitting your PR.

*********************
 Performance Testing
*********************

For performance-critical components:

#. Write benchmarks.
#. Compare performance before and after changes.
#. Set up performance regression tests in CI.

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

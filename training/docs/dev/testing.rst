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

3. Functional Tests
===================

Test entire features or workflows from start to finish. These ensure
that the system works as expected from a user's perspective.

4. Parametrized Tests
=====================

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

5. Fixtures
===========

Use fixtures to set up common test data or objects:

.. code:: python

   @pytest.fixture
   def sample_dataset():
       # Create and return a sample dataset
       pass


   def test_data_loading(sample_dataset):
       # Use the sample_dataset fixture in your test
       pass

***************
 Running Tests
***************

To run all tests:

.. code:: bash

   pytest

To run tests in a specific file:

.. code:: bash

   pytest tests/test_specific_feature.py

To run tests with a specific mark:

.. code:: bash

   pytest -m slow

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

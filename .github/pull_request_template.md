## Description

<!-- Provide a brief summary of the changes introduced in this pull request. -->

## Type of Change

-   [ ] Bug fix (non-breaking change which fixes an issue)
-   [ ] New feature (non-breaking change which adds functionality)
-   [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
-   [ ] Documentation update

## Issue Number

<!-- Link the Issue number this change addresses, ideally in one of the "magic format" such as Closes #XYZ -->

<!-- Alternatively, explain the motivation behind the changes and the context in which they are being made. -->

## Code Compatibility

-   [ ] I have performed a self-review of my code

### Code Performance and Testing

-   [ ] I have added tests that prove my fix is effective or that my feature works
-   [ ] I ran the [complete Pytest test](https://anemoi.readthedocs.io/projects/training/en/latest/dev/testing.html) suite locally, and they pass
-   [ ] I have tested the changes on a single GPU
-   [ ] I have tested the changes on multiple GPUs / multi-node setups
-   [ ] I have run the [Benchmark Profiler](https://anemoi.readthedocs.io/projects/training/en/latest/user-guide/benchmarking.html) against the old version of the code
-   [ ] If they new features introduces modifications at config level, I have made sure to update Pydantic Schemas and default configs accordingly

<!-- In case this affects the model sharding or other specific components please describe these here. -->

### Dependencies

-   [ ] I have ensured that the code is still pip-installable after the changes and runs
-   [ ] I have tested that new dependencies themselves are pip-installable.
-   [ ] I have not introduced new dependencies in the inference portion of the pipeline

<!-- List any new dependencies that are required for this change and the justification to add them. -->

### Documentation

-   [ ] My code follows the style guidelines of this project
-   [ ] I have updated the documentation and docstrings to reflect the changes
-   [ ] I have added comments to my code, particularly in hard-to-understand areas

<!-- Describe any major updates to the documentation -->

## Additional Notes

<!-- Include any additional information, caveats, or considerations that the reviewer should be aware of. -->

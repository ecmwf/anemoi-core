##############
 Contributing
##############

Thank you for your interest in Anemoi! This guide will show you how to
contribute to the Anemoi packages.

****************
 Raise an issue
****************

If you encounter a bug or have a feature request, the first step is to
let us know by raising an issue on GitHub using the following steps:

#. Check the existing issues to avoid duplicates.
#. If it's a new issue, create a detailed bug report or feature request
   by filling in the issue template.
#. Use clear, descriptive titles and provide as much relevant
   information as possible.
#. If you have a bug, include the steps to reproduce it.
#. If you have a feature request, describe the use case and expected
   behaviour.
#. If you are interested in solving the issue yourself, assign the issue
   to yourself and follow the steps below.

**********************
 Developing in Anemoi
**********************

For contributing to the development of the Anemoi packages, please
follow these steps:

#. Fork the anemoi repository on GitHub to your personal/organisation
   account. See the `GitHub tutorial
   <https://docs.github.com/en/get-started/quickstart/fork-a-repo>`_.

#. Set up the development environment following the instructions in the
   :ref:`setting-up-the-development-environment` section.

#. Create a new branch for your developments, following the
   :ref:`branching-guidelines`.

#. Make your changes and ensure that your changes adhere to the
   :ref:`development-guidelines`.

#. Commit the changes using the :ref:`commit-guidelines`.

#. Push your branch to your fork on GitHub.

#. Open a Pull Request against the `main` branch of the original
   repository and fill in the Pull Request template.

#. Request a review from maintainers or other contributors, which will
   follow the :ref:`code-review-process`.

.. _code-review-process:

*********************
 Code Review Process
*********************

The Anemoi packages have a set of automated checks to enforce coding
guidelines. These checks are run via GitHub Actions on every Pull
Request. For security reasons, maintainers must review code changes
before enabling automated checks.

#. Ensure that all the :ref:`development-guidelines` criteria are met
   before submitting a Pull Request.
#. Request a review from maintainers or other contributors, noting that
   support is on a best-efforts basis.
#. After an initial review, a maintainer will enable automated checks to
   run on the Pull Request.
#. Reviewers may provide feedback or request changes to your
   contribution.
#. Once approved, a maintainer will merge your Pull Request into the
   appropriate branch.

.. _development-guidelines:

########################
 Development Guidelines
########################

Please follow these development guidelines:

#. Open an issue before starting a feature or bug fix to discuss the
   approach with maintainers adn other users.
#. Ensure high-quality code with appropriate tests, documentation,
   linting, and style checks. TODO: Add reference to tests
#. Follow the :ref:`branching-guidelines`.
#. Follow the :ref:`commit-guidelines`.

.. _branching-guidelines:

**********************
 Branching Guidelines
**********************

-  Use feature branches for new features (e.g., `feature/your-feature`)
-  Use fix branches for bug fixes (e.g., `fix/your-bug`)
-  Use a descriptive name that indicates the purpose of the branch
-  Keep branches up to date with `main` before opening a Pull Request

.. _commit-guidelines:

*******************
 Commit Guidelines
*******************

When making commits to the repository, please follow these guidelines:

#. Make small, focused commits with clear and concise messages.

#. Follow the `Conventional Commits guidelines
   <https://www.conventionalcommits.org/>`_. The format is:
   ``type[(scope)][!]: description``. For example:

   - ``feat(training): add new loss function``
   - ``fix(graphs): resolve node indexing bug``
   - ``docs(readme): update installation steps``
   - ``feat(models)!: change model input format`` (breaking change)
   - ``refactor!: restructure project layout`` (breaking change)

   Common types include:

   - ``feat``: New feature
   - ``fix``: Bug fix
   - ``docs``: Documentation only
   - ``style``: Code style changes
   - ``refactor``: Code changes that neither fix bugs nor add features
   - ``test``: Adding or modifying tests
   - ``chore``: Maintenance tasks

   Add ``!`` after the type/scope to indicate a breaking change.

#. Use present tense and imperative mood in commit messages (e.g., "Add
   feature" not "Added feature").

#. Reference relevant issue numbers in commit messages when applicable
   (e.g., "fix: resolve data loading issue #123").

While these commit message conventions are recommended for all branches in
Anemoi, they are strictly enforced only for commits to the ``main``
branch. This enforcement is particularly important as our automated
release process (`release-please<https://github.com/googleapis/release-please>`_) relies on conventional commits to
generate changelogs and determine version bumps automatically.

.. _pullrequest-guidelines:

*************************
 Pull Request Guidelines
*************************

When submitting Pull Requests (PRs), please follow these guidelines:

#. Open a draft Pull Request early in your development process. This helps:

   - Make your work visible to other contributors.
   - Get early feedback on your approach.
   - Avoid duplicate efforts.
   - Track progress on complex changes.

#. Fill the PR template completely, including:

   - Clear description of the changes.
   - Link to related issues using GitHub keywords (e.g., "Fixes #123").
   - List of notable changes.
   - Any breaking changes or deprecations.
   - Testing instructions if applicable.

#. Ensure the PR title follows the :ref:`commit-guidelines`, as this will become
   the squash commit message when merged to ``main``.

#. Keep your PR focused and of reasonable size:

   - One PR should address one concern.
   - Split large changes into smaller, logical PRs.
   - Update documentation along with code changes.

#. Before marking as ready for review:

   - Ensure all tests pass locally.
   - Address any automated check failures.
   - Review your own changes.
   - Update based on any feedback received while in draft.

#. When ready for review:

   - Mark the PR as "Ready for Review"
   - Request reviews from appropriate team members.
   - Be responsive to review comments.
   - Update the PR description if significant changes are made.

#. After approval:

   - PRs are merged using squash merge to maintain a clean history.
   - The squash commit message will use the PR title.



****************
 Code Standards
****************


*************
 Type Hints
*************

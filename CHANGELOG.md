# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please add your functional changes to the appropriate section in the PR.
Keep it human-readable, your future self will thank you!

## [Unreleased]

### Added

#### Subcommands
 - Subcommand for training `anemoi-training train`
 - Subcommand for config generation of configs
 - Subcommand for mlflow: login and sync
 - Subcommand for checkpoint handling

#### Functionality
 - Searchpaths for Hydra configs, to enable configs in CWD, `ANEMOI_CONFIG_PATH` env, and `.config/anemoi/training` in addition to package defaults
 - MlFlow token authentication
 - Configurable pressure level scaling

#### Continuous Integration / Deployment
 - Downstream CI to test all dependencies with changes
 - Changelog Status check
 - Readthedocs PR builder
 - Changelog Release Updater Workflow

#### Miscellaneous
 - Extended ruff Ruleset
 - Added Docsig pre-commit hook
 - `__future__` annotations for typehints
 - Added Typehints where missing
 - Added Changelog
 - Correct errors in callback plots

### Changed

#### Move to Anemoi Ecosystem
 - Fixed PyPI packaging
 - Use of Anemoi models
 - Use of Anemoi graphs
 - Adjusted tests to work with new Anemoi ecosystem
 - Adjusted configs to reasonable common defaults

#### Functionality
 - Changed hardware-specific keys from configs to `???` to trigger "missing"
 - `__len__` of NativeGridDataset
 - Configurable dropout in attention layer

#### Docs
 - Fixed docstrings

#### Miscellaneous
 - Moved callbacks into folder to fascilitate future refactor
 - Adjusted PyPI release infrastructure to common ECMWF workflow
 - Bumped versions in Pre-commit hooks
 - Fix crash when logging hyperparameters with missing values in the config
 - Fixed "null" tracker metadata when tracking is disabled, now returns an empty dict

### Removed
 - Dependency on mlflow-export-import
 - Specific user configs

<!-- Add Git Diffs for Links above -->

[unreleased]: https://github.com/ecmwf/anemoi-training/compare/x.x.x...HEAD

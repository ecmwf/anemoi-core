# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please add your functional changes to the appropriate section in the PR.
Keep it human-readable, your future self will thank you!

## [Unreleased](https://github.com/ecmwf/anemoi-training/compare/0.2.0...HEAD)

### Added
- Mlflow-sync to include new tag for server to server syncing [#83] (https://github.com/ecmwf/anemoi-training/pull/83)
- Mlflow-sync to include functionality to resume and fork server2server runs [#83] (https://github.com/ecmwf/anemoi-training/pull/83)
- Rollout training for Limited Area Models. [#79](https://github.com/ecmwf/anemoi-training/pulls/79)
- Feature: New `Boolean1DMask` class. Enables rollout training for limited area models. [#79](https://github.com/ecmwf/anemoi-training/pulls/79)

### Fixed
- Mlflow-sync to handle creation of new experiments in the remote server [#83] (https://github.com/ecmwf/anemoi-training/pull/83)
- Fix for multi-gpu when using mlflow due to refactoring of _get_mlflow_run_params function [#99] (https://github.com/ecmwf/anemoi-training/pull/99)
- ci: fix pyshtools install error (#100) https://github.com/ecmwf/anemoi-training/pull/100

### Changed
- Update copyright notice

## [0.2.0 - Feature release](https://github.com/ecmwf/anemoi-training/compare/0.1.0...0.2.0) - 2024-10-16

- Make pin_memory of the Dataloader configurable (#64)

### Added

- Add anemoi-transform link to documentation
- Codeowners file (#56)
- Changelog merge strategy (#56)

#### Miscellaneous

- Introduction of remapper to anemoi-models leads to changes in the data indices. Some preprocessors cannot be applied in-place anymore.

#### Functionality

- Enable the callback for plotting a histogram for variables containing NaNs
- Enforce same binning for histograms comparing true data to predicted data
- Fix: Inference checkpoints are now saved according the frequency settings defined in the config [#37](https://github.com/ecmwf/anemoi-training/pull/37)
- Feature: Add configurable models [#50](https://github.com/ecmwf/anemoi-training/pulls/50)
- Feature: Authentication support for mlflow sync - [#51](https://github.com/ecmwf/anemoi-training/pull/51)
- Feature: Support training for datasets with missing time steps [#48](https://github.com/ecmwf/anemoi-training/pulls/48)
- Feature: `AnemoiMlflowClient`, an mlflow client with authentication support [#86](https://github.com/ecmwf/anemoi-training/pull/86)
- Long Rollout Plots

### Fixed

- Fix `TypeError` raised when trying to JSON serialise `datetime.timedelta` object - [#43](https://github.com/ecmwf/anemoi-training/pull/43)
- Bugfixes for CI (#56)
- Fix `mlflow` subcommand on python 3.9 [#62](https://github.com/ecmwf/anemoi-training/pull/62)
- Show correct subcommand in MLFlow - Addresses [#39](https://github.com/ecmwf/anemoi-training/issues/39) in [#61](https://github.com/ecmwf/anemoi-training/pull/61)
- Fix interactive multi-GPU training [#82](https://github.com/ecmwf/anemoi-training/pull/82)
- Allow 500 characters in mlflow logging [#88](https://github.com/ecmwf/anemoi-training/pull/88)

### Changed

- Updated configuration examples in documentation and corrected links - [#46](https://github.com/ecmwf/anemoi-training/pull/46)
- Remove credential prompt from mlflow login, replace with seed refresh token via web - [#78](https://github.com/ecmwf/anemoi-training/pull/78)
- Update CODEOWNERS

## [0.1.0 - Anemoi training - First release](https://github.com/ecmwf/anemoi-training/releases/tag/0.1.0) - 2024-08-16

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
- fix error in the default config
- example slurm config
- ability to configure precip-type plots

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

- First draft on Read the Docs
- Fixed docstrings

#### Miscellaneous

- Moved callbacks into folder to fascilitate future refactor
- Adjusted PyPI release infrastructure to common ECMWF workflow
- Bumped versions in Pre-commit hooks
- Fix crash when logging hyperparameters with missing values in the config
- Fixed "null" tracker metadata when tracking is disabled, now returns an empty dict
- Pinned numpy<2 until we can test all migration
- (ci): path ignore of docs for downstream ci
- (ci): remove yaml anchor, unsupported by Github
- ci: make python QA reusable
- ci: permissions on changelog updater

### Removed

- Dependency on mlflow-export-import
- Specific user configs
- **len** function of NativeGridDataset as it lead to bugs

<!-- Add Git Diffs for Links above -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please add your functional changes to the appropriate section in the PR.
Keep it human-readable, your future self will thank you!

## [Unreleased](https://github.com/ecmwf/anemoi-training/compare/0.1.0...HEAD)

### Added

#### Miscellaneous

- Introduction of remapper to anemoi-models leads to changes in the data indices. Some preprocessors cannot be applied in-place anymore.

#### Functionality

- Enable the callback for plotting a histogram for variables containing NaNs
- Enforce same binning for histograms comparing true data to predicted data
- Fix: Inference checkpoints are now saved according the frequency settings defined in the config
- Feature: Add configurable models [#50](https://github.com/ecmwf/anemoi-training/pulls/50)

### Fixed

- Fix `TypeError` raised when trying to JSON serialise `datetime.timedelta` object - [#43](https://github.com/ecmwf/anemoi-training/pull/43)

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

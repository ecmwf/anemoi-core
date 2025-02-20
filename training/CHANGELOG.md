# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please add your functional changes to the appropriate section in the PR.
Keep it human-readable, your future self will thank you!

## [0.3.3](https://github.com/ecmwf/anemoi-core/compare/training-0.3.2...training-0.3.3) (2025-02-05)


### Features

* make flash attention configurable ([#60](https://github.com/ecmwf/anemoi-core/issues/60)) ([41fcab6](https://github.com/ecmwf/anemoi-core/commit/41fcab6335b334fdbebeb944c904cdbea6388889))
* Model Freezing ❄️  ([#61](https://github.com/ecmwf/anemoi-core/issues/61)) ([54e42cf](https://github.com/ecmwf/anemoi-core/commit/54e42cf42a47e00be96b464f501f6a462cd1bca4))
* **models:** normalization layers ([#47](https://github.com/ecmwf/anemoi-core/issues/47)) ([0e1c7c4](https://github.com/ecmwf/anemoi-core/commit/0e1c7c4840138debf877bb954b45f4c3a1cd0e33))


### Bug Fixes

* cancel RTD builds on no change ([#97](https://github.com/ecmwf/anemoi-core/issues/97)) ([36522d8](https://github.com/ecmwf/anemoi-core/commit/36522d87cdd95a5cb54b4c865eca67a64e22fffa))
* only load shards of grid into cpu mem if possible ([#83](https://github.com/ecmwf/anemoi-core/issues/83)) ([abbef4b](https://github.com/ecmwf/anemoi-core/commit/abbef4bf505af553584d9e0d4825db544d1c9ca7))
* pin dask version to 2024.12.1  ([#94](https://github.com/ecmwf/anemoi-core/issues/94)) ([074c0f2](https://github.com/ecmwf/anemoi-core/commit/074c0f226bf6541df977e43eff626a68d2f1f1fe))
* **training:** profiler 'Model Summary' works when sharding models over multiple GPUs ([#90](https://github.com/ecmwf/anemoi-core/issues/90)) ([9d9e89a](https://github.com/ecmwf/anemoi-core/commit/9d9e89a8fd6b90ecd62f799d942c84a4e984b9ed))
* update configs to avoid DeprecationWarning ([#53](https://github.com/ecmwf/anemoi-core/issues/53)) ([3560290](https://github.com/ecmwf/anemoi-core/commit/356029039e8406bc02a2b22f56107babbf2e7551))


### Documentation

* **graphs:** Refactor anemoi-graphs documentation ([#49](https://github.com/ecmwf/anemoi-core/issues/49)) ([29942b6](https://github.com/ecmwf/anemoi-core/commit/29942b6c088c7a2c6ff3f8ac13277041086cda9f))
* Improve installation docs ([#91](https://github.com/ecmwf/anemoi-core/issues/91)) ([0b5f8fb](https://github.com/ecmwf/anemoi-core/commit/0b5f8fb8b93555d76ebe3316c430121350bf5243))
* point RTD to right subfolder ([5a80cb6](https://github.com/ecmwf/anemoi-core/commit/5a80cb6047e864ea97bed06a76ddc54507e5fcbe))
* Tidy for core ([b24c521](https://github.com/ecmwf/anemoi-core/commit/b24c521c447272afd1b209745b24d16794cdb85a))

## [Unreleased](https://github.com/ecmwf/anemoi-training/compare/0.3.2...HEAD)

### Fixed

- Profilers 'Model Summary' feature works when the model is sharded across GPUs [#90](https://github.com/ecmwf/anemoi-core/pull/90)

## [0.3.2 - Multiple Fixes, Checkpoint updates, Stretched-grid/LAM updates](https://github.com/ecmwf/anemoi-training/compare/0.3.1...0.3.2) - 2024-12-19

### Fixed

- Not update NaN-weight-mask for loss function when using remapper and no imputer [#178](https://github.com/ecmwf/anemoi-training/pull/178)
- Dont crash when using the profiler if certain env vars arent set [#180](https://github.com/ecmwf/anemoi-training/pull/180)
- Remove saving of metadata to training checkpoint [#57](https://github.com/ecmwf/anemoi-training/pull/190)
- Fixes to callback plots [#182] (power spectrum large numpy array error + precip cmap for cases where precip is prognostic).
- GraphTrainableParameters callback will log a warning when no trainable parameters are specified  [#173](https://github.com/ecmwf/anemoi-training/pull/173)
- Fixes to checkpoint saving - ensure last checkpoint if saving when using max_steps [#191] (https://github.com/ecmwf/anemoi-training/pull/191)
- Identify stretched grid models based on graph rather than configuration file [#204](https://github.com/ecmwf/anemoi-training/pull/204)

### Added

- Introduce variable to configure: transfer_learning -> bool, True if loading checkpoint in a transfer learning setting.
-
<b> TRANSFER LEARNING</b>: enabled new functionality. You can now load checkpoints from different models and different training runs.
- Introduce (optional) variable to configure: transfer_learning -> bool, True if loading checkpoint in a transfer learning setting.
- <b> TRANSFER LEARNING</b>: enabled new functionality. You can now load checkpoints from different models and different training runs.
- Effective batch size: `(config.dataloader.batch_size["training"] * config.hardware.num_gpus_per_node * config.hardware.num_nodes) // config.hardware.num_gpus_per_model`.
  Used for experiment reproducibility across different computing configurations.
- Added a check for the variable sorting on pre-trained/finetuned models [#120](https://github.com/ecmwf/anemoi-training/pull/120)
- Added default configuration files for stretched grid and limited area model experiments [173](https://github.com/ecmwf/anemoi-training/pull/173)
- Added new metrics for stretched grid models to track losses inside/outside the regional domain [#199](https://github.com/ecmwf/anemoi-training/pull/199)
- <b> Model Freezing ❄️</b>: enabled new functionality. You can now Freeze parts of your model by specifying a list of submodules to freeze with the new config parameter: submodules_to_freeze.
- Introduce (optional) variable to configure: submodules_to_freeze -> List[str], list of submodules to freeze.
- Add supporting arrrays (numpy) to checkpoint
- Support for masking out unconnected nodes in LAM [#171](https://github.com/ecmwf/anemoi-training/pull/171)
- Improved validation metrics, allow 'all' to be scaled [#202](https://github.com/ecmwf/anemoi-training/pull/202)
- Added default configuration files for hierarchical processor [175](https://github.com/ecmwf/anemoi-training/pull/175)

### Changed

### Removed

- Removed the resolution config entry [#120](https://github.com/ecmwf/anemoi-training/pull/120)

## [0.3.1 - AIFS v0.3 Compatibility](https://github.com/ecmwf/anemoi-training/compare/0.3.0...0.3.1) - 2024-11-28

### Changed

- Perform full shuffle of training dataset [#153](https://github.com/ecmwf/anemoi-training/pull/153)

### Fixed

- Update `n_pixel` used by datashader to better adapt across resolutions [#152](https://github.com/ecmwf/anemoi-training/pull/152)
- Fixed bug in power spectra plotting for the n320 resolution.
- Allow histogram and spectrum plot for one variable [#165](https://github.com/ecmwf/anemoi-training/pull/165)

### Added

- Introduce variable to configure (Cosine Annealing) optimizer warm up [#155](https://github.com/ecmwf/anemoi-training/pull/155)
- Add reader groups to reduce CPU memory usage and increase dataloader throughput [#76](https://github.com/ecmwf/anemoi-training/pull/76)
- Bump `anemoi-graphs` version to 0.4.1 [#159](https://github.com/ecmwf/anemoi-training/pull/159)

## [0.3.0 - Loss & Callback Refactors](https://github.com/ecmwf/anemoi-training/compare/0.2.2...0.3.0) - 2024-11-14

### Fixed

- Rename loss_scaling to variable_loss_scaling [#138](https://github.com/ecmwf/anemoi-training/pull/138)
- Refactored callbacks. [#60](https://github.com/ecmwf/anemoi-training/pulls/60)
  - Updated docs [#115](https://github.com/ecmwf/anemoi-training/pull/115)
  - Fix enabling LearningRateMonitor [#119](https://github.com/ecmwf/anemoi-training/pull/119)

- Refactored rollout [#87](https://github.com/ecmwf/anemoi-training/pulls/87)
  - Enable longer validation rollout than training

- Expand iterables in logging [#91](https://github.com/ecmwf/anemoi-training/pull/91)
  - Save entire config in mlflow


### Added

- Included more loss functions and allowed configuration [#70](https://github.com/ecmwf/anemoi-training/pull/70)
- Include option to use datashader and optimised asyncronohous callbacks [#102](https://github.com/ecmwf/anemoi-training/pull/102)
  - Fix that applies the metric_ranges in the post-processed variable space [#116](https://github.com/ecmwf/anemoi-training/pull/116)

- Allow updates to scalars [#137](https://github.com/ecmwf/anemoi-training/pulls/137)
  - Add without subsetting in ScaleTensor

- Sub-hour datasets [#63](https://github.com/ecmwf/anemoi-training/pull/63)
- Add synchronisation workflow [#92](https://github.com/ecmwf/anemoi-training/pull/92)
- Feat: Anemoi Profiler compatible with mlflow and using Pytorch (Kineto) Profiler for memory report [38](https://github.com/ecmwf/anemoi-training/pull/38/)
- Feat: Save a gif for longer rollouts in validation [#65](https://github.com/ecmwf/anemoi-training/pull/65)
- New limited area config file added, limited_area.yaml. [#134](https://github.com/ecmwf/anemoi-training/pull/134/)
- New stretched grid config added, stretched_grid.yaml [#133](https://github.com/ecmwf/anemoi-training/pull/133)
- Functionality to change the weight attribute of nodes in the graph at the start of training without re-generating the graph. [#136] (https://github.com/ecmwf/anemoi-training/pull/136)
- Custom System monitor for Nvidia and AMD GPUs [#147](https://github.com/ecmwf/anemoi-training/pull/147)

### Changed

- Renamed frequency keys in callbacks configuration. [#118](https://github.com/ecmwf/anemoi-training/pull/118)
- Modified training configuration to support max_steps and tied lr iterations to max_steps by default [#67](https://github.com/ecmwf/anemoi-training/pull/67)
- Merged node & edge trainable feature callbacks into one. [#135](https://github.com/ecmwf/anemoi-training/pull/135)
- Increase the default MlFlow HTTP max retries [#111](https://github.com/ecmwf/anemoi-training/pull/111)

### Removed

## [0.2.2 - Maintenance: pin python <3.13](https://github.com/ecmwf/anemoi-training/compare/0.2.1...0.2.2) - 2024-10-28

### Changed

- Lock python version <3.13 [#107](https://github.com/ecmwf/anemoi-training/pull/107)

## [0.2.1 - Bugfix: resuming mlflow runs](https://github.com/ecmwf/anemoi-training/compare/0.2.0...0.2.1) - 2024-10-24

### Added

- Mlflow-sync to include new tag for server to server syncing [#83](https://github.com/ecmwf/anemoi-training/pull/83)
- Mlflow-sync to include functionality to resume and fork server2server runs [#83](https://github.com/ecmwf/anemoi-training/pull/83)
- Rollout training for Limited Area Models. [#79](https://github.com/ecmwf/anemoi-training/pulls/79)
- Feature: New `Boolean1DMask` class. Enables rollout training for limited area models. [#79](https://github.com/ecmwf/anemoi-training/pulls/79)

### Fixed

- Fix pre-commit regex
- Mlflow-sync to handle creation of new experiments in the remote server [#83] (https://github.com/ecmwf/anemoi-training/pull/83)
- Fix for multi-gpu when using mlflow due to refactoring of _get_mlflow_run_params function [#99] (https://github.com/ecmwf/anemoi-training/pull/99)
- ci: fix pyshtools install error (#100) https://github.com/ecmwf/anemoi-training/pull/100
- Mlflow-sync to handle creation of new experiments in the remote server [#83](https://github.com/ecmwf/anemoi-training/pull/83)
- Fix for multi-gpu when using mlflow due to refactoring of _get_mlflow_run_params function [#99](https://github.com/ecmwf/anemoi-training/pull/99)
- ci: fix pyshtools install error [#100](https://github.com/ecmwf/anemoi-training/pull/100)
- Fix `__version__` import in init

### Changed

- Update copyright notice

## [0.2.0 - Feature release](https://github.com/ecmwf/anemoi-training/compare/0.1.0...0.2.0) - 2024-10-16

- Make pin_memory of the Dataloader configurable (#64)

### Added

- Add anemoi-transform link to documentation
- Codeowners file (#56)
- Changelog merge strategy (#56)
- Contributors file (#106)

#### Miscellaneous

- Introduction of remapper to anemoi-models leads to changes in the data indices. Some preprocessors cannot be applied in-place anymore.

- Variable Bounding as configurable model layers [#13](https://github.com/ecmwf/anemoi-models/issues/13)


#### Functionality

- Enable the callback for plotting a histogram for variables containing NaNs
- Enforce same binning for histograms comparing true data to predicted data
- Fix: Inference checkpoints are now saved according the frequency settings defined in the config [#37](https://github.com/ecmwf/anemoi-training/pull/37)
- Feature: Add configurable models [#50](https://github.com/ecmwf/anemoi-training/pulls/50)
- Feature: Authentication support for mlflow sync - [#51](https://github.com/ecmwf/anemoi-training/pull/51)
- Feature: Support training for datasets with missing time steps [#48](https://github.com/ecmwf/anemoi-training/pulls/48)
- Feature: `AnemoiMlflowClient`, an mlflow client with authentication support [#86](https://github.com/ecmwf/anemoi-training/pull/86)
- Long Rollout Plots
- Mask NaN values in training loss function [#72](https://github.com/ecmwf/anemoi-training/pull/72) and [#271](https://github.com/ecmwf-lab/aifs-mono/issues/271)

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
- Change how mlflow measures CPU Memory usage - [94](https://github.com/ecmwf/anemoi-training/pull/94)

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

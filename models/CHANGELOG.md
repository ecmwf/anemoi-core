# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please add your functional changes to the appropriate section in the PR.
Keep it human-readable, your future self will thank you!

## 0.4.2 (2025-02-04)

<!-- Release notes generated using configuration in .github/release.yml at main -->

## What's Changed
### Training
* fix(training, plots) Exclude nans from error colorbars by @anaprietonem in https://github.com/ecmwf/anemoi-core/pull/59
* fix(training): bump anemoi-datasets required version to 0.5.13 by @JPXKQX in https://github.com/ecmwf/anemoi-core/pull/74
* chore(training): Add default config files for 2 and 3 level hierarchical processors by @JPXKQX in https://github.com/ecmwf/anemoi-core/pull/50
* fix: update graph configs to avoid DeprecationWarning for area weights by @JPXKQX in https://github.com/ecmwf/anemoi-core/pull/53
* feat(models): normalization layers by @jakob-schloer in https://github.com/ecmwf/anemoi-core/pull/47
* Fix crash in diagnostic plots (colorbar limits) by @lzampier in https://github.com/ecmwf/anemoi-core/pull/85
* fix(training): profiler 'Model Summary' works when sharding models over multiple GPUs by @cathalobrien in https://github.com/ecmwf/anemoi-core/pull/90
* docs(graphs): Refactor anemoi-graphs documentation by @JPXKQX in https://github.com/ecmwf/anemoi-core/pull/49
* fix: pin dask version to 2024.12.1  by @JPXKQX in https://github.com/ecmwf/anemoi-core/pull/94
* docs: Improve installation docs by @HCookie in https://github.com/ecmwf/anemoi-core/pull/91
* docs: cancel RTD builds on PRs without change by @JesperDramsch in https://github.com/ecmwf/anemoi-core/pull/97
* feat: Model Freezing ❄️  by @icedoom888 in https://github.com/ecmwf/anemoi-core/pull/61
* feat: make flash attention configurable by @theissenhelen in https://github.com/ecmwf/anemoi-core/pull/60
* fix: cpu memory savings of sharded dataloader by @japols in https://github.com/ecmwf/anemoi-core/pull/83
* chore: synced file(s) with ecmwf-actions/reusable-workflows by @DeployDuck in https://github.com/ecmwf/anemoi-core/pull/84
### Models
* feature(models): Add model comm group to predict_step  by @cathalobrien in https://github.com/ecmwf/anemoi-core/pull/77
* Implementation of NormalizedReluBounding for non-zero thresholds by @lzampier in https://github.com/ecmwf/anemoi-core/pull/64
* fix: normalise in place to reduce memory by @japols in https://github.com/ecmwf/anemoi-core/pull/82
* feat(models): use num_layers of the processor in hierarchical graphs by @JPXKQX in https://github.com/ecmwf/anemoi-core/pull/78
* fix: default behaviour for kernel_layers when not set in config. by @jakob-schloer in https://github.com/ecmwf/anemoi-core/pull/93
* fix:  bug in variables ordering in NormalizedReluBounding by @lzampier in https://github.com/ecmwf/anemoi-core/pull/98
* feat(models): Copy Imputer by @icedoom888 in https://github.com/ecmwf/anemoi-core/pull/72
### Graphs
* feat(graphs,plots): expand support for multi-dimensional node attributes by @JPXKQX in https://github.com/ecmwf/anemoi-core/pull/48
* feat(graphs): New Edge Attribute: AttributeFromNode by @icedoom888 in https://github.com/ecmwf/anemoi-core/pull/62
* feat: support ReducedGaussianGridNodes by @JPXKQX in https://github.com/ecmwf/anemoi-core/pull/54
* chore(main): Preparing Next Release for  anemoi-graphs 0.4.3 by @DeployDuck in https://github.com/ecmwf/anemoi-core/pull/110
* chore(main): Preparing Next Release for  anemoi-graphs 0.4.4 by @DeployDuck in https://github.com/ecmwf/anemoi-core/pull/111
### Other Changes
* pre-commits-for-models-graphs-dev by @sahahner in https://github.com/ecmwf/anemoi-core/pull/45
* ci(docs): bring ReadTheDocs CI pipeline by @JPXKQX in https://github.com/ecmwf/anemoi-core/pull/73
* ci: Reinstantiate CI files by @JesperDramsch in https://github.com/ecmwf/anemoi-core/pull/75
* ci: Propose release-please implementation by @JesperDramsch in https://github.com/ecmwf/anemoi-core/pull/100

## New Contributors
* @anaprietonem made their first contribution in https://github.com/ecmwf/anemoi-core/pull/59
* @cathalobrien made their first contribution in https://github.com/ecmwf/anemoi-core/pull/77
* @lzampier made their first contribution in https://github.com/ecmwf/anemoi-core/pull/64
* @japols made their first contribution in https://github.com/ecmwf/anemoi-core/pull/82
* @JesperDramsch made their first contribution in https://github.com/ecmwf/anemoi-core/pull/75
* @jakob-schloer made their first contribution in https://github.com/ecmwf/anemoi-core/pull/47
* @icedoom888 made their first contribution in https://github.com/ecmwf/anemoi-core/pull/61
* @theissenhelen made their first contribution in https://github.com/ecmwf/anemoi-core/pull/60
* @DeployDuck made their first contribution in https://github.com/ecmwf/anemoi-core/pull/84

**Full Changelog**: https://github.com/ecmwf/anemoi-core/compare/anemoi-models-0.4.1...anemoi-models-0.4.2

## [Unreleased](https://github.com/ecmwf/anemoi-models/compare/0.4.0...HEAD)

### Added

- New AnemoiModelEncProcDecHierarchical class available in models [#37](https://github.com/ecmwf/anemoi-models/pull/37)
- Mask NaN values in training loss function [#56](https://github.com/ecmwf/anemoi-models/pull/56)
- Added dynamic NaN masking for the imputer class with two new classes: DynamicInputImputer, DynamicConstantImputer [#89](https://github.com/ecmwf/anemoi-models/pull/89)
- Reduced memory usage when using chunking in the mapper [#84](https://github.com/ecmwf/anemoi-models/pull/84)
- Added `supporting_arrays` argument, which contains arrays to store in checkpoints. [#97](https://github.com/ecmwf/anemoi-models/pull/97)
- Add remappers, e.g. link functions to apply during training to facilitate learning of variables with a difficult distribution [#88](https://github.com/ecmwf/anemoi-models/pull/88)
- Add Normalized Relu Bounding for minimum bounding thresholds different than 0 [#64](https://github.com/ecmwf/anemoi-core/pull/64)
- 'predict\_step' can take an optional model comm group. [#77](https://github.com/ecmwf/anemoi-core/pull/77)

## [0.4.0](https://github.com/ecmwf/anemoi-models/compare/0.3.0...0.4.0) - Improvements to Model Design

### Added

- Add synchronisation workflow [#60](https://github.com/ecmwf/anemoi-models/pull/60)
- Add anemoi-transform link to documentation
- Codeowners file
- Pygrep precommit hooks
- Docsig precommit hooks
- Changelog merge strategy
- configurabilty of the dropout probability in the the MultiHeadSelfAttention module
- Variable Bounding as configurable model layers [#13](https://github.com/ecmwf/anemoi-models/issues/13)
- GraphTransformerMapperBlock chunking to reduce memory usage during inference [#46](https://github.com/ecmwf/anemoi-models/pull/46)
- New `NamedNodesAttributes` class to handle node attributes in a more flexible way [#64](https://github.com/ecmwf/anemoi-models/pull/64)
- Contributors file [#69](https://github.com/ecmwf/anemoi-models/pull/69)

### Changed
- Bugfixes for CI
- Change Changelog CI to run after successful publish
- pytest for downstream-ci-hpc
- Update CODEOWNERS
- Fix pre-commit regex
- ci: extened python versions to include 3.11 and 3.12 [#66](https://github.com/ecmwf/anemoi-models/pull/66)
- Update copyright notice
- Fix `__version__` import in init
- Fix missing copyrights [#71](https://github.com/ecmwf/anemoi-models/pull/71)

### Removed

## [0.3.0](https://github.com/ecmwf/anemoi-models/compare/0.2.1...0.3.0) - Remapping of (meteorological) Variables

### Added

- CI workflow to update the changelog on release
- add configurability of flash attention (#47)
- configurabilty of the dropout probability in the the MultiHeadSelfAttention module
- CI workflow to update the changelog on release
- Remapper: Preprocessor for remapping one variable to multiple ones. Includes changes to the data indices since the remapper changes the number of variables. With optional config keywords.
- Codeowners file
- Pygrep precommit hooks
- Docsig precommit hooks
- Changelog merge strategy


### Changed

- Update CI to inherit from common infrastructue reusable workflows
- run downstream-ci only when src and tests folders have changed
- New error messages for wrongs graphs.
- Feature: Change model to be instantiatable in the interface, addressing [#28](https://github.com/ecmwf/anemoi-models/issues/28) through [#45](https://github.com/ecmwf/anemoi-models/pulls/45)
- Bugfixes for CI

### Removed

## [0.2.1](https://github.com/ecmwf/anemoi-models/compare/0.2.0...0.2.1) - Dependency update

### Added

- downstream-ci pipeline
- readthedocs PR update check action

### Removed

- anemoi-datasets dependency

## [0.2.0](https://github.com/ecmwf/anemoi-models/compare/0.1.0...0.2.0) - Support Heterodata

### Added

- Option to choose the edge attributes

### Changed

- Updated to support new PyTorch Geometric HeteroData structure (defined by `anemoi-graphs` package).

## [0.1.0](https://github.com/ecmwf/anemoi-models/releases/tag/0.1.0) - Initial Release

### Added

- Documentation
- Initial code release with models, layers, distributed, preprocessing, and data_indices
- Added Changelog

<!-- Add Git Diffs for Links above -->

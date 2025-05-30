# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please add your functional changes to the appropriate section in the PR.
Keep it human-readable, your future self will thank you!

## [0.7.0](https://github.com/ecmwf/anemoi-core/compare/models-0.6.0...models-0.7.0) (2025-05-30)


### ⚠ BREAKING CHANGES

* generalise activation function ([#163](https://github.com/ecmwf/anemoi-core/issues/163))

### Features

* generalise activation function ([#163](https://github.com/ecmwf/anemoi-core/issues/163)) ([98c4d06](https://github.com/ecmwf/anemoi-core/commit/98c4d06dfb5b79f605fab885c67a8a4cd6d35384))
* transformer mapper ([#179](https://github.com/ecmwf/anemoi-core/issues/179)) ([2cea7db](https://github.com/ecmwf/anemoi-core/commit/2cea7db51d5c5ef63bb4b9c266deb05fb2acf66f))


### Bug Fixes

* **models,training:** Remove unnecessary torch-geometric maximum version ([#326](https://github.com/ecmwf/anemoi-core/issues/326)) ([fe93ea8](https://github.com/ecmwf/anemoi-core/commit/fe93ea8feb379147a9f9e5c5358ea8144855dc77))
* remove activation entry from MLP noise block ([#340](https://github.com/ecmwf/anemoi-core/issues/340)) ([2d060f5](https://github.com/ecmwf/anemoi-core/commit/2d060f5e3382454b06c6369141942b8d6367fb4b))

## [0.6.0](https://github.com/ecmwf/anemoi-core/compare/models-0.5.1...models-0.6.0) (2025-05-15)


### ⚠ BREAKING CHANGES

* Rework Loss Scalings to provide better modularity ([#52](https://github.com/ecmwf/anemoi-core/issues/52))

### Features

* GraphtransformerProcessor chunking ([#66](https://github.com/ecmwf/anemoi-core/issues/66)) ([1daa9f2](https://github.com/ecmwf/anemoi-core/commit/1daa9f22ab36426602ab644de6a29ef5e296a485))


### Bug Fixes

* Rework Loss Scalings to provide better modularity ([#52](https://github.com/ecmwf/anemoi-core/issues/52)) ([162b906](https://github.com/ecmwf/anemoi-core/commit/162b9062882c321a4a265b0cf561be3f141ac97a))

## [0.5.1](https://github.com/ecmwf/anemoi-core/compare/models-0.5.0...models-0.5.1) (2025-04-30)


### Bug Fixes

* Adapt predict_step in model interface to pass on arguments for model classes ([#281](https://github.com/ecmwf/anemoi-core/issues/281)) ([a5b2643](https://github.com/ecmwf/anemoi-core/commit/a5b26432bc7b78577cd1febd5091b059cc82805c))

## [0.5.0](https://github.com/ecmwf/anemoi-core/compare/models-0.4.2...models-0.5.0) (2025-04-16)


### ⚠ BREAKING CHANGES

* **models,training:** temporal interpolation ([#153](https://github.com/ecmwf/anemoi-core/issues/153))
* **config:** Improved configuration and data structures ([#79](https://github.com/ecmwf/anemoi-core/issues/79))

### Features

* **config:** Improved configuration and data structures ([#79](https://github.com/ecmwf/anemoi-core/issues/79)) ([1f7812b](https://github.com/ecmwf/anemoi-core/commit/1f7812b559b51d842852df29ace7dda6d0f66ef2))
* Kcrps  ([#182](https://github.com/ecmwf/anemoi-core/issues/182)) ([8bbe898](https://github.com/ecmwf/anemoi-core/commit/8bbe89839e2eff3fcbc35613eb92920d4afc3276))
* **models,training:** temporal interpolation ([#153](https://github.com/ecmwf/anemoi-core/issues/153)) ([ea644ce](https://github.com/ecmwf/anemoi-core/commit/ea644ce1c9aef902333d9cbb30bcde0a3746fbcc))
* **models:** adding leaky boundings ([#256](https://github.com/ecmwf/anemoi-core/issues/256)) ([426e860](https://github.com/ecmwf/anemoi-core/commit/426e86048d6c0a03750fb0e205890841c27c8148))


### Bug Fixes

* pydantic schemas move ([#228](https://github.com/ecmwf/anemoi-core/issues/228)) ([6bca9bc](https://github.com/ecmwf/anemoi-core/commit/6bca9bc66ff54ac294d97793b8cebed1cd1bb8a4))


### Documentation

* Add subheadings to schema doc page ([#149](https://github.com/ecmwf/anemoi-core/issues/149)) ([d3c7de9](https://github.com/ecmwf/anemoi-core/commit/d3c7de905bced2dc9e75a92de4e9abf848936e62))
* fix documentation to refer to anemoi datasets instead of zarr datasets ([#154](https://github.com/ecmwf/anemoi-core/issues/154)) ([ad062b2](https://github.com/ecmwf/anemoi-core/commit/ad062b22cdd05354bc010eabbf8ffa806def081c))
* **models:** Docathon  ([#202](https://github.com/ecmwf/anemoi-core/issues/202)) ([5dba9d3](https://github.com/ecmwf/anemoi-core/commit/5dba9d34d65d4331dabd19355c7a31f7f1468fbf))
* **training:** Docathon ([#201](https://github.com/ecmwf/anemoi-core/issues/201)) ([e69430f](https://github.com/ecmwf/anemoi-core/commit/e69430f8c1ba8e7de50cd99f202e3f4876b806e0))
* Update docs for kcrps. ([#258](https://github.com/ecmwf/anemoi-core/issues/258)) ([79cbd1d](https://github.com/ecmwf/anemoi-core/commit/79cbd1d5e5f0f5aa82ce712bed474a6ad99f17e8))
* use new logo ([#140](https://github.com/ecmwf/anemoi-core/issues/140)) ([c269cea](https://github.com/ecmwf/anemoi-core/commit/c269cea3c84f2e35ef0a318e0cd1b769d285177c))

## [0.4.2](https://github.com/ecmwf/anemoi-core/compare/models-0.4.1...models-0.4.2) (2025-02-11)


### Features

* make flash attention configurable ([#60](https://github.com/ecmwf/anemoi-core/issues/60)) ([41fcab6](https://github.com/ecmwf/anemoi-core/commit/41fcab6335b334fdbebeb944c904cdbea6388889))
* **models:** Copy Imputer ([#72](https://github.com/ecmwf/anemoi-core/issues/72)) ([4690ed5](https://github.com/ecmwf/anemoi-core/commit/4690ed52b9996bc149417d3724c5cd68c234573f))
* **models:** normalization layers ([#47](https://github.com/ecmwf/anemoi-core/issues/47)) ([0e1c7c4](https://github.com/ecmwf/anemoi-core/commit/0e1c7c4840138debf877bb954b45f4c3a1cd0e33))
* **models:** use num_layers of the processor in hierarchical graphs ([#78](https://github.com/ecmwf/anemoi-core/issues/78)) ([7e080ed](https://github.com/ecmwf/anemoi-core/commit/7e080edec94fe5408b45cace339ff6d97f556160))


### Bug Fixes

* bug in variables ordering in NormalizedReluBounding ([#98](https://github.com/ecmwf/anemoi-core/issues/98)) ([f1cc2e6](https://github.com/ecmwf/anemoi-core/commit/f1cc2e66486f29f73ec8d805bf32790d19d44804))
* cancel RTD builds on no change ([#97](https://github.com/ecmwf/anemoi-core/issues/97)) ([36522d8](https://github.com/ecmwf/anemoi-core/commit/36522d87cdd95a5cb54b4c865eca67a64e22fffa))
* **models:** 74 imputer inference mode ([#127](https://github.com/ecmwf/anemoi-core/issues/127)) ([0a9cfa7](https://github.com/ecmwf/anemoi-core/commit/0a9cfa77f0b438d30fac9153a6c6f4cafa0a1c1b))
* normalise in place to reduce memory ([#82](https://github.com/ecmwf/anemoi-core/issues/82)) ([40dd1a1](https://github.com/ecmwf/anemoi-core/commit/40dd1a178a09afea58f6cf461e07c72ac8c6f23d))


### Documentation

* Improve installation docs ([#91](https://github.com/ecmwf/anemoi-core/issues/91)) ([0b5f8fb](https://github.com/ecmwf/anemoi-core/commit/0b5f8fb8b93555d76ebe3316c430121350bf5243))
* point RTD to right subfolder ([5a80cb6](https://github.com/ecmwf/anemoi-core/commit/5a80cb6047e864ea97bed06a76ddc54507e5fcbe))
* Tidy for core ([b24c521](https://github.com/ecmwf/anemoi-core/commit/b24c521c447272afd1b209745b24d16794cdb85a))

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

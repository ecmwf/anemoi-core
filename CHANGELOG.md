# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please add your functional changes to the appropriate section in the PR.
Keep it human-readable, your future self will thank you!

## [Unreleased](https://github.com/ecmwf/anemoi-models/compare/0.3.0...HEAD)

- Add synchronisation workflow

### Added

- Add anemoi-transform link to documentation
- Codeowners file
- Pygrep precommit hooks
- Docsig precommit hooks
- Changelog merge strategy
- configurabilty of the dropout probability in the the MultiHeadSelfAttention module
- Variable Bounding as configurable model layers [#13](https://github.com/ecmwf/anemoi-models/issues/13)
- GraphTransformerMapperBlock chunking to reduce memory usage during inference [#46](https://github.com/ecmwf/anemoi-models/pull/46)
- Mask NaN values in training loss function [#271](https://github.com/ecmwf-lab/aifs-mono/issues/271)
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
- Remapper: Preprocessor for remapping one variable to multiple ones. Includes changes to the data indices since the remapper changes the number of variables. With optional config keywords.

### Changed

- Update CI to inherit from common infrastructue reusable workflows
- run downstream-ci only when src and tests folders have changed
- New error messages for wrongs graphs.
- Feature: Change model to be instantiatable in the interface, addressing [#28](https://github.com/ecmwf/anemoi-models/issues/28) through [#45](https://github.com/ecmwf/anemoi-models/pulls/45)

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

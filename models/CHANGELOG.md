# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please add your functional changes to the appropriate section in the PR.
Keep it human-readable, your future self will thank you!

## [0.10.0](https://github.com/ecmwf/anemoi-core/compare/models-0.9.7...models-0.10.0) (2025-11-17)


### ⚠ BREAKING CHANGES

* **training:** remove support for EDA ([#651](https://github.com/ecmwf/anemoi-core/issues/651))

### Features

* **training:** Remove support for EDA ([#651](https://github.com/ecmwf/anemoi-core/issues/651)) ([921e108](https://github.com/ecmwf/anemoi-core/commit/921e108b8467f1bcd4e69516927efee2f58f9e33))


### Bug Fixes

* Basemodel.predict_step ([#672](https://github.com/ecmwf/anemoi-core/issues/672)) ([0c830e9](https://github.com/ecmwf/anemoi-core/commit/0c830e9f4dda94bdc571f6d3ea3f64e2701d87e7))
* **models:** Assert no dropout ([#638](https://github.com/ecmwf/anemoi-core/issues/638)) ([c1bbcec](https://github.com/ecmwf/anemoi-core/commit/c1bbcece6996d2808691741c5ef56f70e61fde52))
* Shard shape type hints ([#625](https://github.com/ecmwf/anemoi-core/issues/625)) ([fb201fd](https://github.com/ecmwf/anemoi-core/commit/fb201fd4c832fb90e7bb06dbc18371af1f214d03))
* Update readmes ([#655](https://github.com/ecmwf/anemoi-core/issues/655)) ([a58aa64](https://github.com/ecmwf/anemoi-core/commit/a58aa640212cdc6d5de6e38a6d11725d32d662b5))

## [0.9.7](https://github.com/ecmwf/anemoi-core/compare/models-0.9.6...models-0.9.7) (2025-10-20)


### Features

* **models:** Point-mlp proccesor ([#367](https://github.com/ecmwf/anemoi-core/issues/367)) ([ee2a067](https://github.com/ecmwf/anemoi-core/commit/ee2a067689b6efe18acb13ceac8ed0be512633d5))


### Bug Fixes

* **models:** Replace placeholder of migration script anemoi-models version in main ([#600](https://github.com/ecmwf/anemoi-core/issues/600)) ([9c18580](https://github.com/ecmwf/anemoi-core/commit/9c18580eda64b0eb53cf17652ea6b6675fbee77a))

## [0.9.6](https://github.com/ecmwf/anemoi-core/compare/models-0.9.5...models-0.9.6) (2025-10-09)


### Features

* Introducing AnemoiBaseModel ([#440](https://github.com/ecmwf/anemoi-core/issues/440)) ([eb3edc5](https://github.com/ecmwf/anemoi-core/commit/eb3edc59696a5a01a9de77aec72a68dc40928f92))
* Target indices ([#426](https://github.com/ecmwf/anemoi-core/issues/426)) ([d8db2a6](https://github.com/ecmwf/anemoi-core/commit/d8db2a6fc192bc49107df6c137ce4f56866ae4d4))

## [0.9.5](https://github.com/ecmwf/anemoi-core/compare/models-0.9.4...models-0.9.5) (2025-09-09)


### Features

* Flash attention v3 ([#479](https://github.com/ecmwf/anemoi-core/issues/479)) ([00f52df](https://github.com/ecmwf/anemoi-core/commit/00f52df292f8fb8dc0a865f6d288fa151c630a2c))


### Bug Fixes

* Set correct base package for migrations imports ([#531](https://github.com/ecmwf/anemoi-core/issues/531)) ([cfb80fe](https://github.com/ecmwf/anemoi-core/commit/cfb80fe6d5105873e89f20a9213f782b55aa57dd))
* Test dependencies ([#524](https://github.com/ecmwf/anemoi-core/issues/524)) ([3ac7d4f](https://github.com/ecmwf/anemoi-core/commit/3ac7d4fbc35e0ef0f54566454e235aeaf7f6da67))
* Truncation shard shapes ([#536](https://github.com/ecmwf/anemoi-core/issues/536)) ([507b441](https://github.com/ecmwf/anemoi-core/commit/507b44143fc35acc1d3b927cea95e9e1be120407))

## [0.9.4](https://github.com/ecmwf/anemoi-core/compare/models-0.9.3...models-0.9.4) (2025-09-02)


### Features

* Checkpoint migrations ([#386](https://github.com/ecmwf/anemoi-core/issues/386)) ([028f6eb](https://github.com/ecmwf/anemoi-core/commit/028f6eb6426c5ada0c5d95e6492accc99083b46f))

## [0.9.3](https://github.com/ecmwf/anemoi-core/compare/models-0.9.2...models-0.9.3) (2025-08-22)


### Bug Fixes

* Convert pre_processors from attribute to variable in models/encoder_processor_decoder ([#503](https://github.com/ecmwf/anemoi-core/issues/503)) ([60eb52b](https://github.com/ecmwf/anemoi-core/commit/60eb52b465ba260573566223966dede67cd44913))

## [0.9.2](https://github.com/ecmwf/anemoi-core/compare/models-0.9.1...models-0.9.2) (2025-08-20)


### Features

* Diffusion training ([#401](https://github.com/ecmwf/anemoi-core/issues/401)) ([f35ad67](https://github.com/ecmwf/anemoi-core/commit/f35ad673f680ba64acb5c6770a3114262b3b97fd))

## [0.9.1](https://github.com/ecmwf/anemoi-core/compare/models-0.9.0...models-0.9.1) (2025-08-08)


### Features

* Extend DelayedScalers into arbitary UpdatingScalers [#371](https://github.com/ecmwf/anemoi-core/issues/371)  ([b9d7726](https://github.com/ecmwf/anemoi-core/commit/b9d772659679b1d1744c9be6a6602673eb9e6969))
* **models:** Nan locations in imputer calculated on the fly [#378](https://github.com/ecmwf/anemoi-core/issues/378) ([b9d7726](https://github.com/ecmwf/anemoi-core/commit/b9d772659679b1d1744c9be6a6602673eb9e6969))
* **models:** Postprocessor for nans in diagnostic fields ([#461](https://github.com/ecmwf/anemoi-core/issues/461)) ([a7ff22e](https://github.com/ecmwf/anemoi-core/commit/a7ff22e44b956635bcd3e91b9d780aa041a617d3))


### Bug Fixes

* Improve device movement of scalers [#390](https://github.com/ecmwf/anemoi-core/issues/390) ([b9d7726](https://github.com/ecmwf/anemoi-core/commit/b9d772659679b1d1744c9be6a6602673eb9e6969))

## [0.9.0](https://github.com/ecmwf/anemoi-core/compare/models-0.8.1...models-0.9.0) (2025-08-01)


### ⚠ BREAKING CHANGES

* for schemas of data processors ([#433](https://github.com/ecmwf/anemoi-core/issues/433))

### Features

* **model:** Postprocessors for leaky boundings ([#315](https://github.com/ecmwf/anemoi-core/issues/315)) ([b54562b](https://github.com/ecmwf/anemoi-core/commit/b54562b83b4b5620891b28827964f9c554ee0615))
* **models:** Checkpointed Mapper Chunking ([#406](https://github.com/ecmwf/anemoi-core/issues/406)) ([8577772](https://github.com/ecmwf/anemoi-core/commit/8577772927a08d62db74159e1023f5db1dc39438))
* **models:** Mapper edge sharding ([#366](https://github.com/ecmwf/anemoi-core/issues/366)) ([326751d](https://github.com/ecmwf/anemoi-core/commit/326751d25f9bc299f3e19c795d9065a60a6af3d9))


### Bug Fixes

* Dropping 3.9 ([#436](https://github.com/ecmwf/anemoi-core/issues/436)) ([f6c0214](https://github.com/ecmwf/anemoi-core/commit/f6c0214ad09d217930956b7eddaf0c8b35a32185))
* For schemas of data processors ([#433](https://github.com/ecmwf/anemoi-core/issues/433)) ([539939b](https://github.com/ecmwf/anemoi-core/commit/539939be4c4392afcf8ccd73b8de7c44e4b32847))
* **models,traininig:** Hierarchical model + integration test ([#400](https://github.com/ecmwf/anemoi-core/issues/400)) ([71dfd89](https://github.com/ecmwf/anemoi-core/commit/71dfd89d4326d5e59c8ff8fef339b500110ded42))
* **models:** Remove repeated lines ([#377](https://github.com/ecmwf/anemoi-core/issues/377)) ([1f0b861](https://github.com/ecmwf/anemoi-core/commit/1f0b861062db023d7eaaf215846a66adb8560c5c))
* **models:** Uneven channel sharding ([#385](https://github.com/ecmwf/anemoi-core/issues/385)) ([dd095c4](https://github.com/ecmwf/anemoi-core/commit/dd095c416334975185232c0eea7cf98be3085f54))
* Pydantic model validator not working in transformer schema ([#422](https://github.com/ecmwf/anemoi-core/issues/422)) ([42f437a](https://github.com/ecmwf/anemoi-core/commit/42f437a282adbbec6c306037b48758ab02925631))
* Remove dead code and fix typo ([#357](https://github.com/ecmwf/anemoi-core/issues/357)) ([8c615ba](https://github.com/ecmwf/anemoi-core/commit/8c615ba87b68957b4cc53cd82d8f396f572b9943))

## [0.8.1](https://github.com/ecmwf/anemoi-core/compare/models-0.8.0...models-0.8.1) (2025-06-17)


### Features

* **models,training:** Shard everything ([#121](https://github.com/ecmwf/anemoi-core/issues/121)) ([06dde94](https://github.com/ecmwf/anemoi-core/commit/06dde94219119746215b767b846542ee31bbff63))


### Bug Fixes

* Do not inherit from a concrete class ([#359](https://github.com/ecmwf/anemoi-core/issues/359)) ([ca79375](https://github.com/ecmwf/anemoi-core/commit/ca7937590311c412b90dffad48fd8f25230fe5eb))
* Revert PR 359 ([#365](https://github.com/ecmwf/anemoi-core/issues/365)) ([fcebea4](https://github.com/ecmwf/anemoi-core/commit/fcebea484080c5333e8e213784372d1b13dab4c8))

## [0.8.0](https://github.com/ecmwf/anemoi-core/compare/models-0.7.0...models-0.8.0) (2025-06-05)


### ⚠ BREAKING CHANGES

* **models,training:** Remove multimapper ([#268](https://github.com/ecmwf/anemoi-core/issues/268))

### Features

* **models,training:** Remove multimapper ([#268](https://github.com/ecmwf/anemoi-core/issues/268)) ([0e8bb99](https://github.com/ecmwf/anemoi-core/commit/0e8bb998176bea2d653ca40772e4e6e1578551f7))


### Bug Fixes

* Dataset_order ([#334](https://github.com/ecmwf/anemoi-core/issues/334)) ([762227a](https://github.com/ecmwf/anemoi-core/commit/762227a5a25843dd4531eef1a9cbe86516eaffcd))
* **training, models:** Update interpolator to work with new features ([#322](https://github.com/ecmwf/anemoi-core/issues/322)) ([cfdc99f](https://github.com/ecmwf/anemoi-core/commit/cfdc99f984f0038b16cb96d73d02a25284af717e))

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

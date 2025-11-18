# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please add your functional changes to the appropriate section in the PR.
Keep it human-readable, your future self will thank you!

## [0.7.0](https://github.com/ecmwf/anemoi-core/compare/training-0.6.7...training-0.7.0) (2025-11-17)


### ⚠ BREAKING CHANGES

* **training:** remove support for EDA ([#651](https://github.com/ecmwf/anemoi-core/issues/651))

### Features

* Callbacks for GraphEnsForecaster. ([#449](https://github.com/ecmwf/anemoi-core/issues/449)) ([39c2bfc](https://github.com/ecmwf/anemoi-core/commit/39c2bfc7d0c840d92516b139cc5f497bbe1296a0))
* Mlflow azure ([#646](https://github.com/ecmwf/anemoi-core/issues/646)) ([27bd3dd](https://github.com/ecmwf/anemoi-core/commit/27bd3dd8648bac7ead5ec809b874faa962bddd05))
* **training:** Remove support for EDA ([#651](https://github.com/ecmwf/anemoi-core/issues/651)) ([921e108](https://github.com/ecmwf/anemoi-core/commit/921e108b8467f1bcd4e69516927efee2f58f9e33))


### Bug Fixes

* Anemoi-datasets import ([#626](https://github.com/ecmwf/anemoi-core/issues/626)) ([65c8901](https://github.com/ecmwf/anemoi-core/commit/65c89017fbc38cfe84ce61f4cd6f5aeb59b91224))
* Bug for mlflow offline logging ([#675](https://github.com/ecmwf/anemoi-core/issues/675)) ([fdce0f6](https://github.com/ecmwf/anemoi-core/commit/fdce0f6520e67e6d9039ed3417ff2fc0d76b002a))
* Bug in sample plots. ([#632](https://github.com/ecmwf/anemoi-core/issues/632)) ([9e024f3](https://github.com/ecmwf/anemoi-core/commit/9e024f37510780c12980d280a8247553b6e18225))
* Incorrect test for variable mask scaler ([#649](https://github.com/ecmwf/anemoi-core/issues/649)) ([d0f775e](https://github.com/ecmwf/anemoi-core/commit/d0f775e1054ead2c538d0e2391ef71f995123989))
* Integration tests and drop missed reference for  profiler ([#630](https://github.com/ecmwf/anemoi-core/issues/630)) ([f352e17](https://github.com/ecmwf/anemoi-core/commit/f352e17eca68c0cd797fdb5bf5fcf9bd4a2b505d))
* **training:** Provide more informative error when user specifies inexistent node attribute  ([#663](https://github.com/ecmwf/anemoi-core/issues/663)) ([dde3cb6](https://github.com/ecmwf/anemoi-core/commit/dde3cb6726a078115a086ba13bb846313fe29e41))
* Update readmes ([#655](https://github.com/ecmwf/anemoi-core/issues/655)) ([a58aa64](https://github.com/ecmwf/anemoi-core/commit/a58aa640212cdc6d5de6e38a6d11725d32d662b5))

## [0.6.7](https://github.com/ecmwf/anemoi-core/compare/training-0.6.6...training-0.6.7) (2025-10-20)


### Features

* **models:** Point-mlp proccesor ([#367](https://github.com/ecmwf/anemoi-core/issues/367)) ([ee2a067](https://github.com/ecmwf/anemoi-core/commit/ee2a067689b6efe18acb13ceac8ed0be512633d5))


### Bug Fixes

* CPU integration test - update AICON test to use anemoi-utils data download fixtures ([#608](https://github.com/ecmwf/anemoi-core/issues/608)) ([4c53c95](https://github.com/ecmwf/anemoi-core/commit/4c53c95e052afdbc89d03bea9c871b3e09046007))
* Refactored paths to import from package directly ([#601](https://github.com/ecmwf/anemoi-core/issues/601)) ([f28bfd4](https://github.com/ecmwf/anemoi-core/commit/f28bfd4c453dd0cbe0d65e5a7dafdbf531e9db44))
* **training:** Add default empty forcing list to DataSchema ([#602](https://github.com/ecmwf/anemoi-core/issues/602)) ([18a609b](https://github.com/ecmwf/anemoi-core/commit/18a609b5a5f59bfcd65c68ddb2177105317b6a18))
* **training:** Nan mask weights gather for plots ([#537](https://github.com/ecmwf/anemoi-core/issues/537)) ([a293b99](https://github.com/ecmwf/anemoi-core/commit/a293b9963000eb8f31433f0327809fcf99e0a447))


### Documentation

* **training:** More explicit comment about order of preprocessors in config file ([#613](https://github.com/ecmwf/anemoi-core/issues/613)) ([bbd152d](https://github.com/ecmwf/anemoi-core/commit/bbd152d434c9388a9dfc9100c7608255d932f79a))

## [0.6.6](https://github.com/ecmwf/anemoi-core/compare/training-0.6.5...training-0.6.6) (2025-10-09)


### Features

* Log variable scaling in mlflow ([#327](https://github.com/ecmwf/anemoi-core/issues/327)) ([1c4a966](https://github.com/ecmwf/anemoi-core/commit/1c4a96663c0b363fd81364d09dc7cd2f558acd7e))
* **mlflow login:** Support for multiple servers ([#573](https://github.com/ecmwf/anemoi-core/issues/573)) ([7b23100](https://github.com/ecmwf/anemoi-core/commit/7b231000f054b27547335c2c8448329fa1d2354e))
* Target indices ([#426](https://github.com/ecmwf/anemoi-core/issues/426)) ([d8db2a6](https://github.com/ecmwf/anemoi-core/commit/d8db2a6fc192bc49107df6c137ce4f56866ae4d4))


### Bug Fixes

* Allow int values in variable group schema ([#571](https://github.com/ecmwf/anemoi-core/issues/571)) ([a9ccc84](https://github.com/ecmwf/anemoi-core/commit/a9ccc84bc8d4f80a4d0e21829a09040528334a66))
* Interactive mode ([#578](https://github.com/ecmwf/anemoi-core/issues/578)) ([42c0194](https://github.com/ecmwf/anemoi-core/commit/42c0194662cfbcc6d5ae09a2a590848d7169753f))
* Make batch normalization explicit. ([#448](https://github.com/ecmwf/anemoi-core/issues/448)) ([6108f1d](https://github.com/ecmwf/anemoi-core/commit/6108f1d29fd78fc16549e28e59c18c0c50593043))
* Make TimeLimit callback compliant with multiple nodes  ([#564](https://github.com/ecmwf/anemoi-core/issues/564)) ([c94164d](https://github.com/ecmwf/anemoi-core/commit/c94164d28e761e1eb52ff16dfec73a3ad61976d0))
* Mask dimension number change ([#526](https://github.com/ecmwf/anemoi-core/issues/526)) ([e049e83](https://github.com/ecmwf/anemoi-core/commit/e049e839e3a1a1abf293ae5b7d8b2e9d7036efae))
* Remove batch_idx when useless ([#528](https://github.com/ecmwf/anemoi-core/issues/528)) ([2b6c205](https://github.com/ecmwf/anemoi-core/commit/2b6c2055c0155b6cf5e58355f0358245dafd7705))
* **training,schemas:** Pydantic model_validator in training/schemas/base_schema.py ([#562](https://github.com/ecmwf/anemoi-core/issues/562)) ([b64735d](https://github.com/ecmwf/anemoi-core/commit/b64735d6f895952c7976f39f61c2b653569fef4f))
* **training:** Add default diagnostics to empty list ([#597](https://github.com/ecmwf/anemoi-core/issues/597)) ([e243f95](https://github.com/ecmwf/anemoi-core/commit/e243f950822b36e9e706fafcdea9d753c1fdbfb6))
* **training:** Correct Imputer Doc ([#576](https://github.com/ecmwf/anemoi-core/issues/576)) ([e218652](https://github.com/ecmwf/anemoi-core/commit/e218652e3eb9e0881d942cc00c75c0b47f8f7b73))

## [0.6.5](https://github.com/ecmwf/anemoi-core/compare/training-0.6.4...training-0.6.5) (2025-09-09)


### Features

* Flash attention v3 ([#479](https://github.com/ecmwf/anemoi-core/issues/479)) ([00f52df](https://github.com/ecmwf/anemoi-core/commit/00f52df292f8fb8dc0a865f6d288fa151c630a2c))


### Bug Fixes

* Fix dry run check to use current run object ([#515](https://github.com/ecmwf/anemoi-core/issues/515)) ([02b0aee](https://github.com/ecmwf/anemoi-core/commit/02b0aee4a7b0916503cc4eaeb6555cee0d73c682))
* One-variable model ([#478](https://github.com/ecmwf/anemoi-core/issues/478)) ([36addea](https://github.com/ecmwf/anemoi-core/commit/36addea993a0bc19dbe1e7d1c21c95f18975d42a))
* Outdated docs for loss functions ([#525](https://github.com/ecmwf/anemoi-core/issues/525)) ([cc43519](https://github.com/ecmwf/anemoi-core/commit/cc43519df7bb278d2fd87b4163d63fd658fe9484))
* Remove gnn ens config ([#534](https://github.com/ecmwf/anemoi-core/issues/534)) ([d5eecd2](https://github.com/ecmwf/anemoi-core/commit/d5eecd2631bf4000f85cfe5fc8a54ea5506263f5))
* Remove useless code ([#527](https://github.com/ecmwf/anemoi-core/issues/527)) ([d965d06](https://github.com/ecmwf/anemoi-core/commit/d965d067b561c9d0632aff639e7f1a46ddad050e))
* Test dependencies ([#524](https://github.com/ecmwf/anemoi-core/issues/524)) ([3ac7d4f](https://github.com/ecmwf/anemoi-core/commit/3ac7d4fbc35e0ef0f54566454e235aeaf7f6da67))

## [0.6.4](https://github.com/ecmwf/anemoi-core/compare/training-0.6.3...training-0.6.4) (2025-09-02)


### Features

* Checkpoint migrations ([#386](https://github.com/ecmwf/anemoi-core/issues/386)) ([028f6eb](https://github.com/ecmwf/anemoi-core/commit/028f6eb6426c5ada0c5d95e6492accc99083b46f))
* Extend restart checkpoint to consider graphtransformer ([#497](https://github.com/ecmwf/anemoi-core/issues/497)) ([b8fc40e](https://github.com/ecmwf/anemoi-core/commit/b8fc40ec9d559ffab9f17af0d029444b884daf56))


### Bug Fixes

* **training:** LAM sharding ([#481](https://github.com/ecmwf/anemoi-core/issues/481)) ([e513c19](https://github.com/ecmwf/anemoi-core/commit/e513c19a0f80e1f0850627fe51eab992cdd6fa8d))

## [0.6.3](https://github.com/ecmwf/anemoi-core/compare/training-0.6.2...training-0.6.3) (2025-08-22)


### Bug Fixes

* GraphInterpolator Lightning Module - add rollout member for compa… ([#496](https://github.com/ecmwf/anemoi-core/issues/496)) ([40aa71c](https://github.com/ecmwf/anemoi-core/commit/40aa71c8eb097442ce33e7f0ff82e81b0947716e))

## [0.6.2](https://github.com/ecmwf/anemoi-core/compare/training-0.6.1...training-0.6.2) (2025-08-20)


### Features

* Diffusion training ([#401](https://github.com/ecmwf/anemoi-core/issues/401)) ([f35ad67](https://github.com/ecmwf/anemoi-core/commit/f35ad673f680ba64acb5c6770a3114262b3b97fd))


### Bug Fixes

* Mlflow sync cleaning ([#460](https://github.com/ecmwf/anemoi-core/issues/460)) ([50ef49b](https://github.com/ecmwf/anemoi-core/commit/50ef49ba6b5dc09186ae513d908a57e985daae2a))
* **training:** Callbacks when using rollout  ([#490](https://github.com/ecmwf/anemoi-core/issues/490)) ([7c9442a](https://github.com/ecmwf/anemoi-core/commit/7c9442aeb8fdfa8378e531b2ea9bf051fca1cfd0))
* **training:** Sharding nan_loss_mask scaler ([#491](https://github.com/ecmwf/anemoi-core/issues/491)) ([9299991](https://github.com/ecmwf/anemoi-core/commit/92999916b7a81a05b364aa1d4ca3d53e3ed35dbb))
* Update variable groups in lam, ensemble and stretched config files ([#466](https://github.com/ecmwf/anemoi-core/issues/466)) ([8db81f0](https://github.com/ecmwf/anemoi-core/commit/8db81f068e415a913014cdb01b68d048368fd6ff))


### Performance Improvements

* Avoid heavy imports through MlflowSchema ([#482](https://github.com/ecmwf/anemoi-core/issues/482)) ([8e97af5](https://github.com/ecmwf/anemoi-core/commit/8e97af5ddb7fa2c025f101075d3bfddfa4126690))


### Documentation

* Explain inference overrides ([#487](https://github.com/ecmwf/anemoi-core/issues/487)) ([d1996f9](https://github.com/ecmwf/anemoi-core/commit/d1996f9ec4be78d921feba775c2aeb14c52f102c))
* Update diffusion documentation ([#486](https://github.com/ecmwf/anemoi-core/issues/486)) ([ff8f9f2](https://github.com/ecmwf/anemoi-core/commit/ff8f9f2d07985312c9196150ed0c1753c1f78703))

## [0.6.1](https://github.com/ecmwf/anemoi-core/compare/training-0.6.0...training-0.6.1) (2025-08-08)


### Features

* Add extensible checkpoint loading system                                                                                     │ ([ebc55fc](https://github.com/ecmwf/anemoi-core/commit/ebc55fc9564d03d8e090bafab2fdd93042daa6cf))
* Extend DelayedScalers into arbitary UpdatingScalers [#371](https://github.com/ecmwf/anemoi-core/issues/371)  ([b9d7726](https://github.com/ecmwf/anemoi-core/commit/b9d772659679b1d1744c9be6a6602673eb9e6969))
* **models:** Nan locations in imputer calculated on the fly [#378](https://github.com/ecmwf/anemoi-core/issues/378) ([b9d7726](https://github.com/ecmwf/anemoi-core/commit/b9d772659679b1d1744c9be6a6602673eb9e6969))
* **models:** Postprocessor for nans in diagnostic fields ([#461](https://github.com/ecmwf/anemoi-core/issues/461)) ([a7ff22e](https://github.com/ecmwf/anemoi-core/commit/a7ff22e44b956635bcd3e91b9d780aa041a617d3))


### Bug Fixes

* Improve device movement of scalers [#390](https://github.com/ecmwf/anemoi-core/issues/390) ([b9d7726](https://github.com/ecmwf/anemoi-core/commit/b9d772659679b1d1744c9be6a6602673eb9e6969))
* Introducing SLURMID based cuda device selection to fix cuda busy devices error ([#431](https://github.com/ecmwf/anemoi-core/issues/431)) ([c6d2844](https://github.com/ecmwf/anemoi-core/commit/c6d284421ffe4a876e5caca5f73d86faa9ff6e3b))

## [0.6.0](https://github.com/ecmwf/anemoi-core/compare/training-0.5.1...training-0.6.0) (2025-08-01)


### ⚠ BREAKING CHANGES

* for schemas of data processors ([#433](https://github.com/ecmwf/anemoi-core/issues/433))
* BaseGraphModule and tasks introduced in anemoi-core ([#399](https://github.com/ecmwf/anemoi-core/issues/399))

### Features

* Add metadata back to pl checkpoint. ([#303](https://github.com/ecmwf/anemoi-core/issues/303)) ([0193b28](https://github.com/ecmwf/anemoi-core/commit/0193b280230d36acf94bcd1a1c7f587a7bf1abb7))
* BaseGraphModule and tasks introduced in anemoi-core ([#399](https://github.com/ecmwf/anemoi-core/issues/399)) ([f8ab962](https://github.com/ecmwf/anemoi-core/commit/f8ab9620f2ea7df90561911f0abf01105d6c923b))
* **deps:** Use mlflow-skinny instead of mlflow ([#418](https://github.com/ecmwf/anemoi-core/issues/418)) ([6a8beb3](https://github.com/ecmwf/anemoi-core/commit/6a8beb390aec888aa251983a730715e7be41bc3b))
* Log FTT2 loss + Fourier Correlation loss  ([#148](https://github.com/ecmwf/anemoi-core/issues/148)) ([345b0ab](https://github.com/ecmwf/anemoi-core/commit/345b0ab9a03c80e3944dc6e63a2e8a38123f18a3))
* **model:** Postprocessors for leaky boundings ([#315](https://github.com/ecmwf/anemoi-core/issues/315)) ([b54562b](https://github.com/ecmwf/anemoi-core/commit/b54562b83b4b5620891b28827964f9c554ee0615))
* **models:** Checkpointed Mapper Chunking ([#406](https://github.com/ecmwf/anemoi-core/issues/406)) ([8577772](https://github.com/ecmwf/anemoi-core/commit/8577772927a08d62db74159e1023f5db1dc39438))
* **models:** Mapper edge sharding ([#366](https://github.com/ecmwf/anemoi-core/issues/366)) ([326751d](https://github.com/ecmwf/anemoi-core/commit/326751d25f9bc299f3e19c795d9065a60a6af3d9))
* Variable filtering ([#208](https://github.com/ecmwf/anemoi-core/issues/208)) ([fba5e47](https://github.com/ecmwf/anemoi-core/commit/fba5e4773471b7cc9bee839251c20c6dd407ca05))


### Bug Fixes

* Dropping 3.9 ([#436](https://github.com/ecmwf/anemoi-core/issues/436)) ([f6c0214](https://github.com/ecmwf/anemoi-core/commit/f6c0214ad09d217930956b7eddaf0c8b35a32185))
* For schemas of data processors ([#433](https://github.com/ecmwf/anemoi-core/issues/433)) ([539939b](https://github.com/ecmwf/anemoi-core/commit/539939be4c4392afcf8ccd73b8de7c44e4b32847))
* Mlflow hp params limit ([#424](https://github.com/ecmwf/anemoi-core/issues/424)) ([138bc3a](https://github.com/ecmwf/anemoi-core/commit/138bc3af2a922973828aa40dadd765d24e3a076f))
* Mlflowlogger duplicated key ([#414](https://github.com/ecmwf/anemoi-core/issues/414)) ([cb64a1c](https://github.com/ecmwf/anemoi-core/commit/cb64a1c7fc66bd142401669cc1bcf10f0625d096))
* **models,traininig:** Hierarchical model + integration test ([#400](https://github.com/ecmwf/anemoi-core/issues/400)) ([71dfd89](https://github.com/ecmwf/anemoi-core/commit/71dfd89d4326d5e59c8ff8fef339b500110ded42))
* **models:** Add removed sharded_input_key in PR400 ([#425](https://github.com/ecmwf/anemoi-core/issues/425)) ([089fe6f](https://github.com/ecmwf/anemoi-core/commit/089fe6f01c8936887e49f353a17ea1f95b884888))
* New checkpoint ([#445](https://github.com/ecmwf/anemoi-core/issues/445)) ([a25df93](https://github.com/ecmwf/anemoi-core/commit/a25df93800077caa2eedad2e8e4476c5257a06b4))
* Plotting error when precip related params are not diagnostic  ([#369](https://github.com/ecmwf/anemoi-core/issues/369)) ([010cfa3](https://github.com/ecmwf/anemoi-core/commit/010cfa3f131ed4378236e5864b6f4d3922028b46))
* **training:** Address issues with [#208](https://github.com/ecmwf/anemoi-core/issues/208) ([#417](https://github.com/ecmwf/anemoi-core/issues/417)) ([665f462](https://github.com/ecmwf/anemoi-core/commit/665f4629649d2852449059ea2cc7af9bad7e24da))
* **training:** Scaler memory usage ([#391](https://github.com/ecmwf/anemoi-core/issues/391)) ([a9d30e1](https://github.com/ecmwf/anemoi-core/commit/a9d30e16f796df650b9a1b298b0b0501f1cf1577))
* Update import mflow utils unit tests ([#427](https://github.com/ecmwf/anemoi-core/issues/427)) ([70ecdd9](https://github.com/ecmwf/anemoi-core/commit/70ecdd9e46f5c9f46dc644cbb8d179392eab6910))
* Update level retrieval logic ([#405](https://github.com/ecmwf/anemoi-core/issues/405)) ([f393bc3](https://github.com/ecmwf/anemoi-core/commit/f393bc3d133596e98daa27c302ad51cf1bac5d41))
* Use transforms: Variable for ExtractVariableGroupAndLevel ([#321](https://github.com/ecmwf/anemoi-core/issues/321)) ([7649f4f](https://github.com/ecmwf/anemoi-core/commit/7649f4fb283ca6d26f79b7e07046680fcb10fd23))
* Warm restart ([#443](https://github.com/ecmwf/anemoi-core/issues/443)) ([ff96236](https://github.com/ecmwf/anemoi-core/commit/ff962363b28dda50979a031af19384a37415af6e))


### Documentation

* **graphs:** Documenting some missing features ([#423](https://github.com/ecmwf/anemoi-core/issues/423)) ([8addbd8](https://github.com/ecmwf/anemoi-core/commit/8addbd8655ffa71df786d196f23c9615cffd7f2e))

## [0.5.1](https://github.com/ecmwf/anemoi-core/compare/training-0.5.0...training-0.5.1) (2025-06-17)


### Features

* **models,training:** Shard everything ([#121](https://github.com/ecmwf/anemoi-core/issues/121)) ([06dde94](https://github.com/ecmwf/anemoi-core/commit/06dde94219119746215b767b846542ee31bbff63))


### Bug Fixes

* Config validator checks for conflict with bounding and graphtran… ([#352](https://github.com/ecmwf/anemoi-core/issues/352)) ([f97313e](https://github.com/ecmwf/anemoi-core/commit/f97313e608f0ce2eebd46bad8e5300edebe6eec1))
* **training, losses:** NaN handling for crps ([#358](https://github.com/ecmwf/anemoi-core/issues/358)) ([daf9ddd](https://github.com/ecmwf/anemoi-core/commit/daf9dddda8ff8755b59f0559666d6539d525ba3b))

## [0.5.0](https://github.com/ecmwf/anemoi-core/compare/training-0.4.0...training-0.5.0) (2025-06-05)


### ⚠ BREAKING CHANGES

* **models,training:** Remove multimapper ([#268](https://github.com/ecmwf/anemoi-core/issues/268))
* generalise activation function ([#163](https://github.com/ecmwf/anemoi-core/issues/163))
* Rework Loss Scalings to provide better modularity ([#52](https://github.com/ecmwf/anemoi-core/issues/52))

### Features

* Generalise activation function ([#163](https://github.com/ecmwf/anemoi-core/issues/163)) ([98c4d06](https://github.com/ecmwf/anemoi-core/commit/98c4d06dfb5b79f605fab885c67a8a4cd6d35384))
* **graphs:** Add scale_resolutions arg ([#188](https://github.com/ecmwf/anemoi-core/issues/188)) ([0ea0642](https://github.com/ecmwf/anemoi-core/commit/0ea06423e4979084b3afe70c30e43bb5dc5f88d5))
* **models,training:** Remove multimapper ([#268](https://github.com/ecmwf/anemoi-core/issues/268)) ([0e8bb99](https://github.com/ecmwf/anemoi-core/commit/0e8bb998176bea2d653ca40772e4e6e1578551f7))
* Transformer mapper ([#179](https://github.com/ecmwf/anemoi-core/issues/179)) ([2cea7db](https://github.com/ecmwf/anemoi-core/commit/2cea7db51d5c5ef63bb4b9c266deb05fb2acf66f))


### Bug Fixes

* Adapt ensemble configs to work with loss refactor.  ([#300](https://github.com/ecmwf/anemoi-core/issues/300)) ([c210478](https://github.com/ecmwf/anemoi-core/commit/c210478d8e84aaa2e93256aa1b4ceda74e191c7d))
* Adapt predict_step in model interface to pass on arguments for model classes ([#281](https://github.com/ecmwf/anemoi-core/issues/281)) ([a5b2643](https://github.com/ecmwf/anemoi-core/commit/a5b26432bc7b78577cd1febd5091b059cc82805c))
* Change output format in validation step to fix issue in plotting ([#305](https://github.com/ecmwf/anemoi-core/issues/305)) ([cc941f3](https://github.com/ecmwf/anemoi-core/commit/cc941f3d7520e230e733654cb45e5c6473b64152))
* Clean params complement ([#291](https://github.com/ecmwf/anemoi-core/issues/291)) ([4e9ca31](https://github.com/ecmwf/anemoi-core/commit/4e9ca313f838f5eccb305d6bf3b9afd8426f095f))
* Dataset_order ([#334](https://github.com/ecmwf/anemoi-core/issues/334)) ([762227a](https://github.com/ecmwf/anemoi-core/commit/762227a5a25843dd4531eef1a9cbe86516eaffcd))
* **deps:** Bump `anemoi-models >= 0.8` ([#351](https://github.com/ecmwf/anemoi-core/issues/351)) ([1fbd525](https://github.com/ecmwf/anemoi-core/commit/1fbd5252ef1400eb4bbe5bfbbb171dad1d652c63))
* **models,training:** Remove unnecessary torch-geometric maximum version ([#326](https://github.com/ecmwf/anemoi-core/issues/326)) ([fe93ea8](https://github.com/ecmwf/anemoi-core/commit/fe93ea8feb379147a9f9e5c5358ea8144855dc77))
* Move scaler to device in scale ([#317](https://github.com/ecmwf/anemoi-core/issues/317)) ([1592d09](https://github.com/ecmwf/anemoi-core/commit/1592d09f0915dddba7dcfc1c11897e0cae5cb6d0))
* Remove activation entry from MLP noise block ([#340](https://github.com/ecmwf/anemoi-core/issues/340)) ([2d060f5](https://github.com/ecmwf/anemoi-core/commit/2d060f5e3382454b06c6369141942b8d6367fb4b))
* Rework Loss Scalings to provide better modularity ([#52](https://github.com/ecmwf/anemoi-core/issues/52)) ([162b906](https://github.com/ecmwf/anemoi-core/commit/162b9062882c321a4a265b0cf561be3f141ac97a))
* **training, models:** Update interpolator to work with new features ([#322](https://github.com/ecmwf/anemoi-core/issues/322)) ([cfdc99f](https://github.com/ecmwf/anemoi-core/commit/cfdc99f984f0038b16cb96d73d02a25284af717e))
* **training:** Bump anemoi-graphs version to 0.5.2 ([#276](https://github.com/ecmwf/anemoi-core/issues/276)) ([9b8ec13](https://github.com/ecmwf/anemoi-core/commit/9b8ec13b56f0a18ac887c754d5d95a7953b2625d))
* **training:** Explicit Batch invariance ([#318](https://github.com/ecmwf/anemoi-core/issues/318)) ([45f6e15](https://github.com/ecmwf/anemoi-core/commit/45f6e15183dbf61ad259b7ca5ed35d94395d237c))
* **training:** No logging variable metadata to mlflow ([#304](https://github.com/ecmwf/anemoi-core/issues/304)) ([4f7c2c9](https://github.com/ecmwf/anemoi-core/commit/4f7c2c9c75ae6e67a148699a317528cd7651bf46))
* Update defaults for mlflow logging ([#333](https://github.com/ecmwf/anemoi-core/issues/333)) ([5560581](https://github.com/ecmwf/anemoi-core/commit/556058167439ae29e6c9559d67f2e01df8466158))


### Documentation

* Downloading era5 o96 dataset ([#307](https://github.com/ecmwf/anemoi-core/issues/307)) ([b9e2d24](https://github.com/ecmwf/anemoi-core/commit/b9e2d24702132e43f160a428ccad88b7584fcabe))
* Fix minor mistakes in CRPS user guide. ([#264](https://github.com/ecmwf/anemoi-core/issues/264)) ([52d9a2e](https://github.com/ecmwf/anemoi-core/commit/52d9a2e5e0cef82b65b604ed2dc55016a48aeaa2))

## [0.4.0](https://github.com/ecmwf/anemoi-core/compare/training-0.3.3...training-0.4.0) (2025-04-16)


### ⚠ BREAKING CHANGES

* **models,training:** temporal interpolation ([#153](https://github.com/ecmwf/anemoi-core/issues/153))
* **config:** Improved configuration and data structures ([#79](https://github.com/ecmwf/anemoi-core/issues/79))

### Features

* Add a CLI to dump the Hydra configuration files into a single YAML file. ([#137](https://github.com/ecmwf/anemoi-core/issues/137)) ([ef1e76e](https://github.com/ecmwf/anemoi-core/commit/ef1e76e2e15bb412adb184e1b33e003590c72e8a))
* Add EarlyStopping Wrapper ([#130](https://github.com/ecmwf/anemoi-core/issues/130)) ([21d06be](https://github.com/ecmwf/anemoi-core/commit/21d06be94b5ea09889777038da749ee167cf3f3d))
* Add the possibility to train a model with a dry MLflow run ID ([#164](https://github.com/ecmwf/anemoi-core/issues/164)) ([9849d21](https://github.com/ecmwf/anemoi-core/commit/9849d211fcb03fcbb76fcdd3ef6e97e86f1e69b4))
* **config:** Improved configuration and data structures ([#79](https://github.com/ecmwf/anemoi-core/issues/79)) ([1f7812b](https://github.com/ecmwf/anemoi-core/commit/1f7812b559b51d842852df29ace7dda6d0f66ef2))
* edge post-processor ([#199](https://github.com/ecmwf/anemoi-core/issues/199)) ([1450de7](https://github.com/ecmwf/anemoi-core/commit/1450de739be9988cdb23fbdb23a0463859066e7c))
* **graphs:** migrate edge builders to torch-cluster ([#56](https://github.com/ecmwf/anemoi-core/issues/56)) ([f67da66](https://github.com/ecmwf/anemoi-core/commit/f67da664c18762e4c8a8cf0af9d4e97ec7315454))
* Kcrps  ([#182](https://github.com/ecmwf/anemoi-core/issues/182)) ([8bbe898](https://github.com/ecmwf/anemoi-core/commit/8bbe89839e2eff3fcbc35613eb92920d4afc3276))
* make colormaps configurable for groups of variable ([#124](https://github.com/ecmwf/anemoi-core/issues/124)) ([83d72e1](https://github.com/ecmwf/anemoi-core/commit/83d72e17b5c8017a1d9a47f75e0848e0bbb080a3))
* **mlflow:** Make direct endpoint calls robust ([#186](https://github.com/ecmwf/anemoi-core/issues/186)) ([77bd890](https://github.com/ecmwf/anemoi-core/commit/77bd890004ea591861ec1efe6732ea1599008985))
* **models,training:** temporal interpolation ([#153](https://github.com/ecmwf/anemoi-core/issues/153)) ([ea644ce](https://github.com/ecmwf/anemoi-core/commit/ea644ce1c9aef902333d9cbb30bcde0a3746fbcc))
* **models:** adding leaky boundings ([#256](https://github.com/ecmwf/anemoi-core/issues/256)) ([426e860](https://github.com/ecmwf/anemoi-core/commit/426e86048d6c0a03750fb0e205890841c27c8148))
* **training:** Add initial TimeLimit callback ([#115](https://github.com/ecmwf/anemoi-core/issues/115)) ([41ff583](https://github.com/ecmwf/anemoi-core/commit/41ff5830dc0ba08aaa86e052a1a0aac8c4498c7a))


### Bug Fixes

* --longtests not available ([#200](https://github.com/ecmwf/anemoi-core/issues/200)) ([9dfec0a](https://github.com/ecmwf/anemoi-core/commit/9dfec0a3bd2043e646cc49b5302fcc4d669e4a41))
* Checkpoint path check for multiple tasks/GPUs training ([#242](https://github.com/ecmwf/anemoi-core/issues/242)) ([449f8bd](https://github.com/ecmwf/anemoi-core/commit/449f8bd6044043fc46160b602762785b475cd1ce))
* **configs,schemas:** hierarchical schemas ([#221](https://github.com/ecmwf/anemoi-core/issues/221)) ([2d4a54d](https://github.com/ecmwf/anemoi-core/commit/2d4a54d3c6ed2d41fd6cbf2ef3ac57b9efb2f968))
* correct config comment regarding config_validation flag ([#245](https://github.com/ecmwf/anemoi-core/issues/245)) ([d02e0bb](https://github.com/ecmwf/anemoi-core/commit/d02e0bb2a9bf1700de90c8d862f80a510f03eceb))
* dataset schema too defined too strictly ([#143](https://github.com/ecmwf/anemoi-core/issues/143)) ([4792ee1](https://github.com/ecmwf/anemoi-core/commit/4792ee145a3bda05f934180485ddb7b6e3bb25d4))
* datashader new release test ([#104](https://github.com/ecmwf/anemoi-core/issues/104)) ([e9c0701](https://github.com/ecmwf/anemoi-core/commit/e9c0701bce9e5dfff7184fc6210092b82b9386a6))
* dry run forking ([#260](https://github.com/ecmwf/anemoi-core/issues/260)) ([a32cccd](https://github.com/ecmwf/anemoi-core/commit/a32cccd3179b049979b500ad878480ae5e46d050))
* **graphs,training:** skip hidden attributes ([#176](https://github.com/ecmwf/anemoi-core/issues/176)) ([468c45a](https://github.com/ecmwf/anemoi-core/commit/468c45a8a8ccee6f4a11c4ddf3fe4a7220235d08))
* lam rollout ([#213](https://github.com/ecmwf/anemoi-core/issues/213)) ([6f78387](https://github.com/ecmwf/anemoi-core/commit/6f78387da08237b7a63b84575e8cb6c19b254493))
* pydantic schemas move ([#228](https://github.com/ecmwf/anemoi-core/issues/228)) ([6bca9bc](https://github.com/ecmwf/anemoi-core/commit/6bca9bc66ff54ac294d97793b8cebed1cd1bb8a4))
* remove pydantic schemas from checkpoint pickled objects ([#237](https://github.com/ecmwf/anemoi-core/issues/237)) ([ecb945a](https://github.com/ecmwf/anemoi-core/commit/ecb945ad7df226d16ed42b10123372741c5748a5))
* Rename model_run_ids as trajectory_ids ([#216](https://github.com/ecmwf/anemoi-core/issues/216)) ([e5e942d](https://github.com/ecmwf/anemoi-core/commit/e5e942d001640b73a29d5d6156364f3bd99fc1e5))
* **training,configs:** update enc-dec config ([#125](https://github.com/ecmwf/anemoi-core/issues/125)) ([beb8c69](https://github.com/ecmwf/anemoi-core/commit/beb8c69e7c58ae04124030ddf3b080b9c4e6b7e1))
* **training,schema:** error when using ReweightedGraphNodeAttribute ([#169](https://github.com/ecmwf/anemoi-core/issues/169)) ([a6313ef](https://github.com/ecmwf/anemoi-core/commit/a6313ef37ec8e08fe0fbcc58b99fb098774b0229))
* **training:** added weights_only=False flag to torch.load to avoid crashes in torch 2.6 ([#205](https://github.com/ecmwf/anemoi-core/issues/205)) ([02b5117](https://github.com/ecmwf/anemoi-core/commit/02b5117fb2f455fea943ffccc76117d2e0d514f8))
* **training:** Rework Combined Loss ([#103](https://github.com/ecmwf/anemoi-core/issues/103)) ([b63f1aa](https://github.com/ecmwf/anemoi-core/commit/b63f1aa4e6f154898d84310cb03cf244b322efa4))
* update stretched.yaml config to be consistent with schema ([#195](https://github.com/ecmwf/anemoi-core/issues/195)) ([21255ac](https://github.com/ecmwf/anemoi-core/commit/21255ac8e535889ad47ccada2bf4a27047c27021))
* update to null and bump versions ([#263](https://github.com/ecmwf/anemoi-core/issues/263)) ([b4507fb](https://github.com/ecmwf/anemoi-core/commit/b4507fbf926a01f0e042cb61fc35b3901243ddc0))
* updates to schemas to correct minor pydantic issues ([#144](https://github.com/ecmwf/anemoi-core/issues/144)) ([4e51c17](https://github.com/ecmwf/anemoi-core/commit/4e51c178c72c6853cf223e81609e0a91b183a277))


### Documentation

* Add diagnostics to schema docs ([#159](https://github.com/ecmwf/anemoi-core/issues/159)) ([42c5706](https://github.com/ecmwf/anemoi-core/commit/42c570625cfe445441f18e74827b8c1526ff1782))
* Add reference in training getting started guide for intersphinx ([#220](https://github.com/ecmwf/anemoi-core/issues/220)) ([f1d5e1f](https://github.com/ecmwf/anemoi-core/commit/f1d5e1f07c7a92e7152f0a392a046537bc718cab))
* Add subheadings to schema doc page ([#149](https://github.com/ecmwf/anemoi-core/issues/149)) ([d3c7de9](https://github.com/ecmwf/anemoi-core/commit/d3c7de905bced2dc9e75a92de4e9abf848936e62))
* fix documentation to refer to anemoi datasets instead of zarr datasets ([#154](https://github.com/ecmwf/anemoi-core/issues/154)) ([ad062b2](https://github.com/ecmwf/anemoi-core/commit/ad062b22cdd05354bc010eabbf8ffa806def081c))
* **models:** Docathon  ([#202](https://github.com/ecmwf/anemoi-core/issues/202)) ([5dba9d3](https://github.com/ecmwf/anemoi-core/commit/5dba9d34d65d4331dabd19355c7a31f7f1468fbf))
* **training:** add ADR into docs/ ([#168](https://github.com/ecmwf/anemoi-core/issues/168)) ([ee4da88](https://github.com/ecmwf/anemoi-core/commit/ee4da886c32f1edbc197215780bf84dcd323e172))
* **training:** Docathon ([#201](https://github.com/ecmwf/anemoi-core/issues/201)) ([e69430f](https://github.com/ecmwf/anemoi-core/commit/e69430f8c1ba8e7de50cd99f202e3f4876b806e0))
* Update docs for kcrps. ([#258](https://github.com/ecmwf/anemoi-core/issues/258)) ([79cbd1d](https://github.com/ecmwf/anemoi-core/commit/79cbd1d5e5f0f5aa82ce712bed474a6ad99f17e8))
* use new logo ([#140](https://github.com/ecmwf/anemoi-core/issues/140)) ([c269cea](https://github.com/ecmwf/anemoi-core/commit/c269cea3c84f2e35ef0a318e0cd1b769d285177c))

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

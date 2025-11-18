# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please add your functional changes to the appropriate section in the PR.
Keep it human-readable, your future self will thank you!

## [0.7.2](https://github.com/ecmwf/anemoi-core/compare/graphs-0.7.1...graphs-0.7.2) (2025-11-17)


### Features

* **graphs:** Add LimitedAreaMask for stretched hidden nodes ([#671](https://github.com/ecmwf/anemoi-core/issues/671)) ([f155f3c](https://github.com/ecmwf/anemoi-core/commit/f155f3c065ca9ebfe4b1ca713b6c7f59dc02d279))
* **graphs:** New edge attributes and faster graph cleaning ([#617](https://github.com/ecmwf/anemoi-core/issues/617)) ([8659de9](https://github.com/ecmwf/anemoi-core/commit/8659de9238f95c301df42989a62a46ca7f702276))
* New interactive graph visualization ([#609](https://github.com/ecmwf/anemoi-core/issues/609)) ([5b5ede4](https://github.com/ecmwf/anemoi-core/commit/5b5ede4afd5c4a1d4a7ddc11d0dcc4a4e614af1b))


### Bug Fixes

* **graphs,normalisation:** Add assert when dividing by 0 ([#676](https://github.com/ecmwf/anemoi-core/issues/676)) ([01b7034](https://github.com/ecmwf/anemoi-core/commit/01b7034b7dd2898a41d64071255fa27a71156be5))
* **graphs,schemas:** Missing type for mask_attr_name in schema ([#664](https://github.com/ecmwf/anemoi-core/issues/664)) ([f021017](https://github.com/ecmwf/anemoi-core/commit/f021017545ebac20b6bafb9e1bedbb312e7c7e29))
* **graphs,tests:** New test and fix anemoi-graphs tests with gpu ([#637](https://github.com/ecmwf/anemoi-core/issues/637)) ([ca1b542](https://github.com/ecmwf/anemoi-core/commit/ca1b54248c0efda74cc48544be1479e5984c1c13))
* **graphs:** Remove wrong argument from og.Figure ([#616](https://github.com/ecmwf/anemoi-core/issues/616)) ([abd37eb](https://github.com/ecmwf/anemoi-core/commit/abd37eba46dee06287788ae60405d7c019f189ef))
* **graphs:** Unit-range normalisation ([#665](https://github.com/ecmwf/anemoi-core/issues/665)) ([6de4778](https://github.com/ecmwf/anemoi-core/commit/6de4778c5abd02437ff4649d2f26ac16b99c37fc))
* Update readmes ([#655](https://github.com/ecmwf/anemoi-core/issues/655)) ([a58aa64](https://github.com/ecmwf/anemoi-core/commit/a58aa640212cdc6d5de6e38a6d11725d32d662b5))

## [0.7.1](https://github.com/ecmwf/anemoi-core/compare/graphs-0.7.0...graphs-0.7.1) (2025-10-20)


### Bug Fixes

* **graphs:** Skip expensive graph creation when target exists without overwrite ([#607](https://github.com/ecmwf/anemoi-core/issues/607)) ([5ce1c6a](https://github.com/ecmwf/anemoi-core/commit/5ce1c6a5948bda6bf3c194098f306b6b4c818ead))
* Set default value for scale_resolutions parameter ([#592](https://github.com/ecmwf/anemoi-core/issues/592)) ([3b9470d](https://github.com/ecmwf/anemoi-core/commit/3b9470d50e609837b3c12372df165e508ded4052))
* **training:** Allow for string in AnemoiDatasetsMask.get_grid_size  ([#604](https://github.com/ecmwf/anemoi-core/issues/604)) ([30cff57](https://github.com/ecmwf/anemoi-core/commit/30cff5712abbedd1cbf36a2a720867edd1f029f6))

## [0.7.0](https://github.com/ecmwf/anemoi-core/compare/graphs-0.6.6...graphs-0.7.0) (2025-10-09)


### ⚠ BREAKING CHANGES

* **graphs:** remove legacy code supporting deprecated features ([#579](https://github.com/ecmwf/anemoi-core/issues/579))

### Features

* **graphs:** Add NonzeroAnemoiDatasetVariable class ([#554](https://github.com/ecmwf/anemoi-core/issues/554)) ([da7a5b5](https://github.com/ecmwf/anemoi-core/commit/da7a5b5d3349228d7c7b590a4a6f3448fe3574d2))


### Bug Fixes

* **graphs:** Remove legacy code supporting deprecated features ([#579](https://github.com/ecmwf/anemoi-core/issues/579)) ([aec70b5](https://github.com/ecmwf/anemoi-core/commit/aec70b56e1fbded9cc8391a216b65d9c6e84d03b))

## [0.6.6](https://github.com/ecmwf/anemoi-core/compare/graphs-0.6.5...graphs-0.6.6) (2025-09-09)


### Features

* **graphs:** Support area positional argument in GraphInspector ([#415](https://github.com/ecmwf/anemoi-core/issues/415)) ([2ea988a](https://github.com/ecmwf/anemoi-core/commit/2ea988a308cdb83c3d8065823ae982b9c468fe20))


### Bug Fixes

* **graphs:** Move edge_indinces to cpu for post processors ([#529](https://github.com/ecmwf/anemoi-core/issues/529)) ([709687e](https://github.com/ecmwf/anemoi-core/commit/709687e706b5d451a4220807f06ac1ff9d86df47))

## [0.6.5](https://github.com/ecmwf/anemoi-core/compare/graphs-0.6.4...graphs-0.6.5) (2025-09-02)


### Bug Fixes

* **graph:** Update RestrictEdgeLength schema ([#471](https://github.com/ecmwf/anemoi-core/issues/471)) ([24a479a](https://github.com/ecmwf/anemoi-core/commit/24a479a16d7e307c12f71d3864cd0be902bb7f66))

## [0.6.4](https://github.com/ecmwf/anemoi-core/compare/graphs-0.6.3...graphs-0.6.4) (2025-08-08)


### Bug Fixes

* Introducing SLURMID based cuda device selection to fix cuda busy devices error ([#431](https://github.com/ecmwf/anemoi-core/issues/431)) ([c6d2844](https://github.com/ecmwf/anemoi-core/commit/c6d284421ffe4a876e5caca5f73d86faa9ff6e3b))

## [0.6.3](https://github.com/ecmwf/anemoi-core/compare/graphs-0.6.2...graphs-0.6.3) (2025-08-01)


### Features

* **graphs:** Add lat weighted attribute ([#223](https://github.com/ecmwf/anemoi-core/issues/223)) ([5dd32ca](https://github.com/ecmwf/anemoi-core/commit/5dd32ca1fa5a0447a524e5a3f651f087be171ef4))
* **graphs:** Support to export edges to npz ([#395](https://github.com/ecmwf/anemoi-core/issues/395)) ([e21738f](https://github.com/ecmwf/anemoi-core/commit/e21738f0029e799bcf1706a9b5844da0765d4484))


### Bug Fixes

* Dropping 3.9 ([#436](https://github.com/ecmwf/anemoi-core/issues/436)) ([f6c0214](https://github.com/ecmwf/anemoi-core/commit/f6c0214ad09d217930956b7eddaf0c8b35a32185))
* **graphs:** Revert PR [#379](https://github.com/ecmwf/anemoi-core/issues/379) ([#409](https://github.com/ecmwf/anemoi-core/issues/409)) ([d51219f](https://github.com/ecmwf/anemoi-core/commit/d51219f0ef598a149dc2c7901b055947785a26b2))
* **graphs:** Throw error instead of raising warning when graph exists.  ([#379](https://github.com/ecmwf/anemoi-core/issues/379)) ([6ec6c18](https://github.com/ecmwf/anemoi-core/commit/6ec6c18b67e797f1cf2037feabe0462d2058dd25))
* **graphs:** Undo masking when torch-cluster is installed ([#375](https://github.com/ecmwf/anemoi-core/issues/375)) ([9f75c06](https://github.com/ecmwf/anemoi-core/commit/9f75c06c1ff8a720f05341ad055ff345306df7a6))


### Documentation

* **graphs:** Documenting some missing features ([#423](https://github.com/ecmwf/anemoi-core/issues/423)) ([8addbd8](https://github.com/ecmwf/anemoi-core/commit/8addbd8655ffa71df786d196f23c9615cffd7f2e))

## [0.6.2](https://github.com/ecmwf/anemoi-core/compare/graphs-0.6.1...graphs-0.6.2) (2025-06-17)


### Features

* **graphs:** Add GridsMask node attribute builder ([#355](https://github.com/ecmwf/anemoi-core/issues/355)) ([c4db823](https://github.com/ecmwf/anemoi-core/commit/c4db8236374b245be17319caf3dc26911800da10))
* **graphs:** Build nodes from xarray compatible data ([#330](https://github.com/ecmwf/anemoi-core/issues/330)) ([3edaabb](https://github.com/ecmwf/anemoi-core/commit/3edaabb7e68a9ec89eccd5a16b258fbbf941b166))


### Bug Fixes

* **graphs:** Move libraries to optional dependencies ([#338](https://github.com/ecmwf/anemoi-core/issues/338)) ([db215ce](https://github.com/ecmwf/anemoi-core/commit/db215ce4ec0b1835e02cf1418c71292904153547))
* SphericalAreaWeights ([#363](https://github.com/ecmwf/anemoi-core/issues/363)) ([acca570](https://github.com/ecmwf/anemoi-core/commit/acca570ac33b53a4350d2492ceff05725f87ea0f))

## [0.6.1](https://github.com/ecmwf/anemoi-core/compare/graphs-0.6.0...graphs-0.6.1) (2025-06-05)


### Bug Fixes

* **graphs:** Fix device mismatch error ([#339](https://github.com/ecmwf/anemoi-core/issues/339)) ([8712c2a](https://github.com/ecmwf/anemoi-core/commit/8712c2abdd823541834b17bcdb92bd13bac101ee))

## [0.6.0](https://github.com/ecmwf/anemoi-core/compare/graphs-0.5.2...graphs-0.6.0) (2025-05-16)


### ⚠ BREAKING CHANGES

* Rework Loss Scalings to provide better modularity ([#52](https://github.com/ecmwf/anemoi-core/issues/52))

### Bug Fixes

* **graphs,directions:** fix error with direction computation ([#287](https://github.com/ecmwf/anemoi-core/issues/287)) ([55feded](https://github.com/ecmwf/anemoi-core/commit/55feded3b62b3906b1433a9e02988009c7f6d59e))
* Rework Loss Scalings to provide better modularity ([#52](https://github.com/ecmwf/anemoi-core/issues/52)) ([162b906](https://github.com/ecmwf/anemoi-core/commit/162b9062882c321a4a265b0cf561be3f141ac97a))

## [0.5.2](https://github.com/ecmwf/anemoi-core/compare/graphs-0.5.1...graphs-0.5.2) (2025-04-25)


### Features

* **graphs:** add scale_resolutions arg ([#188](https://github.com/ecmwf/anemoi-core/issues/188)) ([0ea0642](https://github.com/ecmwf/anemoi-core/commit/0ea06423e4979084b3afe70c30e43bb5dc5f88d5))
* **graphs:** support edge indices sorting ([#187](https://github.com/ecmwf/anemoi-core/issues/187)) ([1444083](https://github.com/ecmwf/anemoi-core/commit/1444083dbe1dc260918b5141d927e145c962b244))


### Bug Fixes

* **graphs:** rename zarr occurrences in anemoi-graphs ([#273](https://github.com/ecmwf/anemoi-core/issues/273)) ([d0bafe9](https://github.com/ecmwf/anemoi-core/commit/d0bafe91f1e09dc6a7efd35da9bc9102543c40e0))

## [0.5.1](https://github.com/ecmwf/anemoi-core/compare/graphs-0.5.0...graphs-0.5.1) (2025-04-16)


### Features

* edge post-processor ([#199](https://github.com/ecmwf/anemoi-core/issues/199)) ([1450de7](https://github.com/ecmwf/anemoi-core/commit/1450de739be9988cdb23fbdb23a0463859066e7c))


### Bug Fixes

* **graphs:** drop torch_geometric &lt; 2.5 dependency ([#207](https://github.com/ecmwf/anemoi-core/issues/207)) ([bf6c8af](https://github.com/ecmwf/anemoi-core/commit/bf6c8aff4b9289bb3d6195566c91aaa7b2d70f7a))
* **graphs:** load graphs to cpu during inspection ([#206](https://github.com/ecmwf/anemoi-core/issues/206)) ([bb82adf](https://github.com/ecmwf/anemoi-core/commit/bb82adf7a0e285cd0a6068e05f0079450a07d10d))
* **graphs:** torch_geometric version &lt; 2.5 ([#196](https://github.com/ecmwf/anemoi-core/issues/196)) ([843f944](https://github.com/ecmwf/anemoi-core/commit/843f9447aa919845c497f9f5c45997d99d30a4a9))
* pydantic schemas move ([#228](https://github.com/ecmwf/anemoi-core/issues/228)) ([6bca9bc](https://github.com/ecmwf/anemoi-core/commit/6bca9bc66ff54ac294d97793b8cebed1cd1bb8a4))


### Documentation

* **graphs:** update graph docs to template (docathon) ([#219](https://github.com/ecmwf/anemoi-core/issues/219)) ([ae4f1c5](https://github.com/ecmwf/anemoi-core/commit/ae4f1c5a52b8be16480ae7cfb97124f553b5ac07))
* **training:** Docathon ([#201](https://github.com/ecmwf/anemoi-core/issues/201)) ([e69430f](https://github.com/ecmwf/anemoi-core/commit/e69430f8c1ba8e7de50cd99f202e3f4876b806e0))

## [0.5.0](https://github.com/ecmwf/anemoi-core/compare/graphs-0.4.4...graphs-0.5.0) (2025-03-17)


### ⚠ BREAKING CHANGES

* **config:** Improved configuration and data structures ([#79](https://github.com/ecmwf/anemoi-core/issues/79))

### Features

* **config:** Improved configuration and data structures ([#79](https://github.com/ecmwf/anemoi-core/issues/79)) ([1f7812b](https://github.com/ecmwf/anemoi-core/commit/1f7812b559b51d842852df29ace7dda6d0f66ef2))
* **graphs:** migrate edge builders to torch-cluster ([#56](https://github.com/ecmwf/anemoi-core/issues/56)) ([f67da66](https://github.com/ecmwf/anemoi-core/commit/f67da664c18762e4c8a8cf0af9d4e97ec7315454))


### Bug Fixes

* **graphs:** make exception agnostic of zarr version ([#152](https://github.com/ecmwf/anemoi-core/issues/152)) ([f26adc9](https://github.com/ecmwf/anemoi-core/commit/f26adc969a0683711bc6a92a37e04e72d62ab57a))
* **graphs:** support torch v2.6 ([#122](https://github.com/ecmwf/anemoi-core/issues/122)) ([dfef377](https://github.com/ecmwf/anemoi-core/commit/dfef377a48ff093ec193ce77c2f3333b87131920))


### Documentation

* Add subheadings to schema doc page ([#149](https://github.com/ecmwf/anemoi-core/issues/149)) ([d3c7de9](https://github.com/ecmwf/anemoi-core/commit/d3c7de905bced2dc9e75a92de4e9abf848936e62))
* fix documentation to refer to anemoi datasets instead of zarr datasets ([#154](https://github.com/ecmwf/anemoi-core/issues/154)) ([ad062b2](https://github.com/ecmwf/anemoi-core/commit/ad062b22cdd05354bc010eabbf8ffa806def081c))
* use new logo ([#140](https://github.com/ecmwf/anemoi-core/issues/140)) ([c269cea](https://github.com/ecmwf/anemoi-core/commit/c269cea3c84f2e35ef0a318e0cd1b769d285177c))

## 0.4.4 (2025-02-04)

<!-- Release notes generated using configuration in .github/release.yml at main -->
Version fix.


**Full Changelog**: https://github.com/ecmwf/anemoi-core/compare/anemoi-graphs-0.4.3...anemoi-graphs-0.4.4

## 0.4.2.post546 (2025-02-04)

<!-- Release notes generated using configuration in .github/release.yml at main -->

## What's Changed
### Graphs
* feat(graphs,plots): expand support for multi-dimensional node attributes by @JPXKQX in https://github.com/ecmwf/anemoi-core/pull/48
* feat(graphs): New Edge Attribute: AttributeFromNode by @icedoom888 in https://github.com/ecmwf/anemoi-core/pull/62
* feat: support ReducedGaussianGridNodes by @JPXKQX in https://github.com/ecmwf/anemoi-core/pull/54
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

**Full Changelog**: https://github.com/ecmwf/anemoi-core/compare/anemoi-graphs-0.4.2...anemoi-graphs-0.4.3

## [Unreleased](https://github.com/ecmwf/anemoi-graphs/compare/0.4.2...HEAD)

### Added

- feat: Support for multi-dimensional node attributes in plots (#48)

## [0.4.2 - Optimisations and lat-lon](https://github.com/ecmwf/anemoi-graphs/compare/0.4.1...0.4.2) - 2024-12-19

### Added

- feat: Support for providing lon/lat coordinates from a text file (loaded with numpy loadtxt method) to build the graph `TextNodes` (#93)
- feat: Build 2D graphs with `Voronoi` in case `SphericalVoronoi` does not work well/is an overkill (LAM). Set `flat=true` in the nodes attributes to compute area weight using Voronoi with a qhull options preventing the empty region creation (#93) 
- feat: Support for defining nodes from lat& lon NumPy arrays (#98)
- feat: new transform functions to map from sin&cos values to latlon (#98)

### Changed

### Changed

- docs: Documentation structure (#84)
- fix: faster edge builder for tri icosahedron. (#92)

### Added

- feat: Support for multi-dimensional node attributes in plots (#86)

## [0.4.1 - ICON graphs, multiple edge builders and post processors](https://github.com/ecmwf/anemoi-graphs/compare/0.4.0...0.4.1) - 2024-11-26

### Added

- feat: Define node sets and edges based on an ICON icosahedral mesh (#53)
- feat: Add support for `post_processors` in the recipe. (#71)
- feat: Add `RemoveUnconnectedNodes` post processor to clean unconnected nodes in LAM. (#71)
- feat: Define node sets and edges based on an ICON icosahedral mesh (#53)
- feat: Support for multiple edge builders between two sets of nodes (#70)

### Changed

- fix: bug when computing area weights with scipy.Voronoi. (#79)

## [0.4.0 - LAM and stretched graphs](https://github.com/ecmwf/anemoi-graphs/compare/0.3.0...0.4.0) - 2024-11-08

### Added

- ci: hpc-config, CODEOWNERS (#49)
- feat: New node builder class, CutOutZarrDatasetNodes, to create nodes from 2 datasets. (#30)
- feat: New class, KNNAreaMaskBuilder, to specify Area of Interest (AOI) based on a set of nodes. (#30)
- feat: New node builder classes, LimitedAreaXXXXXNodes, to create nodes within an Area of Interest (AOI). (#30)
- feat: Expanded MultiScaleEdges to support multi-scale connections in limited area graphs. (#30)
- feat: New method update_graph(graph) in the GraphCreator class. (#60)
- feat: New class StretchedTriNodes to create a stretched mesh. (#51)
- feat: Expanded MultiScaleEdges to support multi-scale connections in stretched graphs. (#51)
- fix: bug in color when plotting isolated nodes (#63)
- Add anemoi-transform link to documentation (#59)
- Added `CutOutMask` class to create a mask for a cutout. (#68)
- Added `MissingZarrVariable` and `NotMissingZarrVariable` classes to create a mask for missing zarr variables. (#68)
- feat: Add CONTRIBUTORS.md file. (#72)
- Create package documentation.

### Changed

- ci: small fixes and updates pre-commit, downsteam-ci (#49)
- Update CODEOWNERS (#61)
- ci: extened python versions to include 3.11 and 3.12 (#66)
- Update copyright notice (#67)

### Removed

- Remove `CutOutZarrDatasetNodes` class. (#68)
- Update CODEOWNERS
- Fix pre-commit regex
- ci: extened python versions to include 3.11 and 3.12
- Update copyright notice
- Fix `__version__` import in init
- The `edge_builder` field in the recipe is renamed to `edge_builders`. It now receives a list of edge builders. (#70)
- The `{source|target}_mask_attr_name` field is moved to inside the edge builder definition. (#70)

## [0.3.0 Anemoi-graphs, minor release](https://github.com/ecmwf/anemoi-graphs/compare/0.2.1...0.3.0) - 2024-09-03

### Added

- HEALPixNodes - nodebuilder based on Hierarchical Equal Area isoLatitude Pixelation of a sphere

- Inspection tools: interactive plots, and distribution plots of edge & node attributes.

- Graph description print in the console.

- CLI entry point: 'anemoi-graphs inspect ...'.

- added downstream-ci pipeline and cd-pypi reusable workflow

- Changelog release updater

- Create package documentation.


### Changed

- fix: added support for Python3.9.
- fix: bug in graph cleaning method
- fix: `anemoi-graphs create` CLI argument is casted to a Path.
- ci: fix missing binary dependency in ci-config.yaml
- fix: Updated `get_raw_values` method in `AreaWeights` to ensure compatibility with `scipy.spatial.SphericalVoronoi` by converting `latitudes` and `longitudes` to NumPy arrays before passing them to the `latlon_rad_to_cartesian` function. This resolves an issue where the function would fail if passed Torch Tensors directly.
- ci: Reusable workflows for push, PR, and releases
- ci: ignore docs for downstream ci
- ci: changed Changelog action to create PR
- ci: fixes and permissions on changelog updater

### Removed

## [0.2.1](https://github.com/ecmwf/anemoi-graphs/compare/0.2.0...0.2.1) - Anemoi-graph Release, bug fix release

### Added

### Changed

- Fix The 'save_path' argument of the GraphCreator class is optional, allowing users to create graphs without saving them.

### Removed

## [0.2.0](https://github.com/ecmwf/anemoi-graphs/compare/0.1.0...0.2.0) - Anemoi-graph Release, Icosahedral graph building

### Added

- New node builders by iteratively refining an icosahedron: TriNodes, HexNodes.
- New edge builders for building multi-scale connections.
- Added Changelog

### Changed

### Removed

## [0.1.0](https://github.com/ecmwf/anemoi-graphs/releases/tag/0.1.0) - Initial Release, Global graph building

### Added

- Documentation
- Initial implementation for global graph building on the fly from Zarr and NPZ datasets

### Changed

### Removed

<!-- Add Git Diffs for Links above -->

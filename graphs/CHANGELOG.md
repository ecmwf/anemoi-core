# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please add your functional changes to the appropriate section in the PR.
Keep it human-readable, your future self will thank you!

## 0.4.4 (2025-02-04)

<!-- Release notes generated using configuration in .github/release.yml at main -->



**Full Changelog**: https://github.com/ecmwf/anemoi-core/compare/anemoi-graphs-0.4.3...anemoi-graphs-0.4.4

## 0.4.3 (2025-02-04)

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

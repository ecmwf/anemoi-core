# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please add your functional changes to the appropriate section in the PR.
Keep it human-readable, your future self will thank you!

## [Unreleased]

### Added
- HEALPixNodes - nodebuilder based on Hierarchical Equal Area isoLatitude Pixelation of a sphere
- added downstream-ci pipeline and cd-pypi reusable workflow

### Changed
- Fix bug in graph cleaning method
- Fix `anemoi-graphs create`. Config argument is cast to a Path.
- Fix GraphCreator().clean() to not iterate over a dictionary that may change size during iterations.
- Fix missing binary dependency
- **Fix**: Updated `get_raw_values` method in `AreaWeights` to ensure compatibility with `scipy.spatial.SphericalVoronoi` by converting `latitudes` and `longitudes` to NumPy arrays before passing them to the `latlon_rad_to_cartesian` function. This resolves an issue where the function would fail if passed Torch Tensors directly.
- ci: Reusable workflows for push, PR, and releases
- ci: ignore docs for downstream ci

### Removed

## [0.2.1] - Anemoi-graph Release, bug fix release

### Added

### Changed
- Fix The 'save_path' argument of the GraphCreator class is optional, allowing users to create graphs without saving them.

### Removed

## [0.2.0] - Anemoi-graph Release, Icosahedral graph building

### Added
- New node builders by iteratively refining an icosahedron: TriNodes, HexNodes.
- New edge builders for building multi-scale connections.
- Added Changelog

### Changed

### Removed

## [0.1.0] - Initial Release, Global graph building

### Added
- Documentation
- Initial implementation for global graph building on the fly from Zarr and NPZ datasets

### Changed

### Removed

<!-- Add Git Diffs for Links above -->
[unreleased]: https://github.com/ecmwf/anemoi-graphs/compare/0.2.1...HEAD
[0.2.1]: https://github.com/ecmwf/anemoi-graphs/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/ecmwf/anemoi-graphs/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/ecmwf/anemoi-graphs/releases/tag/0.1.0

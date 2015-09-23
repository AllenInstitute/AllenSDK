# Change Log
All notable changes to this projection will be documented in this file.

## [Unreleased][unreleased]

## [0.10.1] - 2015-x-x

### Added

- MouseConnectivityCache.get_projection_matrix, method for building a signal matrix from injection structure to projection structure.
- CellTypesCache.get_morphology_features, method for retrieving morphology features for all cells
- CellTypesCache.get_all_features, method for retrieving both morphology and ephys features for all cells in a single table

### Changed

- Ontology.get_child_ids and Ontology.get_descendant_ids accept a list of ids instead of a variable length argument list.

## [0.10.0] - 2015-8-20

### Added

- Manifest and Cache classes for keeping track of files
- MouseConnectivityApi class for downloading data from the Mouse Connectivity Atlas
- MouseConnectivityCache fclass or keeping track of files
- CellTypesCache for keeping track of cell types files
- Ontology class for manipulating structures
- Rma class for formalizing API queries
- GridDataApi for downloading expression and projection grid volumes
- EphysFeatureExtractor module for computing ephys features used in Cell Types Database

### Fixed

- json_utilities has better numpy data type serialization support

## [0.9.1] - 2015-5-13

### Changed

- Documentation updated

### Fixed

- Installation/Makefile bug

## [0.9.0] - 2015-5-12

### Added

- Everything: initial release!
# Change Log
All notable changes to this projection will be documented in this file.

## [0.10.1] - 2015-9-24

### Added

- MouseConnectivityCache.get\_projection\_matrix, method for building a signal matrix from injection structure to projection structure.
- CellTypesCache.get\_morphology\_features, method for retrieving morphology features for all cells
- CellTypesCache.get\_all\_features, method for retrieving both morphology and ephys features for all cells in a single table
- new RmaTemplate class enables construction of queries with jinja2 template library.
- Jupyter notebook examples added to documentation.

### Fixed

- Api.retrieve\_parsed\_json\_over\_http respects post parameter.
- improved installation of dependencies.

### Changed

- Ontology.get\_child\_ids and Ontology.get\_descendant\_ids accept a list of ids instead of a variable length argument list.
- API Access/Data API Client documentation better reflects new 0.10.x allensdk.api.query modules.
- Cache.wrap method defaults to save_as_json=False.
- Cache.wrap method defaults to returning json rather than a pandas dataframe (new parameter return_dataframe=False).
- Replaced brainscales Dockerfile with neuralenseble Dockerfiles.


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

- json\_utilities has better numpy data type serialization support

## [0.9.1] - 2015-5-13

### Changed

- Documentation updated

### Fixed

- Installation/Makefile bug

## [0.9.0] - 2015-5-12

### Added

- Everything: initial release!
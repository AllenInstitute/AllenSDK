# Change Log
All notable changes to this projection will be documented in this file.

## [0.11.0] - 2016-3-3

### Added

- CellTypesCache.get_cells has a new argument 'reporter_status', which accepts one or more ReporterStatus values.
- CellTypesApi.save_reconstruction_marker and CellTypesCache.get_reconstruction_marker download and open files containing positions that mark special points in reconstruction (truncation, early termination of reconstruction, etc).
- swc.read_marker_file for reading in marker files 
- Morphology has new methods for manipulating the morphology tree structure.

### Changed

- Morphology compartments are now Compartment objects that behave like dictionaries
- Compartment.children stores references to immediate decescendant Compartments.
- spike times in NWB files are stored in "analysis/spike_times" instead of "analysis/aibs_spike_times"
- NwbDataSet looks for spike times in both locations ("spike_times" first)
- glif_neuron_methods.py function names have been changed to be more standardized

### Fixed

- MouseConnectivityCache.get_structure_mask bug fixed (github issue 8)
- CellTypesApi.get_ephys_sweeps returns sweeps in sweep number order (github issue 11)
- NwbDataSet.set_spike_times added maxshape(None,) to create_dataset (github issue 7)

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

- json\_utilities has better numpy data type serialization support

## [0.9.1] - 2015-5-13

### Changed

- Documentation updated

### Fixed

- Installation/Makefile bug

## [0.9.0] - 2015-5-12

### Added

- Everything: initial release!
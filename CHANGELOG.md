# Change Log
All notable changes to this project will be documented in this file.

## [0.13.2] - 2017-06-15

### Added

- BrainObservatoryNwbDataSet.get_demixed_traces
- allensdk.brain_observatory.receptive_field_analysis
- allensdk.brain_observatory.demixer

### Changed

- BrainObservatoryCache.get_ophys_experiments returns "acquisition_age_days" instead of "age_days".  The new field describes the age of the animal on the day of experiment acquisition.
- BrainObservatoryCache.get_experiment_containers no longer returns "age_days".
- BrainObservatoryCache.get_ophys_experiments accepts a list of cell_specimen_ids for additional filtering
- json_utilities uses simplejson for better NaN handling


## [0.13.1] - 2017-03-20

### Fixed

- issue #42: Caching behavior in MouseConnectivityCache for CCF volumes fixed
- GLIF examples, documentation fixed

## [0.13.0] - 2017-03-16

### Added

- ReferenceSpace is a new class for relating structure graphs and annotation volumes.
- Standardized caching and paging decorators

### Changed

- Ontology has been deprecated in favor of StructureTree. 
- MouseConnectivityCache uses StructureTree internally (ontology-based methods are deprecated)
- CellTypesCache and MouseConnectivityCache use cacher decorator
- GlifApi has stateless methods now.  Old methods are deprecated.  

### Fixed

- Github issue #35 - MouseConnectivityCache's manifest now uses CCF-version- and resolution-specific file names for masks.  The masks now live inside the CCF version directory.  Users must download a new manifest.

## [0.12.4] - 2016-10-28

### Fixed

- Github issues #23, #28 - added a new dependency "requests_toolbelt" and upgraded API database for more reliable large file downloads.
- Github issue #26 - better documentation for structure unionize records.
- Github issue #25 - documentation errors in brain observatory analysis.

### Changed

- New CCF annotation volume with complete cortical areas and layers.
- Mouse Connectivity structure unionize records have been computed for new CCF.  Previous records are available here: http://download.alleninstitute.org/informatics-archive/june-2016/mouse_projection/
- Github issue #27 - MouseConnectivityCache.get_structure_unionizes returns only requested structures, not all descendants.  Added a separate argument for descendant inclusion.

### Added

- MouseConnectivityCache has a new constructor argument for specifying CCF version.

## [0.12.2] - 2016-9-1

### Fixed

- Github issue #16 (jinja2 requirement)
- Github pull request #21 (spurious "i" typo) in r_neuropil.py

## [0.12.1] - 2016-8-17

### Changed

- neuropil subtraction algorithm (brain_observatory.r_neuropil) faster and more robust
- formatting changes for better PEP8 compliance
- preparation for Python 3 support
- updated Dockerfiles

### Fixed

- Github issue #17 (scipy requirement)

## [0.12.0] - 2016-6-9

### Added

- Support for the Allen Brain Observatory data (BrainObservatoryCache and BrainObservatoryApi classes).
- Code for neurpil subtraction, dF/F estimation, and tuning analysis.
- New ephys feature extractor (ephys_features.py, ephys_extractor.py).  The old one is still there (feature_extractor.py) but should be considered deprecated.

## [0.11.0] - 2016-3-3

### Added

- CellTypesCache.get_cells has a new argument 'reporter_status', which accepts one or more ReporterStatus values.
- CellTypesApi.save_reconstruction_marker and CellTypesCache.get_reconstruction_marker download and open files containing positions that mark special points in reconstruction (truncation, early termination of reconstruction, etc).
- swc.read_marker_file for reading in marker files 
- Morphology has new methods for manipulating the morphology tree structure
- allensdk.model.biophysical package supports active and passive models

### Changed

- Morphology compartments are now Compartment objects that behave like dictionaries
- Compartment.children stores references to immediate descendant Compartments
- spike times in NWB files are stored in "analysis/spike_times" instead of "analysis/aibs_spike_times"
- NwbDataSet looks for spike times in both locations ("spike_times" first)
- glif_neuron_methods.py function names have been changed to be more standardized
- allensdk.model.biophysical_perisomatic package renamed to allensdk.model.biophysical
- NEURON dependency updated to 7.4 release 1370

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
- BiophysicalApi.cache_data throws an exception if no data is found for a neuronal model id.
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

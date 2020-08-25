# Change Log
All notable changes to this project will be documented in this file.

## [2.1.0] = 2020-07-16

### Added
- Behvaior Ophys NWB File writing capability fixes for updated PyNWB and HDMF versions
- Added warning if using outdated Visual Coding Neuropixels NWB files
- Added documentation file for Visual Behavior terms in AllenSDK for quick lookup

## [2.0.0] = 2020-06-11

### Added
- CCF locations for ecephys neuropixels electrodes have been added to their respective nwb electrodes tables
- Examples for accessing eye tracking ellipse fit and screen gaze location data have been added to ecephys example notebooks

### Changed
- pynwb and hdmf version pinning has been relaxed
- The organization of data for ecephys neuropixels Neurodata Without Borders (NWB) files has been significantly changed to conform with NWB specifications and best practices

**Important Note**:
Due to newer versions of pynwb/hdmf having issues reading previously released Visual Coding Neuropixels NWB files and due to the significant reorganization of their NWB file contents, this release contains breaking changes that necessitate a major version revision. NWB files released prior to 6/11/2020 are not guaranteed to work with the 2.0.0 version of AllenSDK. If you cannot or choose not to re-download the updated NWB files, you can continue using a prior version of AllenSDK (< 2.0.0) to access them. However, no further features or bugfixes for AllenSDK (< 2.0.0) are planned. Data released for other projects (Cell Types, Mouse Connectivity, etc.) are *NOT* affected and will *NOT* need to be re-downloaded

## [1.8.0] = 2020-06-06

### Added
- The biophysical module can now run both current and legacy all-active models.
- Internal users can now access `date_of_acquisition` for behavior-only Session data.
- A pull request template was added to the repository.

### Changed
- The CSV log was removed from `BehaviorProjectCache` (internal users).
- Duplicated demixer module was deprecated, and test coverage was added.
- Docker image for AllenSDK was updated.

### Bug Fixes
- Internal LIMS data served to `BehaviorDataSession` class now all use the same timestamp source

## [1.7.1] = 2020-05-5

### Bug Fixes
- Time sync tests were failing because of removed content. Tests were running only on nightly.
- Notebook tests failing on nightly because of down server, switched tests to production server.

## [1.7.0] = 2020-04-29

### Added
- Internal users can now access `eye_tracking` ellipse fit data from behavior + ophys Session objects
- A new mixin for managing processing parameters for Session objects
- Added support for additional sync file line labels

### Changed
- Monitor delay calculation is updated to properly handle photodiode streams that end
on a rising edge. We are no longer providing a default delay value in case of error.

### Bug Fixes
- experiment\_table from behavior project cache has NaNs in the 'imaging\_depth' column for MultiScope experiments due to incorrect join in behavior\_project\_lims\_api.py and 4 other places where ophys\_sessions was incorrectly queried for imaging\_depth\_id
- get_keys method for sync datasets was returning the wrong line labels and creating incorrect key, value pairs for
data loading from sync files

## [1.6.0] = 2020-03-23

### Added
- tutorial for optotagging for ecephys notebook
- get\_receptive\_field() method in ecephys receptive field mapping

### Changed
- remove redundant sham\_change column in behavior sessions.trials table
- versions for NWB output for ecephys and ophys behavior.
- monitor delay is now calculated for BehaviorOphysLimsApi rather than defaulting to 0.0351

### Bug Fixes
- Fixed a bug where auto-rewarded trials were not properly attributed in the rewards property of a visual behavior
- return None rather than raise exception if no container id was returned from lims id for given ophys id
- Project caches no longer accept arbitrary keywords
- matplotloib.pyplot.hist parameter normed no longer supported


## [1.5.0] = 2020-02-10

### Added
 - Users have an option to provide credentials for accessing the database either explicitly via public API or by setting up the environment variables

### Changed
 - Allow users to modify BehaviorDataSession and BehaviorOphysSession data

### Bug Fixes
 - morphology.apply_affine correctly rescales radii
 - invalid extracellular electrophysiology spikes no longer show up as spikes at time -1
 - (internal) When loading a behavior session, behavior and eye tracking video frame times are assessed from the correct lines

## [1.3.0] = 2019-12-12


### Added
 - Improved Neuropixels data download performance by enabling asynchronous transfers. Data downloads will now raise timeout errors when data cannot be retrieved in a reasonable timeframe.

### Changed
 - Updated AllenSDK readme and contributing documentation

### Bug Fixes
 - https://github.com/AllenInstitute/AllenSDK/issues/1214 Fix hanging downloads for Neuropixels NWB files


## [1.0.0] = 2019-10-3

### Added
 - Support for Brain Observatory - Visual Coding Neuropixels data.
 - https://github.com/AllenInstitute/AllenSDK/issues/447 Implemented improved eye tracking module based on DeepLabCut

### Changed
 - Python 2.x is no longer supported.
 - statsmodels is now pinned at 0.9.0

### Bug Fixes
 - https://github.com/AllenInstitute/AllenSDK/commit/f76678 Fix integer division bug in receptive field analysis


## [0.16.3] = 2019-5-22

### Bug Fixes
 - https://github.com/AllenInstitute/AllenSDK/issues/660 Use pillow/Image.resize instead of scipy.misc.imresize
 - https://github.com/AllenInstitute/AllenSDK/issues/659 Same
 - https://github.com/AllenInstitute/AllenSDK/issues/661 Update example notebooks

## [0.16.2] = 2019-4-23

### Added
 - https://github.com/AllenInstitute/AllenSDK/issues/549 Transforms for tissuecyte registration accessible from allensdk
 - https://github.com/AllenInstitute/AllenSDK/pull/568 Release instructions for future maintainers.

### Changed
 - https://github.com/AllenInstitute/AllenSDK/issues/582 Documentation builds from a yml config in python 3.7

### Bug Fixes
 - https://github.com/AllenInstitute/AllenSDK/issues/523 pip no longer fails on new conda environment because of tables

## [0.16.1] = 2019-3-12

### Added
 - Added tools for working with itksnap label descriptions (https://github.com/AllenInstitute/AllenSDK/issues/312)

### Changed
 - Update pytest version number (now requires pytest>=4.1.1)
 - Update numpy version number (now requires numpy>=1.15.1)
 - Removed deprecated functions from BrainObservatoryNwbDataSet
 - Updated documentation

### Bug Fixes
 - https://github.com/AllenInstitute/AllenSDK/issues/256 Examples python3 compatible
 - https://github.com/AllenInstitute/AllenSDK/issues/267 Fix get_cell_specimens filter argument
 - https://github.com/AllenInstitute/AllenSDK/pull/226 Fix pandas sorting behavior warning
 - https://github.com/AllenInstitute/AllenSDK/issues/207 Fix "two celltypes features"
 - https://github.com/AllenInstitute/AllenSDK/issues/273 Fix "make notebook"
 - https://github.com/AllenInstitute/AllenSDK/issues/275 Fix inconsistent numpy requirement
 - https://github.com/AllenInstitute/AllenSDK/issues/295 Fix BiophysicalApi.get_neuronal_models
 - https://github.com/AllenInstitute/AllenSDK/issues/309 Fix tests to use updated pytest
 - https://github.com/AllenInstitute/AllenSDK/issues/307 Fix openjpeg dependency in python3 in the CI build
 - https://github.com/AllenInstitute/AllenSDK/issues/311 Fix locally sparse noise test
 - https://github.com/AllenInstitute/AllenSDK/issues/330 Fix numpy warning message
 - https://github.com/AllenInstitute/AllenSDK/issues/330 Fix Pillow dependency install
 - https://github.com/AllenInstitute/AllenSDK/issues/405 Fix session analysis regression test (passing CI in python2 and python3)
 - https://github.com/AllenInstitute/AllenSDK/issues/420 Fix installation bug resulting from pytables 3.5
 - https://github.com/AllenInstitute/AllenSDK/commit/1ef88046a58a36d870d3f1048a778806d1db2954 Fix scikit-image dependency install

## [0.16.0] = 2018-10-04

### Added
- BrainObservatoryCache.get_ophys_experiment_events
- BrainObservatoryCache.get_ophys_experiments and get_experiment_containers accept reporter_lines argument
- BrainObservatoryCache.ge_ophys_experiment_analysis
- Cache subclasses can have their manifest locations set via environment variable

## [0.14.5] = 2018-06-14

### Changed
- bumped lower limit of pandas to 0.17, removed upper limit
- changed CellTypesCache cell data structure to be less nested, more flat. Requires update to manifest and re-download of cells.json.
- simplified MouseConnectivityCache data structure.  Requires update to manifest and re-download of experiments.json.
- stopped using deprecated function in SimpleTree

### Added
- regression tests for Brain Observatory analysis

## [0.14.4] - 2018-01-30

### Changed

- Numerous Python 2/3 compatibility fixes
- Improved spike filtering in ephys.ephys_features

## [0.14.3] - 2017-10-19

### Added

- CellTypesCache.get_cells has a new `species` argument.

### Changed

- MouseConnectivityCache downloads structure masks instead of computing them.

## [0.14.2] - 2017-08-17

### Added

- ABS-64; Jupyterhub Dockerfiles
- PR #13; Manifest error check

### Changed

ABS-132; changed to Allen Institute Software License
ABS-129; StructureTree converts color hex triplet to 8-bit color triplet
ABS-125; update expected data for unmocked tests
PBS-358; update extract_trace module to deal with motion border case
ABS-85; allow variable long square start time in ephys feature extractor

## [0.13.2] - 2017-06-15

### Added

- BrainObservatoryNwbDataSet.get_demixed_traces
- BrainObservatoryNwbDataSet.get_pupil_location
- BrainObservatoryNwbDataSet.get_pupil_size
- BrainObservatoryApi.get_cell_specimen_id_mapping
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

.. Allen SDK documentation master file, created by
   sphinx-quickstart on Mon Jul  1 14:31:44 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:tocdepth: -1

Welcome to the Allen SDK
========================

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   install
   data_resources
   models
   examples
   authors
   Source Documentation <allensdk>
   Github Profile <https://github.com/AllenInstitute/AllenSDK>
   

The Allen Software Development Kit houses source code for reading and processing Allen Brain Atlas data.  
The Allen SDK focuses on the Allen Brain Observatory, Cell Types Database, and Mouse Brain Connectivity Atlas.

.. ATTENTION::
    As of October 2019, we have dropped Python 2 support and any files with a py2 dependency (for example analysis files) have been updated.

.. image:: /_static/sdk_cam.png
   :align: right


Allen Brain Observatory
-----------------------

The `Allen Brain Observatory <http://observatory.brain-map.org/visualcoding>`_ is a data resource for
understanding sensory processing in the mouse visual cortex.  This study systematically measures visual
responses in multiple cortical areas and layers using two-photon calcium imaging of GCaMP6-labeled neurons 
targeted using Cre driver lines.  Response characterizations include orientation tuning, spatial and temporal
frequency tuning, temporal dynamics, and spatial receptive field structure.
 
The mean fluorescence traces for all segmented cells are available in the Neurodata Without Borders file format 
(`NWB files <brain_observatory_nwb.html>`_).  These files contain standardized descriptions of visual stimuli to support stimulus-specific tuning analysis.  The Allen SDK provides code to:

   * download and organize experiment data according to cortical area, imaging depth, and Cre line
   * remove the contribution of neuropil signal from fluorescence traces
   * access (or compute) dF/F traces based on the neuropil-corrected traces
   * perform stimulus-specific tuning analysis (e.g. drifting grating direction tuning)

....

.. image:: /_static/ccf_v3_sdk.png
   :align: right



Allen Cell Types Database
-------------------------

The `Allen Cell Types Database <http://celltypes.brain-map.org>`_ contains electrophysiological and morphological characterizations
of individual neurons in the mouse primary visual cortex.  The Allen SDK provides Python code
for accessing electrophysiology measurements (`NWB files <cell_types.html#neurodata-without-borders>`_) 
for all neurons and morphological reconstructions (`SWC files <cell_types.html#morphology-swc-files>`_) for a subset of neurons.

The Database also contains two classes of models fit to this data set: biophysical
models produced using the NEURON simulator and generalized leaky integrate and fire models (GLIFs) 
produced using custom Python code provided with this toolkit. 

The Allen SDK provides sample code 
demonstrating how to download neuronal model parameters from the Allen Brain Atlas API and run 
your own simulations using stimuli from the Allen Cell Types Database or custom current injections:

   * :doc:`biophysical_models`
   * :doc:`glif_models`


....

.. image:: /_static/connectivity.png
   :align: right

Allen Mouse Brain Connectivity Atlas
------------------------------------

The `Allen Mouse Brain Connectivity Atlas <http://connectivity.brain-map.org>`_ is a high-resolution map of neural connections in the mouse brain. Built on an array of transgenic mice genetically engineered to target specific cell types, the Atlas comprises a unique compendium of projections from selected neuronal populations throughout the brain.  The primary data of the Atlas consists of high-resolution images of axonal projections targeting different anatomic regions or various cell types using Cre-dependent specimens. Each data set is processed through an informatics data analysis pipeline to obtain spatially mapped quantified projection information.

The Allen SDK provides Python code for accessing experimental metadata along with projection signal volumes registered to a common coordinate framework.  This framework has structural annotations, which allows users to compute structure-level signal statistics.

See the `mouse connectivity section <connectivity.html>`_ for more details.

What's New - 1.8.0 (June 6, 2020)
-----------------------------------------------------------------------

As of the 1.8.0 release:

- The biophysical module can now run both current and legacy all-active models.
- A pull request template was added to the repository.
- Duplicated demixer module was deprecated, and test coverage was added.
- Docker image for AllenSDK was updated.

For internal users:
- The `date_of_acquisition` field is available for behavior-only Session data.
- The CSV log was removed from `BehaviorProjectCache`
- Fixed a bug so LIMS data served to `BehaviorDataSession` now all use the same timestamp source.

What's New - 1.7.1 (May 5, 2020)
-----------------------------------------------------------------------

As of the 1.7.1 release:

- Added a bug fix to correct nightly tests of AllenSDK and prevent failure
- Added a bug fix to move nightly notebook tests to using production endpoint

What's New - 1.7.0 (April 29, 2020)
-----------------------------------------------------------------------

As of the 1.7.0 release:

- Added functionality so internal users can now access `eye_tracking` ellipse fit data from behavior + ophys Session objects
- Added a new mixin for managing processing parameters for Session objects
- Update the monitor delay calculation to better handle edge cases; no longer provide a default delay value if encounter an error
- Added support for additional sync file line labels
- Fixed bug with loading line labels from sync files

What's New - 1.6.0 (March 23, 2020)
-----------------------------------------------------------------------

As of the 1.6.0 release:

- added get_receptive_field alias() for _get_rf() in allensdk/brain_observatory/ecephys/stimulus_analysis/receptive_field_mapping.py
- Added required version to namespace and caches spec in ecephy nwb outputs in allensdk/brain_observatory/ecephys/nwb/AIBS_ecephys_namespace.yaml
- Added version for ophys behavior nwb output to allensdk/brain_observatory/nwb/AIBS_ophys_behavior_namespace.yaml
- Behavior and ECEphys project caches no longer accept arbitrary keywords to prevent confusion when user supplies incorrect kwargs to constructor.
- New ecephys notebook for optotagging tutorial.


What's New - 1.5.0 (February 10, 2020)
-----------------------------------------------------------------------

As of the 1.5.0 release:

- users have an option to provide credentials for accessing the database either explicitly via public API or by setting up the environment variables
- allow users to modify BehaviorDataSession and BehaviorOphysSession data
- invalid extracellular electrophysiology spikes no longer show up as spikes at time -1
- morphology.apply_affine correctly rescales radii


Previous Release Notes
----------------------
    * `1.4.0 <https://github.com/AllenInstitute/AllenSDK/releases/tag/v1.4.0>`_
    * `1.3.0 <https://github.com/AllenInstitute/AllenSDK/releases/tag/v1.3.0>`_
    * `1.2.0 <https://github.com/AllenInstitute/AllenSDK/releases/tag/v1.2.0>`_
    * `1.1.1 <https://github.com/AllenInstitute/AllenSDK/releases/tag/v1.1.1>`_
    * `1.1.0 <https://github.com/AllenInstitute/AllenSDK/releases/tag/v1.1.0>`_
    * `1.0.2 <https://github.com/AllenInstitute/AllenSDK/releases/tag/v1.0.2>`_
    * `0.16.3 <https://github.com/AllenInstitute/AllenSDK/releases/tag/v0.16.3>`_
    * `0.16.2 <https://github.com/AllenInstitute/AllenSDK/wiki/Release-Notes-(0.16.2)>`_
    * `0.16.1 <https://github.com/AllenInstitute/AllenSDK/wiki/Release-Notes-(0.16.1)>`_
    * `0.16.0 <https://github.com/AllenInstitute/AllenSDK/wiki/Release-Notes-(0.16.0)>`_
    * `0.14.5 <https://github.com/AllenInstitute/AllenSDK/wiki/Release-Notes-(0.14.5)>`_
    * `0.14.4 <https://github.com/AllenInstitute/AllenSDK/wiki/Release-Notes-(0.14.4)>`_
    * `0.14.3 <https://github.com/AllenInstitute/AllenSDK/wiki/Release-Notes-(0.14.3)>`_
    * `0.14.2 <https://github.com/AllenInstitute/AllenSDK/wiki/Release-Notes-(0.14.2)>`_
    * `0.13.2 <https://github.com/AllenInstitute/AllenSDK/wiki/Release-Notes-(0.13.2)>`_
    * `0.13.1 <https://github.com/AllenInstitute/AllenSDK/wiki/Release-Notes-(0.13.1)>`_
    * `0.13.0 <https://github.com/AllenInstitute/AllenSDK/wiki/Release-Notes-(0.13.0)>`_
    * `0.12.4 <https://github.com/AllenInstitute/AllenSDK/wiki/Release-Notes-(0.12.4)>`_

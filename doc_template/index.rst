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

What's New - Release 0.15.2 (October 4th, 2018)
-----------------------------------------------

The 0.15.2 release coincides with the release of significantly more data in the Allen Brain Observatory.  To access the new data, remove the following files from your `BrainObservatoryCache` manifest directory: 

    * ``cell_specimens.json``
    * ``experiment_containers.json``
    * ``ophys_experiments.json``

Users now have access to methods for accessing new data derived from fluorescence traces:

    * :py:meth:`~allensdk.core.BrainObservatoryCache.get_ophys_experiment_analysis`: traces and metrics organized by stimulus condition
    * :py:meth:`~allensdk.core.BrainObservatoryCache.get_ophys_experiment_events`: calcium events determined with `FastLZeroSpikeInference <https://github.com/jewellsean/FastLZeroSpikeInference>`_

Note: we addressed `GitHub issue #69 <https://github.com/AllenInstitute/AllenSDK/issues/69>`_ and a more general time alignment error in our processing pipeline.  Handling of additional frame pulses has been corrected, resulting in 2-photon movie frames being assigned times 30ms earlier than before.  We have updated all previously published NWB files accordingly.  Delete these files and re-download them to access updated time stamps.

This release also contains more mouse and human data in the Allen Cell Types Database.  To access the new data, remove the following files from your `CellTypesCache` manifest directory:

    * :py:meth:`~allensdk.core.cell_types_cache.CellTypesCache` manifest.json
    * ``cells.json`` 
    * ``ephys_features.csv`` 
    * ``morphology_features.csv`` 


Previous Release Notes
----------------------

    * `0.14.5 <https://github.com/AllenInstitute/AllenSDK/wiki/Release-Notes-(0.14.5)>`_
    * `0.14.4 <https://github.com/AllenInstitute/AllenSDK/wiki/Release-Notes-(0.14.4)>`_
    * `0.14.3 <https://github.com/AllenInstitute/AllenSDK/wiki/Release-Notes-(0.14.3)>`_
    * `0.14.2 <https://github.com/AllenInstitute/AllenSDK/wiki/Release-Notes-(0.14.2)>`_
    * `0.13.2 <https://github.com/AllenInstitute/AllenSDK/wiki/Release-Notes-(0.13.2)>`_
    * `0.13.1 <https://github.com/AllenInstitute/AllenSDK/wiki/Release-Notes-(0.13.1)>`_
    * `0.13.0 <https://github.com/AllenInstitute/AllenSDK/wiki/Release-Notes-(0.13.0)>`_
    * `0.12.4 <https://github.com/AllenInstitute/AllenSDK/wiki/Release-Notes-(0.12.4)>`_


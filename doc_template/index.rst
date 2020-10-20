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


What's New - 2.3.2 (October 19, 2020)
-----------------------------------------------------------------------
As of the 2.3.2 release:

- (Internal) Fixed a running_processing bug for behavior ophys experiments when the input data would have one more encoder entry than timestamp. The behavior of the code now matches what the warning says.


What's New - 2.3.1 (October 13, 2020)
-----------------------------------------------------------------------
As of the 2.3.1 release:

- (Internal) Fixed a write_nwb bug for behavior ophys experiments involving the BehaviorOphysJsonApi expecting a mesoscope-specific method.


What's New - 2.3.0 (October 9, 2020)
-----------------------------------------------------------------------
As of the 2.3.0 release:

- Visual behavior running speed is now low-pass filtered at 10Hz. The raw running speed data is still available. The running speed is corrected for encoder threshold croissing artifacts.
- Support for stimulus gratings for visual behavior.
- Fixed an eye-tracking sync problem.
- Updates to some visual behavior pynwb implementations.
- Adds load sync data for individual plane sets to relate accurate event timings to mesoscope data.
- Adds public API method to access the behavior_session_id from an instance of BehaviorOphysSession.


What's New - 2.2.0 (September 3, 2020)
-----------------------------------------------------------------------
As of the 2.2.0 release:

- AllenSDK HTTP engine streaming requests now include a progress bar
- `ImportError: cannot import name 'MultiContainerInterface' from 'hdmf.container'` errors should now be resolved (by removing explicit version bounds on the `hdmf` package).
- The optical physiology 2-photon trace demixer has been modified to be more memory friendly and should no longer result in out of memory errors when trying to demix very large movie stacks.


What's New - 2.1.0 (July 16, 2020)
-----------------------------------------------------------------------
As of the 2.1.0 release:

- behavior ophys nwb files can now be written using updated pynwb and hdmf
- A warning has been added if you are using AllenSDK with outdated NWB files
- A new documentation file has been added which will contain Visual Behavior specific terms for quick lookup

What's New - 2.0.0 (June 11, 2020)
-----------------------------------------------------------------------

As of the 2.0.0 release:

- pynwb and hdmf version requirements have been made less strict
- The organization of data for ecephys neuropixels Neurodata Without Borders (NWB) files has been significantly changed to conform with NWB specifications and best practices
- CCF locations for ecephys neuropixels electrodes are now written to NWB files
- Examples for accessing eye tracking ellipse fit and screen gaze location data have been added to ecephys example notebooks

**Important Note**:
Due to newer versions of pynwb/hdmf having issues reading previously released Visual Coding Neuropixels NWB files and due to the significant reorganization of their NWB file contents, this release contains breaking changes that necessitate a major version revision. NWB files released prior to 6/11/2020 are not guaranteed to work with the 2.0.0 version of AllenSDK. If you cannot or choose not to re-download the updated NWB files, you can continue using a prior version of AllenSDK (< 2.0.0) to access them. However, no further features or bugfixes for AllenSDK (< 2.0.0) are planned. Data released for other projects (Cell Types, Mouse Connectivity, etc.) are *NOT* affected and will *NOT* need to be re-downloaded

Previous Release Notes
----------------------
    * `1.8.0 <https://github.com/AllenInstitute/AllenSDK/releases/tag/v1.8.0>`_
    * `1.7.1 <https://github.com/AllenInstitute/AllenSDK/releases/tag/v1.7.1>`_
    * `1.7.0 <https://github.com/AllenInstitute/AllenSDK/releases/tag/v1.7.0>`_
    * `1.6.0 <https://github.com/AllenInstitute/AllenSDK/releases/tag/v1.6.0>`_
    * `1.5.0 <https://github.com/AllenInstitute/AllenSDK/releases/tag/v1.5.0>`_
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

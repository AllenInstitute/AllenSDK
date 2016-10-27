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

What's New - Release 0.12.3 (October 27th, 2016)
------------------------------------------------

The 0.12.3 release features a new 2016 version of the Allen Mouse Common Coordinate Framework.  This primarily affects Mouse Connectivity Atlas `structure unionize records <unionizes.html>`_ and structure masks.  All structure unionize records downloaded via the MouseConnectivityCache or API now correspond to the new CCF.  More details on CCF construction can be found in the `CCF technical whitepaper <http://help.brain-map.org/download/attachments/2818171/Mouse_Common_Coordinate_Framework.pdf?version=2&modificationDate=1477414987120>`_.  Users can download the old `unionize records here <http://download.alleninstitute.org/informatics-archive/june-2016/mouse_projection/>`_.

This release also addresses issues from the Github issue tracker:
  
    * issues #23, #28 - added a new dependency "requests_toolbelt" and upgraded API database for more reliable large file downloads.
    * issue #27 - MouseConnectivityCache.get_structure_unionizes returns only requested structures, not all descendants.  Added a separate argument for descendant inclusion.
    * issue #26 - documentation for structure unionize records `here <unionizes.html>`_.
    * issue #25 - documentation errors in brain observatory analysis.

To find out more, take a look at our `CHANGELOG <http://github.com/AllenInstitute/AllenSDK/blob/master/CHANGELOG.md>`_. 

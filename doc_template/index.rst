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

What's New - Release 0.14.3 (October 19th, 2017)
-----------------------------------------------

The 0.14.3 release coincides with the first release of human data and models in the Allen Cell Types Database and a complete requantification of structure unionize
records in the Allen Mouse Brain Connectivity Atlas based on a new revision of Common Coordinate Framework structure ontology and voxel annotations.

Users of the :py:class:`~allensdk.core.cell_types_cache.CellTypesCache` can filter for cells based on the species of the cell's donor using the ``species`` argument of 
:py:meth:`~allensdk.core.cell_types_cache.CellTypesCache.get_cells`.  Examples of this are shown in the metadata filtering section of the example 
`Jupyter notebook <_static/examples/nb/cell_types.html>`_

The Allen Mouse Brain Connectivity Atlas contains over 350 new data sets and structure unionize records have been completely reprocessed with updated 3D annotations of the Common Coordinate Framework.
The structure ontology contains new structures, with annotations have changing the most.  :py:meth:`~allensdk.core.mouse_connectivity_cache.MouseConnectivityCache.get_annotation_volume` method will by default return a new volume by default.  You can choose which version of annotations you would like using the ``ccf_version`` :py:class:`~allensdk.core.mouse_connectivity_cache.MouseConnectivityCache` constructor.

To access new experiments and unionize records, you will need to remove a number of files in your manifest directory so that 
:py:class:`~allensdk.core.mouse_connectivity_cache.MouseConnectivityCache` will know to download the new copies:

    * :py:class:`~allensdk.core.mouse_connectivity_cache.MouseConnectivityCache` manifest JSON
    * ``experiments.json`` 
    * ``structures.json`` 
    * ``structure_unionizes.csv`` (one per experiment within experiment subdirectories)  

You can then call :py:meth:`~allensdk.core.mouse_connectivity_cache.MouseConnectivityCache.get_experiments`, 
:py:meth:`~allensdk.core.mouse_connectivity_cache.MouseConnectivityCache.get_structure_tree`, and
:py:meth:`~allensdk.core.mouse_connectivity_cache.MouseConnectivityCache.get_structure_unionizies` to download the files above.

To find out more, take a look at our `CHANGELOG <http://github.com/AllenInstitute/AllenSDK/blob/master/CHANGELOG.md>`_. 

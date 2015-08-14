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
   Source Documentation <allensdk>
   Github Profile <https://github.com/AllenInstitute/AllenSDK>

The Allen Software Development Kit houses source code for reading and processing Allen Brain Atlas data.  
The Allen SDK focuses the Allen Cell Types Database and the Allen Mouse Brain Connectivity Atlas.  
Functionality relevant to other atlases is coming in future releases.  

.. image:: /_static/ccf_v3_sdk.png
    :align: right


Allen Cell Types Database
-------------------------

The `Allen Cell Types Database <http://celltypes.brain-map.org>`_ contains electrophysiological and morphological characterizations
of individual neurons in the mouse primary visual cortex.  The Allen SDK provides Python code
for accessing electrophysiology measurements (`NWB files <cell_types.html#neurodata-without-borders>`_) 
for all neurons and morphological reconstructions (`SWC files <cell_types.html#morphology-swc-files>`_) for a subset of neurons.

The Database also contains two classes of models fit to this data set: perisomatic biophysical 
models produced using the NEURON simulator and generalized leaky integrate and fire models (GLIFs) 
produced using custom Python code provided with this toolkit. 

The Allen SDK provides sample code 
demonstrating how to download neuronal model parameters from the Allen Brain Atlas API and run 
your own simulations using stimuli from the Allen Cell Types Database or custom current injections:

    * :doc:`biophysical_perisomatic_script`
    * :doc:`glif_models`


....

.. image:: /_static/connectivity.png
    :align: right

Allen Mouse Brain Connectivity Atlas
------------------------------------

The `Allen Mouse Brain Connectivity Atlas <http://connectivity.brain-map.org>`_ is a high-resolution map of neural connections in the mouse brain. Built on an array of transgenic mice genetically engineered to target specific cell types, the Atlas comprises a unique compendium of projections from selected neuronal populations throughout the brain.  The Allen SDK provides Python code for accessing experimental metadata along with projection signal volumes registered to a common coordinate framework.

See the `mouse connectivity section <connectivity.html>`_ for more details.

....

Developer Documentation
-----------------------

Please refer to the :ref:`modindex` for inline source code documentation. 

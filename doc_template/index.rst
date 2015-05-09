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
   Package Index <allensdk>

The Allen Software Development Kit houses source code for reading and processing Allen Brain Atlas data.  
For the initial release, the Allen SDK focuses primarily on the newly released Allen Cell Types Database.
Functionality relevant to other atlases is coming in future releases.

.. image:: /_static/ccf_v3_sdk.png
    :align: right

Allen Cell Types Database
-------------------------


The Allen Cell Types Database contains electrophysiological and morphological characterizations
of individual neurons in the mouse primary visual cortex.  The Allen SDK provides Python code
for accessing electrophysiology measurements (`NWB <cell_types.html#neurodata-without-borders>`_ files) for all neurons and morphological 
reconstructions (SWC files) for a subset of neurons.

The Database also contains two classes of models fit to this data set: perisomatic biophysical 
models produced using the NEURON simulator and generalized leaky integrate and fire models (GLIFs) 
produced using custom Python code provided with this toolkit. 

The Allen SDK provides sample code 
demonstrating how to download neuronal model parameters from the Allen Brain Atlas API and run 
your own simulations using stimuli from the Allen Cell Types Database or custom current injections:

    * :doc:`biophysical_perisomatic_script`
    * :doc:`glif_models`


....

Developer Documentation
-----------------------

Please refer to the :ref:`modindex` for inline source code documentation. 

Brain Observatory
=================

The `Allen Brain Observatory <http://observatory.brain-map.org/visualcoding>`_ is a database of the visually-evoked functional
responses of neurons in mouse visual cortex based on 2-photon fluorescence imaging.  Characterized responses include orientation 
tuning, spatial and temporal frequency tuning, temporal dynamics, and spatial receptive field structure. 

The data is organized into experiments and experiment containers.  An experiment container represents a group of 
experiments with the same targeted imaging area, imaging depth, and Cre line.  The individual experiments within 
an experiment container have different stimulus protocols, but cover the same imaging field of view.  

.. image:: /_static/container_session_layout.png
   :align: center

Data collected after September 2016 uses a new session C stimulus designed to better-characterize spatial receptive fields in 
higher visual areas.  The original locally sparse noise stimulus used 4 visual degree pixels.  Session C2 broke that stimulus 
blocks into two separate stimuli: one with 4 degree pixels and one with 8 degree pixels.  

For more information on experimental design and a data overview, please visit the `Allen Brain Observatory data portal <http://observatory.brain-map.org/visualcoding>`_.  


Data Processing
---------------

For all data in Allen Brain Observatory, we perform the following processing:

   1. Segment cell masks from each experiment's 2-photon fluorescence video
   2. Associate cells from experiments belonging to the same experiment container and assign unique IDs
   3. Extract each cell's mean fluorescence trace
   4. Extract mean fluorescence traces from each cell's surrounding neuropil
   5. Estimate neuropil-corrected fluorescence traces
   6. Demix traces from overlapping ROIs
   7. Compute dF/F 
   8. Compute stimulus-specific tuning metrics 

All traces and masks for segmented cells in an experiment are stored in a Neurodata Without Borders (NWB) file.
Stored traces include the raw fluoresence trace, neuropil trace, demixed trace, and dF/F trace.  Code for extracting neuropil-corrected
fluorescence traces, computing dF/F, and computing tuning metrics is available in the SDK.  

**Note:** trace demixing is a new addition as of June 2017.  All past data was reprocessed using the new demixing algorithm.

For more information about data processing, please `read the technical whitepapers <http://help.brain-map.org/display/observatory/Documentation>`_.


Getting Started
---------------

The Brain Observatory `Jupyter notebook <_static/examples/nb/brain_observatory.html>`_ has many code samples to help get
started with the available data:

    - `Download experimental metadata by visual area, imaging depth, and Cre line <_static/examples/nb/brain_observatory.html#Experiment-Containers>`_
    - `Find cells with specific response properties, like direction tuning <_static/examples/nb/brain_observatory.html#Find-Cells-of-Interest>`_
    - `Download data for an experiment <_static/examples/nb/brain_observatory.html#Download-Experiment-Data-for-a-Cell>`_
    - `Plot raw fluorescences traces, neuropil-corrected traces, and dF/F <_static/examples/nb/brain_observatory.html#Fluorescence-Traces>`_
    - `Find the ROI mask for a given cell <_static/examples/nb/brain_observatory.html#ROI-Masks>`_    
    - `Run neuropil correction <_static/examples/nb/brain_observatory.html#Neuropil-Correction>`_

The code used to analyze and visualize data in the `Allen Brain Observatory data portal <http://observatory.brain-map.org/visualcoding>`_ 
is available as part of the SDK.  Take a look at this `Jupyter notebook <_static/examples/nb/brain_observatory_analysis.html>`_ to find out how to:

    - `Plot cell's response to its preferred stimulus condition <_static/examples/nb/brain_observatory_analysis.html#Drifting-Gratings>`_    
    - `Compute a cell's on/off receptive field based on the locally sparse noise stimulus <_static/examples/nb/brain_observatory_analysis.html#Locally-Sparse-Noise>`_ 

More detailed documentation is available demonstrating how to: 

    - `Read and visualize the stimulus presentation tables in the NWB files <_static/examples/nb/brain_observatory_stimuli.html>`_
    - `Understand the layout of Brain Observatory NWB files <brain_observatory_nwb.html>`_ 





   







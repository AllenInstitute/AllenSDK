Brain Observatory
=================

The `Allen Brain Observatory <http://activity.brain-map.org/visualcoding>`_ is database of the visually-evoked functional
responses of neurons in mouse visual cortex based on 2-photon fluorescence imaging.  Characterized responses include orientation 
tuning, spatial and temporal frequency tuning, temporal dynamics, and spatial receptive field structure. 

The data is organized into experiments and experiment containers.  An experiment container represents a group of 
experiments with the same targeted imaging area, imaging depth, and Cre line.  The individual experiments within 
an experiment container have different stimulus protocols, but cover the same field of view.  

.. image:: /_static/container_session_layout.png
   :align: center

For more information on the visual stimuli, please visit the 
`Allen Brain Observatory data portal <http://activity.brain-map.org/visualcoding>`_.  Individual stimuli are described 
in detail (e.g. `static gratings <http://activity.brain-map.org/visualcoding/stimulus/static_gratings>`_).


Data Processing
---------------

For all data in Allen Brain Observatory, we perform the following processing:

   1. Segment cell masks from each experiment
   2. Associate cells from experiments belonging to the same experiment container and assign unique IDs
   3. For each cell:

      a. Extract mean fluorescence traces
      b. Extract mean fluorescence traces from surrounding neuropil
      c. Compute neuropil-corrected fluorescence traces 
      d. Compute dF/F
      
   4. Compute stimulus-specific tuning metrics

All traces and masks for segmented cells in an experiment are stored in the Neurodata Without Borders (NWB) format.
Stored traces include the raw fluoresence trace, neuropil trace, and dF/F trace.  Code for extracting neuropil-corrected
fluorescence traces, computing dF/F, and computing tuning metrics is available in the SDK.  

For more information about data processing, please `read the technical whitepapers <help.alleninstitute.org/display/cam/Documentation>`_.


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

More detailed documentation is available demonstrating how to: 

    - `Read and visualize the stimulus presentation tables in the NWB files <_static/examples/nb/brain_observatory_stimuli.html>`_
    - `Run stimulus-specific tuning analysis <_static/examples/nb/brain_observatory_analysis.html>`_
    - `Understand the layout of Brain Observatory NWB files <brain_observatory_nwb.html>`_ 





   







Brain Observatory
=================

The `Allen Brain Observatory <http://activity.brain-map.org/visualcoding>`_ is database of the visually-evoked functional
responses neurons in mouse visual cortex based on 2-photon fluorescence imaging.  Characterized responses include orientation 
tuning, spatial and temporal frequency tuning, temporal dynamics, and spatial receptive field structure. 

The data is primarily organized into experiments and experiment containers.  An experiment container represents a group of 
experiments imaged at the same location (cortical area, and depth).  The individual experiments within 
an experiment container have different stimulus protocols, but cover the same field of view.  For example, the static grating
stimulus is presented in a different experiment than the natural scene stimulus.  

Because different stimuli may evoke responses from disjoint sets of cells, cells are segmented separately in each 
experiment. Segmented cells for experiments belonging to the same experiment container are then associated
with each other in a post-process.  

All traces for segmented cells in a single experiment are stored in the Neurodata Without Borders (NWB) format.
Traces include every cell's mean fluoresence trace, neuropil trace, and dF/F trace.  Code for extracting neuropil-corrected
traces is available in the SDK. 

For more details on experimental design and data processing, please refer to the technical whitepapers (TODO).

Visual Stimuli
--------------

**Locally sparse noise**

Locally sparse noise stimuli consist of several black and/or white squares placed in random locations on a gray background.
The displayed image was warped when displayed to help compensate for angular differences in square size when displayed on a flat monitor (e.g., the left and right edges of the display were magnified to compensate for this part of the monitor being further way from the mouse).

.. image:: images/locally_sparse_noise.png

**Natural scenes**

A variety of natural images were presented. These include things like flowers, animals and outdoor scenes.

.. image:: images/natural_scenes.png

**Natural movie**

Parts of the Orson Welles movie 'Touch of Evil' were shown. Popcorn was not provided.

.. image:: images/natural_movie.png

**Drifting grating**

Moving gratings with different orientation, spatial and temporal frequency were presented.

.. image:: images/animated_drifting_grating.gif

**Static grating**

Non-moving gratings with similar variation in orientation, spatial and temporal frequency were also shown.

.. image:: images/static_250px_sfreq_0.04.png



Code Samples
------------

The Brain Observatory `Jupyter notebook <_static/examples/nb/brain_observatory.html>`_ has many code samples to help get
started with analysis:

    - `Download experimental metadata by visual area, imaging depth, and transgenic line <_static/examples/nb/brain_observatory.html#Experiment-Containers>`_
    - `Find cells with specific response properties, like direction tuning <_static/examples/nb/brain_observatory.html#Find-Cells-of-Interest>`_
    - `Visualize the experiment by stimulus type <_static/examples/nb/brain_observatory_stimuli.html#Drifting-Gratings-Stimulus>`_
    - `Find the ROI mask for a given cell <_static/examples/nb/brain_observatory.html#ROI-Masks>`_
    - `Run drifting gratings tuning analysis <_static/examples/nb/brain_observatory.html#ROI-Analysis>`_
    - `Plot raw fluorescences traces, neuropil-corrected traces, and dF/F <_static/examples/nb/brain_observatory.html#Fluorescence-Traces>`_
    - `Run neuropil correction <_static/examples/nb/brain_observatory.html#Neuropil-Correction>`_



NWB File
--------

Details about the data files available for download is `here <brain_observatory_nwb.html>`_.



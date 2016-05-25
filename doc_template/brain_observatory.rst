Brain Observatory
=================

The `Allen Brain Observatory <http://activity.brain-map.org/visualcoding>`_ is database of the visually-evoked functional
responses of neurons in mouse visual cortex based on 2-photon fluorescence imaging.  Characterized responses include orientation 
tuning, spatial and temporal frequency tuning, temporal dynamics, and spatial receptive field structure. 

The data is primarily organized into experiments and experiment containers.  An experiment container represents a group of 
experiments imaged at the same cortical area, in the same animal, and and the same depth.  The individual experiments within 
an experiment container have different stimulus protocols, but cover the same field of view.  For example, the static grating
stimulus is presented in a different experiment than the natural scene stimulus.  

Because different stimuli may evoke responses from disjoint sets of cells, cells are segmented separately in each 
experiment. Segmented cells for experiments belonging to the same experiment container are then associated
with each other in a post-process. 

All traces for segmented cells in a single experiment are stored in the Neurodata Without Borders (NWB) format.
Traces include every cell's mean fluoresence trace, neuropil trace, and dF/F trace.  Code for extracting neuropil-corrected
traces is available in the SDK. 


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

For more information, please `read our technical whitepapers <help.alleninstitute.org/display/cam/Documentation>`_.


Overview of Visual Stimuli
--------------------------

All displayed stimuli were warped when displayed to help compensate for angular differences in square size when displayed on a flat monitor (e.g., the left and right edges of the display were magnified to compensate for this part of the monitor being further way from the mouse). The mapping of movie pixels to screen vertices is provided at:

    :py:meth:`allensdk.core.brain_observatory_nwb_data_set.warp_stimulus_coords`

**Locally sparse noise**

Locally sparse noise stimuli consist of ~11 black and/or white squares placed in random locations on a gray background. The stimulus consisted of a 16x28 array of pixels, 4-degrees on a side. White and black spots were distributed such that no two spots were within 20 degress of one another.

.. image:: /_static/locally_sparse_noise.png

**Natural scenes**

A variety of natural images were presented from the Berkeley Segmentation Dataset, the Hateren Natural Image Dataset, and the McGill Calibrated Colour Image Database. These include things like flowers, animals and outdoor scenes. Each image was contrast-normalized and presented in grayscale. Images were presented for 0.25 seconds, with no inter-image gray period, and were presented 50 times in random order. Blank sweeps were presented roughly once every 100 images.

.. image:: /_static/natural_scenes.png

**Natural movie**

Parts of the Orson Welles movie 'Touch of Evil' were shown. Popcorn was not provided. There were three movie clips, the first two being 30 seconds, and the third being 120 seconds. All were contrast-normalized and presented in grayscale at 30fps. Each clip was presented 10 times in a row with no inter-trial gray period.

.. image:: /_static/natural_movie.png

**Drifting grating**

Moving gratings with different orientation and temporal frequency were presented. 
Stimuli were full-screen static sinusoidal gratings at a single spatial frequency (0.04cpd) and contrast (80%). eight different orientations were presented (separated by 45 degrees) and five spatial freuqencies (0.02, 0.04, 0.08, 0.16, 0.32 cpd). Blank sweeps were presented roughly once every 20 gratings.

.. image:: /_static/animated_drifting_grating.gif

**Static grating**

Non-moving gratings with similar variation in orientation, spatial and temporal frequency were also shown.
Stimuli were full-screen static sinusoidal gratings at a single contrast (80%). Six different orientations were presented (separated by 30 degrees), five spatial freuqencies (0.02, 0.04, 0.08, 0.16, 0.32 cpd) and four phases (0, 0.25, 0.5, 0.75). Blank sweeps were presented roughly once every 25 gratings.

.. image:: /_static/static_250px_sfreq_0.04.png

**Spontaneous activity**

This stimulus was used to record activity from neurons when there was no visual stimulus being presented. The stimulus was 5 minutes of mean luminance gray.



   







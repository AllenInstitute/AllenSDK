Organization of a Visual Behavior NWB file
==========================================

The Allen Institute's Brain Observatory data sets include data from 2-photon microscope recordings including image segmentation, motion correction and fluorescence traces. 
These data are separated into two different types of files - files with just data about specimen behavior and files which include this _and_ optical physiology data.
It also includes information about the visual stimuli presented during the recording. This information is packaged in the `NWB format <http://www.nwb.org>`_. 
Here is a brief description how the data is stored in the NWB file.

Browsing the file
-----------------

The NWB format is a self-documenting file format that is designed to store a wide range of neurophysiology data. 
Structurally, NWB is nothing more than an HDF5 file that has data organized in a particular way.
For example, the format has a dedicated location for storing all data that is acquired during an experiment, and another place to store stimulus data that was presented. 
Data is usually stored in a time series, and each time series has information to document the type of data that it stores.
Importantly, all HDF5 tools and libraries are able to read the NWB file.

When using NWB files, the most important tool to become familiar with is HDFView. 
HDFView is an open-source tool that is published by the HDF Group. 
It allows you to browse the contents of an HDF5 (and NWB) file in an intuitive way, and also to view some of the data stored within it. 
HDFView can be downloaded at `www.hdfgroup.org <http://www.hdfgroup.org/products/java/hdfview/>`_.
It is *highly* recommended that anyone using an NWB file download and install this tool. 


NWB file organization
---------------------

Each NWB file is organized into seven sections. These sections store acquired data ('acquisition'), stimulus data ('stimulus'), general metadata about the experiment ('general'), experiment organization ('intervals'), processed data ('processing'), metadata about data types found in the NWB ('specifications'), and a free-form area for storing analysis data ('analysis'). 

.. figure:: /_static/behavior_ophys_nwb_fig-1.png

   **Figure 1. An NWB file, viewed in HDFView**

At the left of HDFView shows the organization of data in an NWB file.
It is most useful to imagine an HDF5 file as a container that stores many files and folders within it. 
Because of this structure, you can browse an HDF5 file, and an NWB file, in the same way that you browse the files and folders on your computer.
Folders can be opened by double-clicking on the folder icon.
The contents of a 'file' within HDF5, which called a dataset, can be seen by double-clicking it. 
An NWB file has seven top-level folders and five top-level datasets. 
In Figure 1, the contents of the 'session_description' dataset is displayed.
An NWB file is designed to store data from a single experimental session on a single animal. 
The `session_description` dataset provides a quick summary of the information available in the file.
The NWB specification, describing the format and how data is stored, can be downloaded `here <http://github.com/NeurodataWithoutBorders/specification/blob/master/version_1.0.3/nwb_file_format_specification_v1.0.3.pdf>`_.


Stimulus data
-------------

Within an NWB file, stimulus data is organized into 'templates', which are stimulus descriptions that can be used one or more times, and 'presentation', which stores the time-series data about what was presented and when. 

.. figure:: /_static/behavior_ophys_nwb_fig-2.png

   **Figure 2. Stimuli presented during the experiment**

In this experiment shown in Figure 2, there was only one stimulus presented - a set of natural images. 
When browsing the file, note the panel to the right. 
Clicking on a folder or dataset will cause information about that object to be displayed there. 
This will display a table of information related to that dataset, including comments, descriptions, and other relevant metadata. 
Double clicking one of these will further reveal the value stored inside.

Processing
----------

In the NWB file, the processing folder is designed to store the contents of the different levels of intermediate processing of data that are necessary to convert raw data into something that can be used for scientific analysis, as well as information about the behavior of the specimen during data collection. 
In the Allen Institute's NWB file, these levels of processing include motion correction of the acquired 2-photon movie frames, image segmentation into regions of interest, the fluorescence signal for each of these regions, and the dF/F signal that is useful for correlating cell activity with presented stimuli. 

This is also where you will find the main difference between the two types on NWB files produced for the Allen Institute Brain Observatory dataset. 
In the processing directory, you will notice behavior data information like licking, running, rewards, etc. 
If this is an NWB strictly containing behavior data, then that will be all. If the NWB file is one that includes optical physiology data as well then there wil also be a directory in processing called `ophys`, where you will find the intermediate processing data. 

.. figure:: /_static/behavior_ophys_nwb_fig-3.png

   **Figure 3. Location of dF/F data**


HDFView provides a very efficient way to browse the contents of an NWB file.

Below is a brief description and location for several types of data in the NWB file.

+-----------+-----------------------+------------------------------------------------------------------------------+----------------------------------------------+
| Category  | Data                  | Location                                                                     | SDK function(s)                              |
+===========+=======================+==============================================================================+==============================================+
| Metadata  | Cre line              | /general/subject/genotype                                                    |                                              |
+           +-----------------------+------------------------------------------------------------------------------+                                              |
|           | Imaging depth         | /general/metadata                                                            |                                              |
+           +-----------------------+------------------------------------------------------------------------------+                                              |
|           | Target structure      |                                                                              |                                              |
+           +-----------------------+------------------------------------------------------------------------------+                                              |
|           | Session type          | /general/task_parameters                                                     |                                              |
+-----------+-----------------------+------------------------------------------------------------------------------+----------------------------------------------+
| Stimulus  | Natural Images        | /stimulus/presentation/Natural_Images_Lum_Matched_set_training_2017.07.14    |                                              |
|           |                       |                                                                              |                                              |
+           +-----------------------+------------------------------------------------------------------------------+----------------------------------------------+
|           | Stimulus Distribution | /general/task_parameters                                                     |                                              |
|           |                       |                                                                              |                                              |
+-----------+-----------------------+------------------------------------------------------------------------------+----------------------------------------------+
| Processed | Motion correction     | /processing/ophys/ophys_motion_correction_x                                  |                                              |
| data      |                       | /processing/ophys/ophys_motion_correction_y                                  |                                              |
|           |                       |                                                                              |                                              |
+           +-----------------------+------------------------------------------------------------------------------+----------------------------------------------+
|           | Image segmentation    | /processing/ophys/image_segmentation                                         |                                              |
|           |                       |                                                                              |                                              |
+           +-----------------------+------------------------------------------------------------------------------+----------------------------------------------+
|           | Fluorescence          | /processing/ophys/corrected_fluorescence                                     |                                              |
|           |                       |                                                                              |                                              |
+           +-----------------------+------------------------------------------------------------------------------+----------------------------------------------+
|           | dF/F                  | /processing/ophys/dff                                                        |                                              |
+-----------+-----------------------+------------------------------------------------------------------------------+----------------------------------------------+



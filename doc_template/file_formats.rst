File Formats
============

Neurodata Without Borders
-------------------------

The electrophysiology data collected in the Allen Cell Types Database 
is stored in the `Neurodata Without Borders`_ (NWB) file format.
This format, created as part of the NWB_ initiative, is designed to store
a variety of neurophysiology data, including data from intra- and
extracellular electrophysiology experiments, optophysiology experiments,
as well as tracking and stimulus data.  It has a defined schema and metadata
labeling system designed so software tools can easily access contained data.

.. _Neurodata Without Borders: NWB_
.. _NWB: http://crcns.org/NWB/Overview
.. _NWB Github Repository: http://github.com/NeurodataWithoutBorders

Allen Cell Types Database NWB Files
-----------------------------------

The Allen SDK provides a basic Python class for extracting data from 
Allen Cell Types Database NWB files. These files store data from intracellular 
patch-clamp recordings. A stimulus current is presented to the cell and the cell's 
voltage response is recorded.  The file stores both stimulus and response for
several experimental trials, here called "sweeps."  The following code snippet
demonstrates how to extract a sweep's stimulus, response, sampling rate, 
and estimated spike times::

    from allensdk.core.nwb_data_set import NwbDataSet

    file_name = 'example.nwb'
    data_set = NwbDataSet(file_name)

    sweep_number = 61
    sweep_data = data_set.get_sweep(sweep_number)

    # spike times are in seconds relative to the start of the sweep
    spike_times = data_set.get_spike_times(sweep_number)

    # stimulus is a numpy array in amps
    stimulus = sweep_data['stimulus']

    # response is a numpy array in volts
    reponse = sweep_data['response']

    # sampling rate is in Hz
    sampling_rate = sweep_data['sampling_rate']
    
    # start/stop indices that exclude the experimental test pulse (if applicable)
    index_range = sweep_data['index_range']

HDF5 Overview
-------------

NWB is implemented in HDF5_.  HDF5 files provide a hierarchical data storage that mirrors the organization of a file system.  Just as a file system has directories and files, and HDF5 file has groups and datasets.  The best way to understand an HDF5 (and NWB) file is to open a data file in an HDF5 browser. HDFView_ is the recommended browser from the makers of HDF5.  

There are HDF5 manipulation libraries for many languages and platorms.  MATLAB and Python in particular have strong HDF5 support.  

.. _HDF5: https://hdfgroup.org/HDF5
.. _HDFView: https://hdfgroup.org/products/java/hdfview



    


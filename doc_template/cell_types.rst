Cell Types
==========

The Allen Cell Types data set is a database of neuronal cell types based on multimodal characterization
of single cells to enable data-driven approaches to classification and is fully integrated with other
Allen Brain Atlas resources.  The database currently includes:

    * **electrophysiology**: whole cell current clamp recordings made from Cre-positive neurons
    * **morphology**: 3D bright-field images of the complete structure of neurons from the visual cortex

Cell Types API Access
---------------------

The :py:mod:`~allensdk.api.queries.cell_types_api.CellTypesApi` class provides a Python interface for downloading data
in the Allen Cell Types Database.  The following example demonstrates how to download meta data for
all cells with 3D reconstructions, then download the reconstruction and electrophysiology recordings
for one of those cells::

    from allensdk.api.queries.cell_types_api import CellTypesApi

    ct = CellTypesApi()

    # a list of dictionaries containing metadata for cells
    # that have morphological reconstructions
    cells = ct.list_cells(require_reconstruction=True)

    # download the electrophysiology data for one cell
    ct.save_ephys_data(cells[0]['id'], 'example.nwb')

    # download the reconstruction for the same cell
    ct.save_reconstruction(cells[0]['id'], 'example.swc')

File Formats
------------

This section provides a short description of the file formats used for Allen Cell Types data.

Morphology SWC Files
--------------------

Morphological neuron reconstructions are available for download as SWC files.  The SWC file format is a white-space delimited text file with a standard set of headers.  The file lists a set of 3D neuronal compartments, each of which has:

====== ========= ===========================
Column Data Type Description
====== ========= ===========================
id     string    compartment ID
type   integer   compartment type
x      float     3D compartment position (x)
y      float     3D compartment position (y)
z      float     3D compartment position (z)
radius float     compartment radius
parent string    parent compartment ID
====== ========= ===========================

Comment lines begin with a '#'.  Reconstructions in the Allen Cell Types Database can contain the following compartment types:

==== ===============
Type Description
==== ===============
0    unknown
1    soma
2    axon
3    basal dendrite
4    apical dendrite
==== ===============

The Allen SDK comes with a :py:mod:`~allensdk.core.swc` Python module that provides helper functions and classes for manipulating SWC files.  Consider the following example::

    import allensdk.core.swc as swc

    file_name = 'example.swc'
    morphology = swc.read_swc(file_name)
    
    # subsample the morphology 3x. root, soma, junctions, and the first child of the root are preserved.
    sparse_morphology = morphology.sparsify(3)

    # compartments in the order that they were specified in the file
    compartment_list = sparse_morphology.compartment_list

    # a dictionary of compartments indexed by compartment id
    compartments_by_id = sparse_morphology.compartment_index

    # the root compartment (usually the soma)
    root = morphology.root

    # all compartments are dictionaries of compartment properties
    # compartments also keep track of ids of their children
    for child_id in root['children']:
        child = compartments_by_id[child_id]
        print child['x'], child['y'], child['z'], child['radius']
    

Neurodata Without Borders
-------------------------

The electrophysiology data collected in the Allen Cell Types Database 
is stored in the `Neurodata Without Borders`_ (NWB) file format.
This format, created as part of the `NWB initiative`_, is designed to store
a variety of neurophysiology data, including data from intra- and
extracellular electrophysiology experiments, optophysiology experiments,
as well as tracking and stimulus data.  It has a defined schema and metadata
labeling system designed so software tools can easily access contained data.

.. _Neurodata Without Borders: http://neurodatawithoutborders.github.io/
.. _NWB initiative: http://crcns.org/NWB/Overview

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
+++++++++++++

NWB is implemented in HDF5_.  HDF5 files provide a hierarchical data storage that mirrors the organization of a file system.  Just as a file system has directories and files, and HDF5 file has groups and datasets.  The best way to understand an HDF5 (and NWB) file is to open a data file in an HDF5 browser. HDFView_ is the recommended browser from the makers of HDF5.  

There are HDF5 manipulation libraries for many languages and platorms.  MATLAB and Python in particular have strong HDF5 support.  

.. _HDF5: https://hdfgroup.org/HDF5
.. _HDFView: https://hdfgroup.org/products/java/hdfview

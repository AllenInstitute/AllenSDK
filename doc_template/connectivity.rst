Mouse Connectivity
==================

The Allen Mouse Brain Connectivity Atlas consists of high-resolution images of axonal projections targeting different anatomic regions or various cell types using Cre-dependent specimens. Each data set is processed through an informatics data analysis pipeline to obtain spatially mapped quantified projection information.  

This page describes how to use the SDK to access experimental projection data and metadata.  For more information, 
please visit the Connectivity Atlas `home page <http://connectivity.brain-map.org/>`_ and the 
`API documentation <http://help.brain-map.org/display/mouseconnectivity/ALLEN+Mouse+Brain+Connectivity+Atlas>`_

Mouse Connectivity API
----------------------

The :py:class:`~allensdk.api.queries.mouse_connectivity_api.MouseConnectivityApi` class provides a Python interface 
for downloading data in the Allen Mouse Brain Connectivity Atlas.  The following example demonstrates how to download 
meta data for all wild-type mice and the projection signal density for one cell:

.. literalinclude:: examples/connectivity_ex1.py

Mouse Connectivity Cache
------------------------

File Formats
------------


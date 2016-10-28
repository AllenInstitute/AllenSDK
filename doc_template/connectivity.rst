Mouse Connectivity
==================

The Allen Mouse Brain Connectivity Atlas consists of high-resolution images of axonal projections targeting different anatomic regions or various cell types using Cre-dependent specimens. Each data set is processed through an informatics data analysis pipeline to obtain spatially mapped quantified projection information.  

This page describes how to use the SDK to access experimental projection data and metadata.  For more information, 
please visit the Connectivity Atlas `home page <http://connectivity.brain-map.org/>`_ and the 
`API documentation <http://help.brain-map.org/display/mouseconnectivity/ALLEN+Mouse+Brain+Connectivity+Atlas>`_


Code Examples
-------------

The Mouse Connectivity `Jupyter notebook <_static/examples/nb/mouse_connectivity.html>`_ has many code samples to help get
started with analysis:

    - `Download experimental metadata by injection structure and transgenic line <_static/examples/nb/mouse_connectivity.html#Mouse-Connectivity>`_
    - `Download projection signal statistics at a structure level <_static/examples/nb/mouse_connectivity.html#Structure-Signal-Unionization>`_
    - `Build a structure-to-structure matrix of projection signal values <_static/examples/nb/mouse_connectivity.html#Generating-a-Projection-Matrix>`_
    - `Download and visualized gridded projection signal volumes <_static/examples/nb/mouse_connectivity.html#Manipulating-Grid-Data>`_
      
      
Mouse Connectivity API
----------------------

The :py:class:`~allensdk.api.queries.mouse_connectivity_api.MouseConnectivityApi` class provides a Python interface 
for downloading data in the Allen Mouse Brain Connectivity Atlas.  The following example demonstrates how to download 
meta data for all wild-type mice and the projection signal density for one experiment:

.. literalinclude:: _static/examples/connectivity_ex1.py


Mouse Connectivity Cache
------------------------

The :py:class:`~allensdk.core.mouse_connectivity_cache.MouseConnectivityCache` class saves all of the data you can download 
via the :py:class:`~allensdk.api.queries.mouse_connectivity_api.MouseConenctivityApi` in well known locations so that you 
don't have to think about file names and directories.  It also takes care of knowing if you've already downloaded some files 
and reads them from disk instead of downloading them again.  The following example demonstrates how to download meta data for
all experiments with injections in the isocortex and download the projetion density volume for one of them:

.. literalinclude:: _static/examples/connectivity_ex2.py


Structure-Level Projection Data
-------------------------------

All AAV projection signal in the Allen Mouse Connectivity Atlas has been registered to the expert-annotated Common Coordinate Framework
and summarized to structures in the adult mouse structure ontology.  This data is available for download and is described
in more detail on the :doc:`structure unionizes page </unionizes>`.


File Formats
------------

This section provides a short description of the file formats used for data in the Allen Mouse Connectivity Atlas.


NRRD Files
++++++++++

All of the volumetric data in the connectivity atlas are stored as 
`NRRD (Nearly Raw Raster Data) <http://teem.sourceforge.net/nrrd/>`_ files. A NRRD file 
consists of a short ASCII header followed by a binary array of data values.  

To read these in Python, we recommend the `pynrrd package <https://github.com/mhe/pynrrd>`_. 
Usage is straightforward:

.. literalinclude:: _static/examples/connectivity_ex3.py





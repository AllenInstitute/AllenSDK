Install Guide
=============
This guide is a resource for using the Allen SDK package.
It is maintained by the `Allen Institute for Brain Science <http://www.alleninstitute.org/>`_.

The Allen SDK was developed and tested with Python 2.7.13 and Python 3.6.4, installed
as part of `Anaconda Python <https://store.continuum.io/cshop/anaconda/>`_ distribution 
version `4.3.13 <http://repo.continuum.io/archive/index.html>`_.  We do not guarantee
consistent behavior with other Python versions.  

Quick Start Using Pip
---------------------

First ensure you have `pip <http://pypi.python.org/pypi/pip>`_ installed.
It is included with the Anaconda distribution.

::

    pip install allensdk


To uninstall the SDK::

    pip uninstall allensdk

Other Distribution Formats
--------------------------

The Allen SDK is also available from the Github source repository.

Required Dependencies
---------------------

 * `NumPy <http://wiki.scipy.org/Tentative_NumPy_Tutorial>`_
 * `SciPy <http://www.scipy.org/>`_
 * `matplotlib <http://matplotlib.org/>`_
 * `h5py <http://www.h5py.org>`_
 * `pandas <http://pandas.pydata.org>`_
 * `pynrrd <http://pypi.python.org/pypi/pynrrd>`_
 * `Jinja2 <http://jinja.pocoo.org>`_

Optional Dependencies
---------------------

 * `pytest <http://pytest.org/latest>`_
 * `coverage <http://nedbatchelder.com/code/coverage>`_

Installation with Docker (Optional)
-----------------------------------

`Docker <http://www.docker.com/>`_ is an open-source technology
for building and deploying applications with a consistent environment
including required dependencies.
The AllenSDK is not distributed as a Docker image, but
example Dockerfiles are available.

 #. Ensure you have Docker installed.

 #. Use Docker to build one of the images.
 
     Anaconda::

         docker pull alleninstitute/allensdk
 
     Other docker configurations are also available under docker directory in the source repository.
 
 #. Run the docker image::
 
     docker run -i -t -p 8888:8888 -v /data:/data alleninstitute/allensdk /bin/bash
     cd allensdk
     make test
 
 #. Start a Jupyter Notebook::
 
     cd allensdk/doc_template/examples/nb
     jupyter-notebook --ip=* --no-browser
     

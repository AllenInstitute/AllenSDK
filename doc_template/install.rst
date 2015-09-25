Install Guide
=============
This guide is a resource for using the Allen SDK package.
It is maintained by the `Allen Institute for Brain Science <http://www.alleninstitute.org/>`_.

The Allen SDK was developed and tested with Python 2.7.8, installed
as part of `Anaconda Python <https://store.continuum.io/cshop/anaconda/>`_ distribution 
version `2.1.0 <http://repo.continuum.io/archive/index.html>`_.  We do not guarantee
consistent behavior with other Python versions.  

Quick Start Using Pip
---------------------

First ensure you have `pip <http://pypi.python.org/pypi/pip>`_ installed.
To install the Allen SDK for a single user::

    pip install |tgz_url| --user


.. note::

    On Windows use the options '--process-dependency-links --trusted-host github.com'
    to install the correct version of the pynrrd dependency. 



Uninstalling the SDK is simple with pip::

    pip uninstall allensdk

Other Distribution Formats
--------------------------

The Allen SDK is also available from the source repository or as a downloadable .zip or .tar.gz archive.
The package can also be `installed from these formats <https://packaging.python.org/en/latest/installing.html>`_.

.. include:: links.rst


Required Dependencies
---------------------

 * `NumPy <http://wiki.scipy.org/Tentative_NumPy_Tutorial>`_
 * `SciPy <http://www.scipy.org/>`_


Optional Dependencies
---------------------

 * `nose <https://nose.readthedocs.org/en/latest>`_ is nicer testing for python
 * `coverage <http://nedbatchelder.com/code/coverage>`_
 * `matplotlib <http://matplotlib.org/>`_
 * `h5py <http://www.h5py.org>`_
 * `pandas <http://pandas.pydata.org>`_
 * `pynrrd <http://pypi.python.org/pypi/pynrrd>`_

Installation with Docker (Optional)
-----------------------------------

`Docker <http://www.docker.com/>`_ is an open-source technology
for building and deploying applications with a consistent environment
including required dependencies.
The AllenSDK is not distributed as a Docker image, but
example Dockerfiles are available.

 #. Ensure you have Docker installed.

 #. Download one of the example Docker files:
     * :download:`Ubuntu Standalone <./_static/examples/docker/Dockerfile.ubuntu>`.
     * :download:`Neural Ensemble combined simulators <./_static/examples/docker/Dockerfile.neuralensemble>`.
     * :download:`Neural Ensemble combined simulators with X11 <./_static/examples/docker/Dockerfile.neuralensemblex>`.

 #. Use Docker to build the image::
 
     mkdir docker_build
     cp Dockerfile.ubuntu docker_build/Dockerfile
     cd docker_build
     docker build --tag alleninstitute/allensdk:ubuntu .
     
 #. Run the docker image::
 
     docker run -it -v /data:/data alleninstitute/allensdk:ubuntu /bin/bash


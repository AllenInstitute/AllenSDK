Install Guide
=============
This guide is a resource for using the Allen SDK package.
It is maintained by the `Allen Institute for Brain Science <http://www.alleninstitute.org/>`_.

Quick Start Install Using Pip (Linux)
-------------------------------------

 #. Pip install for a single user.
    ::
    
        pip install http://bamboo.corp.alleninstitute.org/browse/MAT-AWBS/latestSuccessful/artifact/shared/tgz/allensdk-|version|.tar.gz --user


Uninstall
---------

 #. Simply use pip.
    ::
    
        pip uninstall allensdk


Docker Installation
-------------------

 #. Download one of the example `Docker <http://www.docker.com/>`_ files:
     * :download:`Ubuntu Standalone <./examples/docker/Dockerfile.ubuntu>`.
     * :download:`BrainScales combined Neural-Networks <./examples/docker/Dockerfile.brainscales>`.
 #. Use Docker to build the image:
     ::
     
         mkdir work
         cp Dockerfile.ubuntu work
         cd work
         docker build --tag alleninstitute/allensdk -f Dockerfile.ubuntu .
         
 #. Run the docker image:
     ::
     
         docker run -it -v /data:/data alleninstitute/allensdk /bin/bash


Other Distribution Formats
--------------------------

 .. include:: links.rst

 		 
Required Dependencies
---------------------

 * `NumPy <http://wiki.scipy.org/Tentative_NumPy_Tutorial>`_
 * `SciPy <http://www.scipy.org/>`_
 * `MatPlotLib <http://matplotlib.org/>`_


Optional Dependencies
---------------------

 * `nose <https://nose.readthedocs.org/en/latest>`_ is nicer testing for python
 * `coverage <http://nedbatchelder.com/code/coverage>`_
 
	
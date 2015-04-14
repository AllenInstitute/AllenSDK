User Guide
==========
This guide is a resource for using the Allen Wrench package.
It is maintained by the `Allen Institute for Brain Science <http://www.alleninstitute.org/>`_.

Quick Start Install Using Pip (Linux)
-------------------------------------

 #. Pip install for a single user.
    ::
        pip install http://bamboo.corp.alleninstitute.org/browse/MAT-AWBS/latestSuccessful/artifact/shared/tgz/allensdk-|version|.tar.gz --user


Install Using Setup Tools
-------------------------

 #. Download the distribution.
    ::
        wget http://bamboo.corp.alleninstitute.org/browse/MAT-AWBS/latestSuccessful/artifact/shared/tgz/allensdk-|version|.tar.gz
 #. Unpack the distribution.
    ::
        tar xvzf allensdk-|version|.tar.gz
 #. Install using setuptools
    ::
        cd allensdk-|version|
        python setup.py install --user
        
Uninstall
---------

 #. Simply use pip.
    ::
        pip uninstall allensdk
       
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

 * `NEURON <http://www.neuron.yale.edu/neuron>`_
 * `pandas <http://pandas.pydata.org>`_ and `pytables <http://www.pytables.org/moin>`_ for loading and saving configuration tables. 
 * `mpi4py <http://mpi4pi.scipy.org>`_ is a message passing interface for distributed processing
 * `nose <https://nose.readthedocs.org/en/latest>`_ is nicer testing for python
 
	
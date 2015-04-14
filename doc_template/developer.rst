Developer Guide
===============
This guide is a resource for contributing to the Allen SDK library.  It is maintained by the `Allen Institute for Brain Science <http://www.alleninstitute.org/>`_.

Quick Start (Linux)
-------------------

 #. Download the source distribution.
    ::
        wget http://bamboo.corp.alleninstitute.org/browse/MAT-AWBS/latestSuccessful/artifact/shared/tgz/allensdk-|version|.tar.gz
 #. Unpack the source code.
    ::
        tar -x -v --ungzip -f allensdk-|version|.tar.gz
 #. Install using setuptools
    ::
        cd allensdk-|version|
        python setup.py install
 #. Run the tests with coverage
    :: 
    	python setup.py nosetests
        
 #. Rebuild from source
    ::
        make clean doc sdist

Other Distribution Formats
--------------------------

 .. include:: links.rst

WIKI
----
 * `Allen Institute Internal Confluence Page <http://confluence.corp.alleninstitute.org/display/IT/LIMS+Support+for+Models>`_
 
Source Control
--------------

 * `Allen Institute Internal Stash Server <http://stash.corp.alleninstitute.org/projects/INF/repos/allensdk/browse/>`_
 
Build Server
------------

 * `Allen Institute Internal Bamboo Server <http://http://bamboo.corp.alleninstitute.org/browse/IFR-Allen SDK/>`_
 * `Allen SDK / Biophys Sim Combined build <http://bamboo.corp.alleninstitute.org/browse/MAT-AWBS/>`_

 Unit test results and code coverage reports can be found on here.

 		
Editing Documentation
---------------------

    Static documentation is in .rst files under the doc_template directory.  Add your new file (without the .rst extension) to the toclist in index.rst.
    The document syntax is `reStructuredText <http://sphinx-doc.org/rest.html#rst-primer>`_ and is rebuilt by the Makefile ('make clean doc').
    The generated html documentation can be found at doc/index.html


    For API docstring documentation, refer to `PEP-0257 <http://www.python.org/dev/peps/pep-0257>`_ for conventions.
    Please document all important packages, classes and methods.


Quick Links
-----------
 * `Python Testing: Nose Introduction <http://pythontesting.net/framework/nose/nose-introduction>`_
 * `Sphinx Tutorial <http://sphinx-doc.org/tutorial.html>`_
 * `An Example PyPi Project <http://pythonhosted.org/an_example_pypi_project/_downloads/an_example_pypi_project.pdf>`_
 * `Easy and Beautiful Documentation with Sphinx <https://www.ibm.com/developerworks/library/os-spinx-documentation>`_
 * `T+1: Some Notes on Nosetests and Coverage <http://blog.tplus1.com/blog/2009/05/13/some-notes-on-nosetests-and-coverage>`_
 * `Documenting matplotlib <http://matplotlib.org/devel/documenting_mpl.html>`_

 
Required Dependencies
---------------------

 * `NumPy <http://wiki.scipy.org/Tentative_NumPy_Tutorial>`_
 * `SciPy <http://www.scipy.org/>`_
 * `MatPlotLib <http://matplotlib.org/>`_
 * `lxml` <http://lxml.de>`_
 * `json` <http://docs.python.org/2/library/json.html>`_

	
Optional Dependencies
---------------------

 * `nose <https://nose.readthedocs.org/en/latest>`_ is nicer testing for python
 * `coverage <http://nedbatchelder.com/code/coverage>`_
 * `mpi4py <http://mpi4pi.scipy.org>`_ is a message passing interface for distributed processing
	
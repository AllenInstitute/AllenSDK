Install Guide
=============
This guide is a resource for using the Allen SDK package.
It is maintained by the `Allen Institute for Brain Science <http://www.alleninstitute.org/>`_.

.. ATTENTION::
    As of October 2019, we have dropped Python 2 support. The Allen SDK is developed and tested with Python 3.6 and 3.7. We do not guarantee consistent behavior with other Python versions.


Quick Start Using Anaconda
--------------------------
 #. From the `Anaconda downloads page <https://www.anaconda.com/products/individual>`_, download the Python 3.7 version for your operating system and run the installer.

 #. After the installation is complete, open up a terminal (in Windows open Anaconda3 Command Prompt).

 #. Install the AllenSDK using PIP::

     pip install allensdk

 #. Download one of our many `Jupyter Notebook examples <https://allensdk.readthedocs.io/en/latest/examples.html>`_ to a new folder.

 #. In your terminal, navigate to the directory where you downloaded the Jupyter Notebook example and run the following command::

     jupyter notebook

 #. Your browser should open and you should see the Jupyter Notebook example. Enjoy using the Allen SDK!

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

Installation with Docker (Optional)
-----------------------------------

`Docker <http://www.docker.com/>`_ is an open-source technology
for building and deploying applications with a consistent environment
including required dependencies.
The AllenSDK is not distributed as a Docker image, but
example Dockerfiles are available.

 #. Ensure you have Docker installed.

 #. Use Docker to build the image::

     docker pull alleninstitute/allensdk
 
    Other docker configurations are also available under docker directory in the source repository.
 
 #. Run the docker image::
 
     docker run -i -t -p 8888:8888 alleninstitute/allensdk /bin/bash

 #. Run the SDK tests::

     cd allensdk
     make test
 
 #. Start a Jupyter Notebook::
 
     cd allensdk/doc_template/examples_root/examples/nb
     jupyter notebook --ip=* --no-browser --allow-root

    Using the browser on your host machine, navigate to the path provided by the output from the jupyter notebook command.
     

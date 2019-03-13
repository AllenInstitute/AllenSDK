Internal
========

This subpackage contains code that depends on Allen Institute for Brain Science-specific resources, such as tools for accessing our internal databases. The code here should only be expected to run if you are at the Allen Institute!

The dependencies specific to this subpackage are listed in internal_requirements.txt. There are some of additional dependencies with complicated installation requirements that we don't list in internal_requirements.txt. These are:
- neuron : we build this from source by:
    ```
    ./configure --without-iv --prefix={install location} --with-nrnpython={path to your python}
    make
    make install
    ```
- mpi4py : we conda install this like `conda install mpi4py`
- opencv : we conda install this like `conda install -c conda-forge opencv`
A Simple Script
===============

Reading the Description File
----------------------------

A simple .conf file simply has a header and the model file.

bps.conf
::

    model_file: my_model.json

my_model.py
::

    from allen_wrench.model.biophys_sim.config import Config
    
    description = Config().load('bps.conf')


Initialize HOC
--------------

::

    from allen_wrench.model.biophys_sim.neuron.hoc_utils import HocUtils
    hoc_utils = HocUtils(description.manifest)


Create a Cell 
-------------

::

    from 


TODO

Stimulus
--------




Recording and Saving Results
----------------------------

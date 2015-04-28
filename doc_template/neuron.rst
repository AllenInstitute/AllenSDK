NEURON Models
=============
This guide is a resource for creating a biophysical model using
NEURON and the Allen SDK package.

Create the Directory Structure
------------------------------

 #. Create a directory for the model data and configuration files.
    ::
        my_biophysical_project
        |-- bps.conf
        |-- my_model_description.json
        |-- my_stimulus.nwb
        |-- manifest.json
        |-- morphology
        |   |-- my_cell.swc
        |   `-- another_cell.swc
        |-- modfiles
        |   `-- my_modfile.mod
        |-- templates
        |   |-- my_simulator.py
        |   `-- my_utilities.py
        `---work
        
    Your NEURON NMODL (.mod) files go in the modfiles directory.
    Your .swc morphologies go in the morphology directory.
    The remaining files are specific to Allen SDK and are described below.
    
 #. Create your model description.
    
    The :doc:`model description </model_description>` JSON file describes the topology of your
    neural model, including parameters for individual neurons and their connections.
    
 #. Create your application template.
    Currently applications are written using the Python programming language.
    At minimum, it consist of a top level script that 
    reads the model description, runs the simulation and produces output.
    Often the template directory also contains user-written functions
    that provide additional functionality.
    
  #. Create a conf file.
     If you choose to use the :doc:`bps utility </bps>`, a .conf file holds the options that you would
     pass to your template script, including the name of the model description file,
     the input file, and other application settings.
     
Run Your Simulation
-------------------

  #. Compile the NMODL files
     If you are using a template script, simply use NEURON's nrnivmodl command:
     ::
         nrnivmodl modfiles
     or using bps:
     ::
         bps nrnivmodl bps.conf
  #. Run your simulation
     If you are using a template script, pass the model description and stimulus files
     to your script:
     ::
         template/my_simulator.py my_model_description.json my_stimulus.nwb
     or using bps:
     ::
         bps run_simple bps.conf
     The output from your simulation and any intermediate files should go in the work directory.


Further Reading
---------------

 * `NEURON <http://www.neuron.yale.edu/neuron>`_
 * `Python <https://www.python.org/>`_
 * `JSON <http://www.w3schools.com/json/>`_
 * `pandas <http://pandas.pydata.org>`_ and `pytables <http://www.pytables.org/moin>`_ for loading and saving configuration tables. 

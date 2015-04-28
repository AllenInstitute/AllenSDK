Perisomatic Biophysical Models
==============================

Prerequisites
-------------

You must have pyNEURON and Allen SDK installed.

The Allen Institute data is generated using
NEURON `version v7.3.ansi-1078 <http://www.neuron.yale.edu/ftp/neuron/versions/v7.3/v7.3.ansi-1078>`_.
Instructions for `compiling NEURON <http://www.neuron.yale.edu/neuron/download/compile_linux>`_ with the Python interpreter 
are available from the `NEURON team <http://www.neuron.yale.edu/neuron/>`_.

Allen SDK is compatible with Python version 2.7, including the Anaconda distribution.

Instructions for `Docker installation <./install.html#docker-installation>`_ are also available.


Retrieving Data from the Allen Institute
----------------------------------------

This may be done programmatically
::

    from allensdk.api.queries.single_cell_biophysical import \
        SingleCellBiophysical
    
    scb = SingleCellBiophysical('http://api.brain-map.org')
    scb.cache_stimulus = False # change to True to download the stimulus file
    neuronal_model_id = 464137111    # get this from the web site as below
    scb.cache_data(neuronal_model_id, working_directory='neuronal_model')

or you can download the data manually from the web site:

    #. Go to `http://tcelltypes.corp.alleninstitute.org <http://tcelltypes.corp.alleninstitute.org>`_
    #. Use the Filters, Cell Location and Cell Feature Filters to narrow your results.
    #. Click on a Cell Summary to view the Mouse Experiment Electrophysiology.
    #. Click "show model response" and select "Biophysical - perisomatic".
    #. Scroll down and click the Biophysical - perisomatic"download model" link.

More help can be found in the
`online help <http://help.brain-map.org/display/celltypes/Allen+Cell+Types+Database>`_
for the ALLEN **Cell Types Database** web application.


Directory Structure
-------------------

The structure of the directory created looks like this.
It includes stimulus files, model parameters, morphology, cellular mechanisms
and application configuration.
::

    neuronal_model
    |-- manifest.json
    |-- 169248.04.02.01_fit.json
    |-- Nr5a1-Cre_Ai14_IVSCC_-169248.04.02.01.nwb
    |-- Nr5a1-Cre_Ai14_IVSCC_-169248.04.02.01_403165543_m.swc
    |-- modfiles
    |   |--CaDynamics.mod
    |   |--Ca_HVA.mod
    |   |--Ca_LVA.mod
    |   |--Ih.mod
    |   `--...etc.
    |
    |--x86_64
    `---work


Running the Simulation
--------------------------------------------

::

    cd neuronal_model
    nrnivmodl ./modfiles
    python -m allensdk.model.single_cell_biophysical.runner manifest.json


Simulation Main Loop
--------------------

The top level script is in the
:py:meth:`~allensdk.model.single_cell_biophysical.runner.run`
method of the :py:mod:`allensdk.model.single_cell_biophysical.runner`
module.

The first step is to configure NEURON based on the configuration file.
The configuration file was read in from the command line at the very bottom of the script.
::

    # configure NEURON
    utils = Utils(description)
    h = utils.h

The next step is to get the path of the morphology file and pass it to NEURON.
::

    # configure model
    manifest = description.manifest
    morphology_path = description.manifest.get_path('MORPHOLOGY')
    utils.generate_morphology(morphology_path.encode('ascii', 'ignore'))
    utils.load_cell_parameters()

Then read the stimulus and recording configuration and configure NEURON
::

    # configure stimulus and recording
    stimulus_path = description.manifest.get_path('stimulus_path')
    nwb_out_path = manifest.get_path("output")
    output = NwbDataSet(nwb_out_path)
    run_params = description.data['runs'][0]
    sweeps = run_params['sweeps']
    junction_potential = description.data['fitting'][0]['junction_potential']
    mV = 1.0e-3

Loop through the stimulus sweeps and write the output.
::

    # run sweeps
    for sweep in sweeps:
        utils.setup_iclamp(stimulus_path, sweep=sweep)
        vec = utils.record_values()
        
        h.finitialize()
        h.run()
        
        # write to an NWB File
        output_data = (numpy.array(vec['v']) - junction_potential) * mV
        output.set_sweep(sweep, None, output_data)


Customized Utilities
--------------------

Much of the code in the single cell example is not core Allen SDK code.
The runner.py script largely reads the configuration file and calls into
methods in the :py:class:`~allensdk.model.single_cell_biophysical.utils.Utils` class.
Utils is a subclass of the :py:class:`~allensdk.model.biophys_sim.neuron.hoc_utils.HocUtils`
class, which provides access to objects in the NEURON package.

::

    from allensdk.model.biophys_sim.neuron.hoc_utils import HocUtils
    
    .....
    
    class Utils(HocUtils):
    .....
    
        def __init__(self, description):
            super(Utils, self).__init__(description)
    ....


The various methods called by the runner.script are implemented here, including:
:py:meth:`~allensdk.model.single_cell_biophysical.utils.Utils.generate_morphology`,
:py:meth:`~allensdk.model.single_cell_biophysical.utils.Utils.load_cell_parameters`,
:py:meth:`~allensdk.model.single_cell_biophysical.utils.Utils.setup_iclamp`,
:py:meth:`~allensdk.model.single_cell_biophysical.utils.Utils.read_stimulus`
and
:py:meth:`~allensdk.model.single_cell_biophysical.utils.Utils.record_values`.
Other applications are free to implement their own subclasses of HocUtils as needed.


Simple Example
--------------

A :download:`minimal example (simple_example.tgz)<./examples/simple_example.tgz>`
is available to use as a starting point for your own projects.


Multicell Example
-----------------

A :download:`multicell example (multicell_example.tgz)<./examples/multicell_example.tgz>`
is available to use as a starting point for your own projects.


Selecting a Specific Sweep
--------------------------

The sweeps are listed in manifest.json.
You can remove all of the sweep numbers that you do not want run.


Exporting Output to Text Format
-------------------------------


Single Cell Biophysical Models
==============================

Running a Single Cell Biophysical Simulation
--------------------------------------------

::
    nrnivmodl .
    python -m allensdk.model.single_cell_biophysical.runner manifest.json


Retrieving Data from the Allen Institute
----------------------------------------

This may be done programmatically
::

    from allensdk.wh_client.queries.single_cell_biophysical import \
        SingleCellBiophysical
    
    scb = SingleCellBiophysical('http://iwarehouse.corp.alleninstitute.org')  ### TODO REMOVE INTERNAL URL
    scb.cache_data(464137111, working_directory=<dir>)

or you can download the data manually from the web site.

Simulation Main Loop
--------------------

From allensdk.model.single_cell_biophysical.runner#run():

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
    orcas_out_path = manifest.get_path("output_orca")
    output = OrcaDataSet(orcas_out_path)
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
        
        # write to an Orca File
        output_data = (numpy.array(vec['v']) - junction_potential) * mV
        output.set_sweep(sweep, None, output_data)


Selecting a Specific Sweep
--------------------------


Exporting Output to Text Format
-------------------------------


Directory Structure
-------------------

A typical directory created to reproduce a neuronal model run looks like this.
It includes stimulus files, model parameters, morphology, cellular mechanisms
and application configuration.
::

    neuronal_model_run_123456
    |-- manifest.json
    |-- 169248.04.02.01_fit.json
    |-- Nr5a1-Cre_Ai14_IVSCC_-169248.04.02.01.orca
    |-- Nr5a1-Cre_Ai14_IVSCC_-169248.04.02.01_403165543_m.swc
    |-- modfiles
    |   |--CaDynamics.mod
    |   |--Ca_HVA.mod
    |   |--Ca_LVA.mod
    |   |--Ih.mod
    |   `--...etc.
    
    |--x86_64
    `---work
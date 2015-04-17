Single Cell Biophysical Models
==============================

Running a Single Cell Biophysical Simulation
--------------------------------------------


Retrieving Data from the Allen Institute
----------------------------------------

This may be done programmatically
::

    from allensdk.wh_client.queries.single_cell_biophysical import \
        SingleCellBiophysical
    
    scb = SingleCellBiophysical('http://iwarehouse.corp.alleninstitute.org/api/v2/data')
    print scb.build_rma_url_biophysical_neuronal_model_run(464137111)

or you can download the data from the web site.

Simulation Main Loop
--------------------

From allensdk.model.single_cell_biophysical.runner#run():
::

    # configure NEURON
    utils = Utils(description)
    h = utils.h

::

    # configure model
    manifest = description.manifest
    morphology_path = description.manifest.get_path('MORPHOLOGY')
    utils.generate_morphology(morphology_path.encode('ascii', 'ignore'))
    utils.load_cell_parameters()

::

    # configure stimulus and recording
    stimulus_path = description.manifest.get_path('stimulus_path')
    orcas_out_path = manifest.get_path("output_orca")
    output = OrcaDataSet(orcas_out_path)
    run_params = description.data['runs'][0]
    sweeps = run_params['sweeps']
    junction_potential = description.data['fitting'][0]['junction_potential']
    mV = 1.0e-3

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

The directory for the model data, configuration files and output files.
::

    neuronal_model_run_123456
    |-- manifest.json
    |-- ABCD_123456.orca
    |-- morphology.swc
    |-- modfiles
    |   |--xyz.mod
    |   `--...etc.
    |--x86_64
    `---work
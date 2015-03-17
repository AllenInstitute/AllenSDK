from allen_wrench.model.biophys_sim.config import Config
import allen_wrench.model.single_cell_biophysical.model_load as ml
from allen_wrench.model.single_cell_biophysical.iclamp_stimulus import IclampStimulus
from load_cell_parameters import load_cell_parameters


import numpy as np
from allen_wrench.core.orca_data_set import OrcaDataSet

def run(description):
    manifest = description.manifest

    from neuron import h
    h.load_file("stdgui.hoc")
    h.load_file("import3d.hoc")
    
    morphology_path = description.manifest.get_path('MORPHOLOGY')
    ml.generate_morphology(morphology_path.encode('ascii', 'ignore'))
    load_cell_parameters(h,
                         description.data['passive'][0],
                         description.data['genome'],
                         description.data['conditions'][0])
    ml.setup_conditions(h, description.data['conditions'][0])
    
    orcas_out_path = manifest.get_path("output_orca")
    output = OrcaDataSet(orcas_out_path)
    
    run_params = description.data['runs'][0]
    sweeps = run_params['sweeps']
    
    stimulus_path = description.manifest.get_path('stimulus_path')
    
    for sweep in sweeps:
        iclamp = IclampStimulus(h)
        iclamp.setup_instance(stimulus_path, sweep=sweep)
    
        vec = ml.record_values()
    
        h.finitialize()
        h.run()
        
        # And to an Orca File
        
        excess_data = 5
        output_data = np.array(vec['v'])[0:-excess_data] * 1.0e-3
        output.set_sweep(sweep, None, output_data)


if '__main__' == __name__:
    manifest_json_path = 'manifest.json'
    
    description = Config().load(manifest_json_path)
    
    run(description)


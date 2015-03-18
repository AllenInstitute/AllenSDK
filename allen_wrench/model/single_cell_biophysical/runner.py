from allen_wrench.model.biophys_sim.config import Config
import allen_wrench.model.single_cell_biophysical.model_load as ml
from allen_wrench.model.single_cell_biophysical.iclamp_stimulus import IclampStimulus
from load_cell_parameters import load_cell_parameters

import h5py
import numpy as np
from allen_wrench.core.orca_data_set import OrcaDataSet

def run(description):
    manifest = description.manifest

    from neuron import h
    h.load_file("stdgui.hoc")
    h.load_file("import3d.hoc")
    
    morphology_path = description.manifest.get_path('MORPHOLOGY')
    ml.generate_morphology(h, morphology_path.encode('ascii', 'ignore'))
    load_cell_parameters(h,
                         description.data['passive'][0],
                         description.data['genome'],
                         description.data['conditions'][0])
    ml.setup_conditions(h, description.data['conditions'][0])
    
    # zero out all sweeps in the output file
    orcas_out_path = manifest.get_path("output_orca")
    zero_sweeps(orcas_out_path)
    
    run_params = description.data['runs'][0]
    sweeps = run_params['sweeps']
    
    stimulus_path = description.manifest.get_path('stimulus_path')
    output = OrcaDataSet(orcas_out_path)
    
    for sweep in sweeps:
        iclamp = IclampStimulus(h)
        iclamp.setup_instance(stimulus_path, sweep=sweep)
    
        vec = ml.record_values(h)
    
        h.finitialize()
        h.run()
        
        # And to an Orca File
        
        mV = 1.0e-3
        junction_potential = description.data['fitting'][0]['junction_potential']
        output_data = (np.array(vec['v']) - junction_potential) * mV
        output.set_sweep(sweep, None, output_data)

def zero_sweeps(file_path):
    with h5py.File(file_path, 'a') as f:
        for sweep in f['epochs'].keys():
            if sweep.startswith('Sweep'):
                f['epochs'][sweep]['response']['timeseries']['data'][...] = 0


if '__main__' == __name__:
    import sys
    manifest_json_path = sys.argv[-1]
    
    fix_sections = ['passive', 'axon_morph,', 'conditions', 'fitting']
    
    description = Config().load(manifest_json_path)
    description.fix_unary_sections(fix_sections)
    
    run(description)


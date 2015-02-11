import allen_wrench.model.single_cell_biophysical.model_load as ml
from allen_wrench.model.single_cell_biophysical.iclamp_stimulus import IclampStimulus

from neuron import h
import numpy as np
from allen_wrench.core.orca_data_set import OrcaDataSet

def run(stimulus_path, morphology_path,
        fit_parameter_path, sweeps,
        orcas_out_path="output.orca",
        json_out_path="lims_response.json"):
    ml.load_model_from_lims(morphology_path,
                            fit_parameter_path)   # TODO: different load style, separate params from swc, sweeps, etc.
    
    output = OrcaDataSet(orcas_out_path)
    
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
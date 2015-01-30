#!/usr/bin/env python

import allen_wrench.model.single_cell_biophysical.model_load as ml
from allen_wrench.model.single_cell_biophysical.iclamp_stimulus import IclampStimulus
from allen_wrench.model.single_cell_biophysical.orca_lob_parser import OrcaLobParser
from neuron import h
import numpy as np

def run(stimulus_path, morphology_path,
        fit_parameter_path, sweeps,
        orcas_out_path="output.orca",
        json_out_path="lims_response.json"):
    ml.load_model_from_lims(morphology_path,
                            fit_parameter_path)   # TODO: different load style, separate params from swc, sweeps, etc.
    
    for sweep in sweeps:
        iclamp = IclampStimulus(h)
        iclamp.setup_instance(stimulus_path, sweep=sweep)
    
        vec = ml.track_typical_values()
    
        h.finitialize()
        h.run()
            
        # And to an Orca File
        output_parser = OrcaLobParser()
        output_data = np.array(vec['v']) * 1.0e-3
        output_parser.write(orcas_out_path, output_data, sweep=sweep)    
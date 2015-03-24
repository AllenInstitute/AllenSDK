from allen_wrench.model.biophys_sim.config import Config
from allen_wrench.model.single_cell_biophysical.utils import Utils
from allen_wrench.core.orca_data_set import OrcaDataSet
import numpy


def run(description):
    # configure NEURON
    utils = Utils(description)
    h = utils.h
    
    # configure model
    manifest = description.manifest
    morphology_path = description.manifest.get_path('MORPHOLOGY')
    utils.generate_morphology(morphology_path.encode('ascii', 'ignore'))
    utils.load_cell_parameters()
    
    # configure stimulus and recording
    stimulus_path = description.manifest.get_path('stimulus_path')
    orcas_out_path = manifest.get_path("output_orca")
    output = OrcaDataSet(orcas_out_path)
    run_params = description.data['runs'][0]
    sweeps = run_params['sweeps']
    junction_potential = description.data['fitting'][0]['junction_potential']
    mV = 1.0e-3
    
    # run sweeps
    for sweep in sweeps:
        utils.setup_iclamp(stimulus_path, sweep=sweep)
        vec = utils.record_values()
        
        h.finitialize()
        h.run()
        
        # write to an Orca File
        output_data = (numpy.array(vec['v']) - junction_potential) * mV
        output.set_sweep(sweep, None, output_data)


def load_description(manifest_json_path):
    description = Config().load(manifest_json_path)
    
    # fix nonstandard description sections
    fix_sections = ['passive', 'axon_morph,', 'conditions', 'fitting']
    description.fix_unary_sections(fix_sections)
    
    return description


if '__main__' == __name__:
    import sys
    
    description = load_description(sys.argv[-1])
    
    run(description)


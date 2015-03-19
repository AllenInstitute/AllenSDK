from allen_wrench.model.biophys_sim.config import Config
import allen_wrench.model.single_cell_biophysical.model_load as ml
from allen_wrench.model.single_cell_biophysical.iclamp_stimulus import IclampStimulus
from allen_wrench.core.orca_data_set import OrcaDataSet
from load_cell_parameters import load_cell_parameters
import h5py, numpy


def run(description):
    manifest = description.manifest
    orcas_out_path = manifest.get_path("output_orca")
    
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
        
        # write to an Orca File
        mV = 1.0e-3
        junction_potential = description.data['fitting'][0]['junction_potential']
        output_data = (numpy.array(vec['v']) - junction_potential) * mV
        output.set_sweep(sweep, None, output_data)


if '__main__' == __name__:
    import sys
    manifest_json_path = sys.argv[-1]
    
    fix_sections = ['passive', 'axon_morph,', 'conditions', 'fitting']
    
    description = Config().load(manifest_json_path)
    description.fix_unary_sections(fix_sections)
    
    run(description)


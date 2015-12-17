# Copyright 2015 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

from allensdk.model.biophys_sim.config import Config
from allensdk.model.biophysical_perisomatic.utils import Utils
from allensdk.core.nwb_data_set import NwbDataSet
import allensdk.ephys.extract_cell_features as extract_cell_features
from shutil import copy
import numpy

def run(description, sweeps=None):
    '''Main function for running a perisomatic biophysical experiment.
    
    Parameters
    ----------
    description : Config
        All information needed to run the experiment.
    '''
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
    run_params = description.data['runs'][0]
    if sweeps == None:
        sweeps = run_params['sweeps']
    junction_potential = description.data['fitting'][0]['junction_potential']
    mV = 1.0e-3
    
    prepare_nwb_output(manifest.get_path('stimulus_path'),
                       manifest.get_path('output'))
    
    # run sweeps
    for sweep in sweeps:
        utils.setup_iclamp(stimulus_path, sweep=sweep)
        
        vec = utils.record_values()
        
        h.finitialize()
        h.run()
        
        # write to an NWB File
        output_data = (numpy.array(vec['v']) - junction_potential) * mV
        
        output_path = manifest.get_path("output")
        save_nwb(output_path, output_data, sweep)


def prepare_nwb_output(nwb_stimulus_path,
                       nwb_result_path):
    '''Copy the stimulus file, zero out the recorded voltages and spike times.
    
    Parameters
    ----------
    nwb_stimulus_path : string
        NWB file name
    nwb_result_path : string
        NWB file name
    '''
    copy(nwb_stimulus_path, nwb_result_path)
    data_set = NwbDataSet(nwb_result_path)
    data_set.fill_sweep_responses(0.0)
    for sweep in data_set.get_sweep_numbers():
        data_set.set_spike_times(sweep, [])


def save_nwb(output_path, v, sweep):
    '''Save a single voltage output result into an existing sweep in a NWB file.
    This is intended to overwrite a recorded trace with a simulated voltage.
    
    Parameters
    ----------
    output_path : string
        file name of a pre-existing NWB file.
    v : numpy array
        voltage
    sweep : integer
        which entry to overwrite in the file.
    '''
    output = NwbDataSet(output_path)
    output.set_sweep(sweep, None, v)
    
    sweep_features = extract_cell_features.extract_sweep_features(output_path,
                                                                  [sweep])
    spikes = sweep_features[sweep]['mean']['spikes']
    spike_times = [ s['t'] for s in spikes ]
    output.set_spike_times(sweep, spike_times)


def load_description(manifest_json_path):
    '''Read configuration file.
    
    Parameters
    ----------
    manifest_json_path : string
        File containing the experiment configuration.
    
    Returns
    -------
    Config
        Object with all information needed to run the experiment.
    '''
    description = Config().load(manifest_json_path)
    
    # fix nonstandard description sections
    fix_sections = ['passive', 'axon_morph,', 'conditions', 'fitting']
    description.fix_unary_sections(fix_sections)
    
    return description


if '__main__' == __name__:
    import sys
    
    description = load_description(sys.argv[-1])
    
    run(description)


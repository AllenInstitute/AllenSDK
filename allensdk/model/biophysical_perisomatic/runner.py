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
from shutil import copy
import numpy


def run(description):
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
    sweeps = run_params['sweeps']
    junction_potential = description.data['fitting'][0]['junction_potential']
    mV = 1.0e-3
    
    stimulus_format = manifest.get_format('stimulus_path')
    output_format = manifest.get_format('output')
    
    # prepare output file
    if output_format == 'NWB':
        output_path = manifest.get_path('output')
        copy(stimulus_path,
             manifest.get_path('output'))
        utils.zero_sweeps(output_path)
        utils.zero_firing_times(output_path)
    
    # run sweeps
    for sweep in sweeps:
        if stimulus_format == 'NWB':
            utils.setup_iclamp(stimulus_path, sweep=sweep)
        elif stimulus_format == 'dat':
            utils.setup_iclamp_dat(stimulus_path)
        
        vec = utils.record_values()
        
        h.finitialize()
        h.run()
        
        # write to an NWB File
        output_data = (numpy.array(vec['v']) - junction_potential) * mV
        output_times = numpy.array(vec['t'])
        
        if output_format == 'NWB':
            output_path = manifest.get_path("output")
            save_nwb(output_path, output_data, sweep)
        elif output_format == 'dat':
            output_path = manifest.get_path("output", sweep)
            save_dat(output_path, output_data, output_times)


def save_nwb(output_path, v, sweep):
    output = NwbDataSet(output_path)
    output.set_sweep(sweep, None, v)


def save_dat(output_path, v, t):
    data = numpy.transpose(numpy.vstack((t, v)))
    with open (output_path, "w") as f:
        numpy.savetxt(f, data)


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


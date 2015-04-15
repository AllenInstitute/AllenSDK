# Copyright 2015 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

from allensdk.model.biophys_sim.config import Config
from allensdk.model.single_cell_biophysical.utils import Utils
from allensdk.core.orca_data_set import OrcaDataSet
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


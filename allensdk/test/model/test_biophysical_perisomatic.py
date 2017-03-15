# Copyright 2016 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# Merchantability Or Fitness FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

import pytest
import os
import numpy
from allensdk.model.biophys_sim.config import Config
from allensdk.model.biophysical.utils import Utils
from allensdk.core.dat_utilities import DatUtilities
from allensdk.api.queries.biophysical_api import BiophysicalApi


@pytest.mark.skipif(True,
                    reason="partial testing")
@pytest.mark.xfail
def test_biophysical():
    neuronal_model_id = 472451419    # get this from the web site

    model_directory = '.'

    bp = BiophysicalApi('http://api.brain-map.org')
    bp.cache_stimulus = False  # don't want to download the large stimulus NWB file
    bp.cache_data(neuronal_model_id, working_directory=model_directory)
    os.system('nrnivmodl modfiles')

    description = Config().load('manifest.json')
    utils = Utils(description)
    h = utils.h

    manifest = description.manifest
    morphology_path = manifest.get_path('MORPHOLOGY')
    utils.generate_morphology(morphology_path.encode('ascii', 'ignore'))
    utils.load_cell_parameters()

    stim = h.IClamp(h.soma[0](0.5))
    stim.amp = 0.18
    stim.delay = 1000.0
    stim.dur = 1000.0

    h.tstop = 3000.0

    vec = utils.record_values()

    h.finitialize()
    h.run()

    output_path = 'output_voltage.dat'

    junction_potential = description.data['fitting'][0]['junction_potential']
    mV = 1.0e-3
    ms = 1.0e-3

    output_data = (numpy.array(vec['v']) - junction_potential) * mV
    output_times = numpy.array(vec['t']) * ms

    DatUtilities.save_voltage(output_path, output_data, output_times)
    
    assert numpy.count_nonzero(output_data) > 0

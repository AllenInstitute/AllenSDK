# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2017. Allen Institute. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Redistributions for commercial purposes are not permitted without the
# Allen Institute's written permission.
# For purposes of this license, commercial purposes is the incorporation of the
# Allen Institute's software into anything for which you will charge fees or
# other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
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

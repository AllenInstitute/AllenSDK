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
from allensdk.api.queries.glif_api import GlifApi
import allensdk.core.json_utilities as json_utilities
from allensdk.model.glif.glif_neuron import GlifNeuron
from allensdk.core.nwb_data_set import NwbDataSet
import os


@pytest.fixture
def neuron_config_file(fn_temp_dir):
    return os.path.join(fn_temp_dir, "neuron_config.json")


@pytest.fixture
def ephys_sweeps_file(fn_temp_dir):
    return os.path.join(fn_temp_dir, "ephys_sweeps.json")


@pytest.fixture
def glif_api():
    endpoint = None

    if 'TEST_API_ENDPOINT' in os.environ:
        endpoint = os.environ['TEST_API_ENDPOINT']
        return GlifApi(endpoint)
    else:
        return GlifApi()


@pytest.fixture
def neuronal_model_id():
    neuronal_model_id = 566302806

    return neuronal_model_id


@pytest.fixture
def configured_glif_api(glif_api, neuronal_model_id, neuron_config_file,
                        ephys_sweeps_file):
    glif_api.get_neuronal_model(neuronal_model_id)

    neuron_config = glif_api.get_neuron_config()
    json_utilities.write(neuron_config_file, neuron_config)

    ephys_sweeps = glif_api.get_ephys_sweeps()
    json_utilities.write(ephys_sweeps_file, ephys_sweeps)

    return glif_api


@pytest.fixture
def output(neuron_config_file, ephys_sweeps_file):
    neuron_config = json_utilities.read(neuron_config_file)
    ephys_sweeps = json_utilities.read(ephys_sweeps_file)
    ephys_file_name = 'stimulus.nwb'

    # pull out the stimulus for the first sweep
    ephys_sweep = ephys_sweeps[0]
    ds = NwbDataSet(ephys_file_name)
    data = ds.get_sweep(ephys_sweep['sweep_number'])
    stimulus = data['stimulus']

    # initialize the neuron
    # important! update the neuron's dt for your stimulus
    neuron = GlifNeuron.from_dict(neuron_config)
    neuron.dt = 1.0 / data['sampling_rate']

    # simulate the neuron
    truncate = 56041
    output = neuron.run(stimulus[0:truncate])

    return output


@pytest.fixture
def stimulus(neuron_config_file, ephys_sweeps_file):
    ephys_sweeps = json_utilities.read(ephys_sweeps_file)
    ephys_file_name = 'stimulus.nwb'

    # pull out the stimulus for the first sweep
    ephys_sweep = ephys_sweeps[0]
    ds = NwbDataSet(ephys_file_name)
    data = ds.get_sweep(ephys_sweep['sweep_number'])
    stimulus = data['stimulus']

    return stimulus


def test_run_glifneuron(configured_glif_api, neuron_config_file):
    # initialize the neuron
    neuron_config = json_utilities.read(neuron_config_file)
    neuron = GlifNeuron.from_dict(neuron_config)

    # make a short square pulse. stimulus units should be in Amps.
    stimulus = [0.0] * 100 + [10e-9] * 100 + [0.0] * 100

    # important! set the neuron's dt value for your stimulus in seconds
    neuron.dt = 5e-6

    # simulate the neuron
    output = neuron.run(stimulus)

    expected_fields = {"AScurrents", "grid_spike_times",
                       "interpolated_spike_threshold",
                       "interpolated_spike_times",
                       "interpolated_spike_voltage",
                       "spike_time_steps", "threshold", "voltage"}

    assert expected_fields.difference(output.keys()) == set()

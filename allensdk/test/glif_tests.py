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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt # noqa: #402
from allensdk.api.queries.glif_api import GlifApi  # noqa: #402
import allensdk.core.json_utilities as json_utilities  # noqa: #402
from allensdk.model.glif.glif_neuron import GlifNeuron  # noqa: #402
from allensdk.model.glif.simulate_neuron import simulate_neuron  # noqa: #402
import os         # noqa: #402
import shutil     # noqa: #402
import logging    # noqa: #402


# NEURONAL_MODEL_ID = 491547163 # level 1 LIF
NEURONAL_MODEL_ID = 491547171  # level 5 GLIF

OUTPUT_DIR = 'tmp'


def test_download():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR)

    glif_api = GlifApi()
    glif_api.get_neuronal_model(NEURONAL_MODEL_ID)
    glif_api.cache_stimulus_file(os.path.join(
        OUTPUT_DIR, '%d.nwb' % NEURONAL_MODEL_ID))

    neuron_config = glif_api.get_neuron_config()
    json_utilities.write(os.path.join(
        OUTPUT_DIR, '%d_neuron_config.json' % NEURONAL_MODEL_ID), neuron_config)

    ephys_sweeps = glif_api.get_ephys_sweeps()
    json_utilities.write(os.path.join(
        OUTPUT_DIR, 'ephys_sweeps.json'), ephys_sweeps)


def test_run():
    # initialize the neuron
    neuron_config = json_utilities.read(os.path.join(
        OUTPUT_DIR, '%d_neuron_config.json' % NEURONAL_MODEL_ID))
    neuron = GlifNeuron.from_dict(neuron_config)

    # make a short square pulse. stimulus units should be in Amps.
    stimulus = [0.0] * 100 + [10e-9] * 100 + [0.0] * 100

    # important! set the neuron's dt value for your stimulus in seconds
    neuron.dt = 5e-6

    # simulate the neuron
    output = neuron.run(stimulus)

    voltage = output['voltage']
    threshold = output['threshold']

    plt.plot(voltage)
    plt.plot(threshold)
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot.png'))


def test_simulate():
    logging.getLogger().setLevel(logging.DEBUG)
    neuron_config = json_utilities.read(os.path.join(
        OUTPUT_DIR, '%d_neuron_config.json' % NEURONAL_MODEL_ID))
    ephys_sweeps = json_utilities.read(
        os.path.join(OUTPUT_DIR, 'ephys_sweeps.json'))
    ephys_file_name = os.path.join(OUTPUT_DIR, '%d.nwb' % NEURONAL_MODEL_ID)

    neuron = GlifNeuron.from_dict(neuron_config)

    sweep_numbers = [s['sweep_number'] for s in ephys_sweeps]
    simulate_neuron(neuron, sweep_numbers,
                    ephys_file_name, ephys_file_name, 0.05)

if __name__ == "__main__":
    # test_download()
    # test_run()
    test_simulate()

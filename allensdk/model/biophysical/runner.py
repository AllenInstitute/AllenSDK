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
from ..biophys_sim.config import Config
from .utils import create_utils
from allensdk.core.nwb_data_set import NwbDataSet
import allensdk.ephys.extract_cell_features as extract_cell_features
from shutil import copy
import numpy
import logging
import time
import os

_runner_log = logging.getLogger('allensdk.model.biophysical.runner')


def run(description, sweeps=None):
    '''Main function for running a biophysical experiment.

    Parameters
    ----------
    description : Config
        All information needed to run the experiment.
    '''
    model_type = description.data['biophys'][0]['model_type']

    # configure NEURON
    utils = create_utils(description, model_type)
    h = utils.h

    # configure model
    manifest = description.manifest
    morphology_path = description.manifest.get_path('MORPHOLOGY')
    utils.generate_morphology(morphology_path.encode('ascii', 'ignore'))
    utils.load_cell_parameters()

    # configure stimulus and recording
    stimulus_path = description.manifest.get_path('stimulus_path')
    run_params = description.data['runs'][0]
    if sweeps is None:
        sweeps = run_params['sweeps']
    sweeps_by_type = run_params['sweeps_by_type']
    junction_potential = description.data['fitting'][0]['junction_potential']
    mV = 1.0e-3

    prepare_nwb_output(manifest.get_path('stimulus_path'),
                       manifest.get_path('output_path'))

    # run sweeps
    for sweep in sweeps:
        _runner_log.info("Running sweep: %d" % (sweep))
        utils.setup_iclamp(stimulus_path, sweep=sweep)
        _runner_log.info("Done loading sweep: %d" % (sweep))
        vec = utils.record_values()
        tstart = time.time()
        h.finitialize()
        h.run()
        tstop = time.time()
        _runner_log.info("Time: %f" % (tstop - tstart))

        # write to an NWB File
        output_data = (numpy.array(vec['v']) - junction_potential) * mV

        output_path = manifest.get_path("output_path")

        save_nwb(output_path, output_data, sweep, sweeps_by_type)


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

    output_dir = os.path.dirname(nwb_result_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    copy(nwb_stimulus_path, nwb_result_path)
    data_set = NwbDataSet(nwb_result_path)
    data_set.fill_sweep_responses(0.0)
    for sweep in data_set.get_sweep_numbers():
        data_set.set_spike_times(sweep, [])


def save_nwb(output_path, v, sweep, sweeps_by_type):
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

    sweep_by_type = {t: [sweep]
                     for t, ss in sweeps_by_type.items() if sweep in ss}
    sweep_features = extract_cell_features.extract_sweep_features(output,
                                                                  sweep_by_type)
    try:
        spikes = sweep_features[sweep]['spikes']
        spike_times = [s['threshold_t'] for s in spikes]
        output.set_spike_times(sweep, spike_times)
    except Exception as e:
        logging.info("sweep %d has no sweep features. %s" % (sweep, e.args))


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

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

from ..biophys_sim.config import Config
from .utils import create_utils
from allensdk.core.nwb_data_set import NwbDataSet
import allensdk.ephys.extract_cell_features as extract_cell_features
from shutil import copy
import numpy
import logging
import time
import os
import multiprocessing as mp
from functools import partial

_runner_log = logging.getLogger('allensdk.model.biophysical.runner')

_lock = None

def _init_lock(lock):
    global _lock
    _lock = lock

def run(description, sweeps=None, procs=4):
    prepare_nwb_output(description.manifest.get_path('stimulus_path'),
                       description.manifest.get_path('output_path'))

    if procs == 1:
        run_sync(description, sweeps)
        return

    if sweeps is None:
        stimulus_path = description.manifest.get_path('stimulus_path')
        run_params = description.data['runs'][0]
        sweeps = run_params['sweeps']

    lock = mp.Lock()
    pool = mp.Pool(procs, initializer=_init_lock, initargs=(lock,))
    pool.map(partial(run_sync, description), [[sweep] for sweep in sweeps])
    pool.close()
    pool.join()


def run_sync(description, sweeps=None):
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

    output_path = manifest.get_path("output_path")

    # run sweeps
    for sweep in sweeps:
        _runner_log.info("Loading sweep: %d" % (sweep))
        utils.setup_iclamp(stimulus_path, sweep=sweep)

        _runner_log.info("Simulating sweep: %d" % (sweep))
        vec = utils.record_values()
        tstart = time.time()
        h.finitialize()
        h.run()
        tstop = time.time()
        _runner_log.info("Time: %f" % (tstop - tstart))

        # write to an NWB File
        _runner_log.info("Writing sweep: %d" % (sweep))
        recorded_data = utils.get_recorded_data(vec)

        if _lock is not None:
            _lock.acquire()
        save_nwb(output_path, recorded_data["v"], sweep, sweeps_by_type)
        if _lock is not None:
            _lock.release()


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

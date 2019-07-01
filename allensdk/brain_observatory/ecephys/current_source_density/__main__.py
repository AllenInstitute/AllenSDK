import numpy as np
import requests
import logging
import sys

import os
import pandas as pd

from ._schemas import InputParameters, OutputParameters
from ._current_source_density import accumulate_lfp_data, compute_csd, extract_trial_windows, identify_lfp_channels, get_missing_channels
from allensdk.brain_observatory.ecephys.file_io.continuous_file import ContinuousFile
from allensdk.brain_observatory.argschema_utilities import (
    write_or_print_outputs, optional_lims_inputs
)


def get_inputs_from_lims(args):

    session_id = args.session_id
    output_root = args.output_root
    host = args.host

    request_str = ''.join('''
    {}/input_jsons?
    strategy_class=EcephysCurrentSourceDensityStrategy&
    object_id={}&
    object_class=EcephysSession&
    job_queue_name=ECEPHYS_CURRENT_SOURCE_DENSITY_QUEUE
    '''.format(host, session_id).split())

    response = requests.get(request_str)
    data = response.json()

    if data['num_trials'] == 'null':
        data['num_trials'] = None
    else:
        data['num_trials'] = int(data['num_trials'])

    data['pre_stimulus_time'] = float(data['pre_stimulus_time'])
    data['post_stimulus_time'] = float(data['post_stimulus_time'])
    data['surface_channel_adjustment'] = int(data['surface_channel_adjustment'])

    for probe in data['probes']:
        probe['surface_channel_adjustment'] = int(probe['surface_channel_adjustment'])
        probe['csd_output_path'] = os.path.join(output_root, os.path.split(probe['csd_output_path'])[-1])
        probe['relative_window_output_path'] = os.path.join(output_root, os.path.split(probe['relative_window_output_path'])[-1])

    return data


def run_csd(args):
    """
    """

    stimulus_table = pd.read_csv(args['stimulus']['stimulus_table_path'])

    probewise_outputs = []
    for probe_idx, probe in enumerate(args['probes']):
        logging.info('processing probe: {} (index: {})'.format(probe['name'], probe_idx))

        time_step = 1.0 / probe['sampling_rate']
        logging.info('calculated time step: {}'.format(time_step))

        trial_windows, relative_window = extract_trial_windows(
            stimulus_table, args['stimulus']['key'], time_step, args['pre_stimulus_time'], args['post_stimulus_time'], 
            args['num_trials'], args['stimulus']['index']
        )

        lfp_data_file = ContinuousFile(probe['lfp_data_path'],probe['lfp_timestamps_path'], probe['total_channels'])
        lfp_raw, timestamps = lfp_data_file.load(memmap=args['memmap'], memmap_thresh=args['memmap_thresh'])
        
        surface_channel = min(probe['surface_channel'] + probe['surface_channel_adjustment'], probe['total_channels'] - 1)
        logging.info('calculated surface channel: {}'.format(surface_channel))

        lfp_channels = identify_lfp_channels(surface_channel, probe['reference_channels'])
        missing_channels = get_missing_channels(lfp_channels)

        accumulated_lfp_data = accumulate_lfp_data(timestamps, lfp_raw, lfp_channels, trial_windows)
        current_source_density, csd_channels = compute_csd(accumulated_lfp_data, lfp_channels, missing_channels, spacing=probe['spacing'])

        np.save(probe['csd_output_path'], current_source_density, allow_pickle=False)
        np.save(probe['relative_window_output_path'], relative_window, allow_pickle=False)
        probewise_outputs.append({
            'name': probe['name'], 
            'csd_path': probe['csd_output_path'],
            'relative_window_path': probe['relative_window_output_path'],
            'csd_channels': csd_channels.tolist()
        })

    return {
        'probe_outputs': probewise_outputs, 
        "stimulus_name": args["stimulus"]["key"], 
        "stimulus_index": args["stimulus"]["index"]
    }


def main():

    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s')
    parser = optional_lims_inputs(sys.argv, InputParameters, OutputParameters, get_inputs_from_lims)
    output = run_csd(parser.args)
    write_or_print_outputs(output, parser)


if __name__ == "__main__":
    main()

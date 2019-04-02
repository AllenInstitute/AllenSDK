import functools
import warnings
import re

import pandas as pd
import numpy as np

from . import ephys_pre_spikes
from allensdk.brain_observatory.ecephys.file_io.ecephys_sync_dataset import EcephysSyncDataset 
from allensdk.brain_observatory.argschema_utilities import ArgSchemaParserPlus
from allensdk.brain_observatory.ecephys.file_io.stim_file import CamStimOnePickleStimFile
from ._schemas import InputParameters, OutputParameters


def validate_epoch_durations(table, start_key='Start', end_key='End'):
    durations = table[end_key].values - table[start_key].values
    min_duration = np.amin(durations)

    assert min_duration >= 0
    if min_duration == 0:
        warnings.warn(UserWarning, 'there is a 0-duration epoch in this stimulus table')


def validate_epoch_order(table, time_keys=('Start', 'End')):
    for time_key in time_keys:
        change = np.diff(table[time_key].values)
        assert np.amin(change) > 0


def validate_max_spontaneous_epoch_duration(
    table, max_duration, get_spontanous_epochs=None, index_key='stimulus_index', start_key='Start', end_key='End'
):
    if get_spontanous_epochs is None:
        get_spontanous_epochs = lambda table: table[np.isnan(table[index_key])]

    spontaneous_epochs = get_spontanous_epochs(table)
    durations = spontaneous_epochs[end_key].values - spontaneous_epochs[start_key].values
    if  np.amax(durations) > max_duration:
        warnings.warn(UserWarning, f'there is a spontaneous activity duration longer than {max_duration}')


def build_stimulus_table(args):

    print('Building stimulus table...')

    stim_file = CamStimOnePickleStimFile.factory(args['stimulus_pkl_path'])

    sync_dataset = EcephysSyncDataset.factory(args['sync_h5_path'])
    frame_times = sync_dataset.extract_frame_times(strategy=args['frame_time_strategy'])

    seconds_to_frames = lambda seconds: (np.array(seconds) + stim_file.pre_blank_sec) * stim_file.frames_per_second
    minimum_spontaneous_activity_duration = args['minimum_spontaneous_activity_duration'] / stim_file.frames_per_second

    stimulus_tabler = functools.partial(ephys_pre_spikes.build_stimuluswise_table, seconds_to_frames=seconds_to_frames)
    spon_tabler = functools.partial(ephys_pre_spikes.make_spontaneous_activity_tables, duration_threshold=minimum_spontaneous_activity_duration)
    
    stim_table_full = ephys_pre_spikes.create_stim_table(stim_file.stimuli, stimulus_tabler, spon_tabler)
    stim_table_full = ephys_pre_spikes.apply_frame_times(stim_table_full, frame_times, stim_file.frames_per_second, True)

    validate_epoch_durations(stim_table_full)
    validate_max_spontaneous_epoch_duration(stim_table_full, args['maximum_expected_spontanous_activity_duration'])

    # special case the gabor_20_deg_250ms stimulus
    stim_table_full['diameter'] = np.nan
    stim_table_full['diameter'][stim_table_full['stimulus_name'] == 'gabor_20_deg_250ms'] = 20
    stim_table_full['stimulus_name'][stim_table_full['stimulus_name'] == 'gabor_20_deg_250ms'] = 'gabor'

    if args.get('stimulus_name_map', None) is not None:
        stim_table_full['stimulus_name'].fillna('', inplace=True) # exposes no-stim condition as empty string
        stim_table_full['stimulus_name'].replace(to_replace=args['stimulus_name_map'], inplace=True)
        stim_table_full['stimulus_name'].replace(to_replace={'': np.nan}, inplace=True)
    if args.get('column_name_map', None) is not None:
        stim_table_full.rename(columns=args['column_name_map'], inplace=True)

    stim_table_full.to_csv(args['output_stimulus_table_path'], index=False)
    return {'output_path': args['output_stimulus_table_path']}


def main():

    mod = ArgSchemaParserPlus(schema_type=InputParameters, output_schema_type=OutputParameters)
    output = build_stimulus_table(mod.args)

    output.update({"input_parameters": mod.args})
    if "output_json" in mod.args:
        mod.output(output, indent=2)
    else:
        print(mod.get_output_json(output))

    
if __name__ == "__main__":
    main()


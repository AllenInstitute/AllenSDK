import functools
import warnings
import re

import pandas as pd
import numpy as np

from allensdk.brain_observatory.ecephys.file_io.ecephys_sync_dataset import EcephysSyncDataset 
from allensdk.brain_observatory.argschema_utilities import ArgSchemaParserPlus
from allensdk.brain_observatory.ecephys.file_io.stim_file import CamStimOnePickleStimFile

from . import ephys_pre_spikes
from . import naming_utilities
from . import output_validation
from ._schemas import InputParameters, OutputParameters


def build_stimulus_table(args):

    stim_file = CamStimOnePickleStimFile.factory(args['stimulus_pkl_path'])

    sync_dataset = EcephysSyncDataset.factory(args['sync_h5_path'])
    frame_times = sync_dataset.extract_frame_times(strategy=args['frame_time_strategy'])

    seconds_to_frames = lambda seconds: (np.array(seconds) + stim_file.pre_blank_sec) * stim_file.frames_per_second
    minimum_spontaneous_activity_duration = args['minimum_spontaneous_activity_duration'] / stim_file.frames_per_second

    stimulus_tabler = functools.partial(ephys_pre_spikes.build_stimuluswise_table, seconds_to_frames=seconds_to_frames)
    spon_tabler = functools.partial(ephys_pre_spikes.make_spontaneous_activity_tables, duration_threshold=minimum_spontaneous_activity_duration)
    
    stim_table_full = ephys_pre_spikes.create_stim_table(stim_file.stimuli, stimulus_tabler, spon_tabler)
    stim_table_full = ephys_pre_spikes.apply_frame_times(stim_table_full, frame_times, stim_file.frames_per_second, True)

    output_validation.validate_epoch_durations(stim_table_full)
    output_validation.validate_max_spontaneous_epoch_duration(stim_table_full, args['maximum_expected_spontanous_activity_duration'])

    stim_table_full = naming_utilities.collapse_columns(stim_table_full)
    stim_table_full = naming_utilities.standardize_movie_numbers(stim_table_full)
    stim_table_full = naming_utilities.add_number_to_shuffled_movie(stim_table_full)
    stim_table_full = naming_utilities.map_stimulus_names(stim_table_full, args['stimulus_name_map'])
    stim_table_full.rename(columns=args['column_name_map'], inplace=True)

    stim_table_full.to_csv(args['output_stimulus_table_path'], index=False)
    np.save(args['output_frame_times_path'], frame_times, allow_pickle=False)
    return {
        'output_path': args['output_stimulus_table_path'],
        'output_frame_times_path': args['output_frame_times_path']
    }


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


"""
This file defines an ArgSchemaParser for generating an HDF5 file
containing all of the running speed data from a session in which
multiple stimulus blocks (behavior, mapping, replay) were presented
to the mouse and need to be registered to the sync file.
"""

import pandas as pd
import argschema
import json

from allensdk.brain_observatory.behavior.data_files.stimulus_file import (
    BehaviorStimulusFile,
    MappingStimulusFile,
    ReplayStimulusFile)

from allensdk.brain_observatory.dual_stimulus_running_speed._schemas import (
    DualStimulusRunningSpeedInputParameters,
    DualStimulusRunningSpeedOutputParameters
)

from allensdk.brain_observatory.behavior.data_objects.\
    running_speed.multi_stim_running_processing import (
       dual_stim_running_df_from_raw_data)


class DualStimulusRunningSpeed(argschema.ArgSchemaParser):
    default_schema = DualStimulusRunningSpeedInputParameters
    default_output_schema = DualStimulusRunningSpeedOutputParameters

    START_FRAME = 0

    def _write_output_json(self):
        """
        Write the output json file
        """

        ouput_data = {}
        ouput_data['output_path'] = self.args['output_path']
        ouput_data['input_parameters'] = self.args

        with open(self.args['output_json'], 'w') as output_file:
            json.dump(ouput_data, output_file, indent=2)

    def process(
        self
    ):
        """
        Process an experiment with a three stimulus sessions
        """

        bstim = BehaviorStimulusFile.from_json(
               dict_repr={'behavior_stimulus_file':
                          self.args['behavior_pkl_path']})

        mstim = MappingStimulusFile.from_json(
               dict_repr={'mapping_stimulus_file':
                          self.args['mapping_pkl_path']})
        
        (velocities,
         raw_data) = dual_stim_running_df_from_raw_data(
                 sync_path=self.args['sync_h5_path'],
                 behavior_stimulus_file=bstim,
                 mapping_stimulus_file=mstim,
                 use_lowpass_filter=self.args['use_lowpass_filter'],
                 zscore_threshold=self.args['zscore_threshold'],
                 behavior_start_frame=DualStimulusRunningSpeed.START_FRAME)

        store = pd.HDFStore(self.args['output_path'])
        store.put("running_speed", velocities)
        store.put("raw_data", raw_data)
        store.close()

        self._write_output_json()

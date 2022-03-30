import pandas as pd
import argschema
import json

from allensdk.brain_observatory.multi_stimulus_running_speed._schemas import (
    MultiStimulusRunningSpeedInputParameters,
    MultiStimulusRunningSpeedOutputParameters
)

from allensdk.brain_observatory.ecephys.data_objects.\
    running_speed.multi_stim_running_processing import (
        multi_stim_running_df_from_raw_data)


class MultiStimulusRunningSpeed(argschema.ArgSchemaParser):
    default_schema = MultiStimulusRunningSpeedInputParameters
    default_output_schema = MultiStimulusRunningSpeedOutputParameters

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

        (velocities,
         raw_data) = multi_stim_running_df_from_raw_data(
                 sync_path=self.args['sync_h5_path'],
                 behavior_pkl_path=self.args['behavior_pkl_path'],
                 mapping_pkl_path=self.args['mapping_pkl_path'],
                 replay_pkl_path=self.args['replay_pkl_path'],
                 use_lowpass_filter=self.args['use_lowpass_filter'],
                 zscore_threshold=self.args['zscore_threshold'],
                 behavior_start_frame=MultiStimulusRunningSpeed.START_FRAME)

        store = pd.HDFStore(self.args['output_path'])
        store.put("running_speed", velocities)
        store.put("raw_data", raw_data)
        store.close()

        self._write_output_json()

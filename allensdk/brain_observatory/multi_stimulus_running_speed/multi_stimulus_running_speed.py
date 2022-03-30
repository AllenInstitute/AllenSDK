import numpy as np
import pandas as pd
import argschema
import json

from allensdk.brain_observatory.multi_stimulus_running_speed._schemas import (
    MultiStimulusRunningSpeedInputParameters,
    MultiStimulusRunningSpeedOutputParameters
)

from allensdk.brain_observatory.ecephys.data_objects.\
    running_speed.multi_stim_running_processing import (
        _extract_dx_info,
        _get_frame_counts,
        _get_frame_times)


class MultiStimulusRunningSpeed(argschema.ArgSchemaParser):
    default_schema = MultiStimulusRunningSpeedInputParameters
    default_output_schema = MultiStimulusRunningSpeedOutputParameters

    START_FRAME = 0

    def _get_stimulus_starts_and_ends(self) -> list:
        """
        Get the start and stop frame indexes for each stimulus
        """

        (
            behavior_frame_count,
            mapping_frame_count,
            replay_frames_count
        ) = _get_frame_counts(
                behavior_pkl_path=self.args['behavior_pkl_path'],
                mapping_pkl_path=self.args['mapping_pkl_path'],
                replay_pkl_path=self.args['replay_pkl_path'])

        behavior_start = MultiStimulusRunningSpeed.START_FRAME
        mapping_start = behavior_frame_count
        replay_start = mapping_start + mapping_frame_count
        replay_end = replay_start + replay_frames_count

        return (
            behavior_start,
            mapping_start,
            replay_start,
            replay_end
        )

    def _merge_dx_data(
        self,
        mapping_velocities: pd.core.frame.DataFrame,
        behavior_velocities: pd.core.frame.DataFrame,
        replay_velocities: pd.core.frame.DataFrame,
        frame_times: np.ndarray
    ) -> list:
        """
        Concatenate all of the running speed data

        Parameters
        ----------
        mapping_velocities: pandas.core.frame.DataFrame
            Velocity data from mapping stimulus
        behavior_velocities: pandas.core.frame.DataFrame
           Velocity data from behavior stimulus
        replay_velocities: pandas.core.frame.DataFrame
            Velocity data from replay stimulus
        frame_times: numpy.ndarray
            list of the vsync times

        Returns
        -------
        list[pd.DataFrame, pd.DataFrame]
            concatenated velocity data, raw data
        """

        speed = np.concatenate(
            (
                behavior_velocities['speed'],
                mapping_velocities['speed'],
                replay_velocities['speed']),
            axis=None
        )

        dx = np.concatenate(
            (
                behavior_velocities['dx'],
                mapping_velocities['dx'],
                replay_velocities['dx']),
            axis=None
        )

        vsig = np.concatenate(
            (
                behavior_velocities['v_sig'],
                mapping_velocities['v_sig'],
                replay_velocities['v_sig']),
            axis=None
        )

        vin = np.concatenate(
            (
                behavior_velocities['v_in'],
                mapping_velocities['v_in'],
                replay_velocities['v_in']),
            axis=None
        )

        frame_indexes = list(
            range(MultiStimulusRunningSpeed.START_FRAME, len(frame_times))
        )

        velocities = pd.DataFrame(
            {
                "velocity": speed,
                "net_rotation": dx,
                "frame_indexes": frame_indexes,
                "frame_time": frame_times
            }
        )

        # Warning - the 'isclose' line below needs to be refactored
        # is it exists in multiple places

        # due to an acquisition bug (the buffer of raw orientations
        # may be updated more slowly than it is read, leading to
        # a 0 value for the change in orientation over an interval)
        # there may be exact zeros in the velocity.
        velocities = velocities[~(np.isclose(velocities["net_rotation"], 0.0))]

        raw_data = pd.DataFrame(
            {"vsig": vsig, "vin": vin, "frame_time": frame_times, "dx": dx}
        )

        return velocities, raw_data

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

        (
            behavior_start,
            mapping_start,
            replay_start,
            replay_end
        ) = self._get_stimulus_starts_and_ends()

        frame_times = _get_frame_times(
                          sync_path=self.args['sync_h5_path'])

        behavior_velocities = _extract_dx_info(
            frame_times,
            behavior_start,
            mapping_start,
            self.args['behavior_pkl_path'],
            zscore_threshold=self.args['zscore_threshold'],
            use_lowpass_filter=self.args['use_lowpass_filter']
        )

        mapping_velocities = _extract_dx_info(
            frame_times,
            mapping_start,
            replay_start,
            self.args['mapping_pkl_path'],
            zscore_threshold=self.args['zscore_threshold'],
            use_lowpass_filter=self.args['use_lowpass_filter']
        )

        replay_velocities = _extract_dx_info(
            frame_times,
            replay_start,
            replay_end,
            self.args['replay_pkl_path'],
            zscore_threshold=self.args['zscore_threshold'],
            use_lowpass_filter=self.args['use_lowpass_filter']
        )

        velocities, raw_data = self._merge_dx_data(
            mapping_velocities,
            behavior_velocities,
            replay_velocities,
            frame_times
        )

        store = pd.HDFStore(self.args['output_path'])
        store.put("running_speed", velocities)
        store.put("raw_data", raw_data)
        store.close()

        self._write_output_json()

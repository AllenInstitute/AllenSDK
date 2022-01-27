import numpy as np
import pandas as pd
from allensdk.brain_observatory.sync_dataset import Dataset as SyncDataset
from allensdk.brain_observatory import sync_utilities
import argschema
import json

from allensdk.brain_observatory.filtered_running_speed._schemas import (
    MultiStimulusRunningSpeedInputParameters,
    MultiStimulusRunningSpeedOutputParameters
)

from allensdk.brain_observatory.behavior.data_objects.\
    running_speed.running_processing import (
        get_running_df
    )


class MultiStimulusRunningSpeed(argschema.ArgSchemaParser):
    default_schema = MultiStimulusRunningSpeedInputParameters
    default_output_schema = MultiStimulusRunningSpeedOutputParameters

    START_FRAME = 0

    def _extract_dx_info(
        self,
        frame_times: np.ndarray,
        start_index: int,
        end_index: int,
        pkl_path: str
    ) -> pd.core.frame.DataFrame:
        """
        Extract all of the running speed data

        Parameters
        ----------
        frame_times: numpy.ndarray
            list of the vsync times
        start_index: int
            Index to the first frame of the stimulus
        end_index: int
           Index to the last frame of the stimulus
        pkl_path: string
            Path to the stimulus pickle file

        Returns
        -------
        pd.DataFrame

        Notes
        -------
            velocity pd.DataFrame:
                columns:
                    "velocity": computed running speed
                    "net_rotation": dx in radians
                    "frame_indexes": frame indexes into
                        the full vsync times list

            raw data pd.DataFrame:
                Dataframe with an index of timestamps and the following
                columns:
                    "vsig": voltage signal from the encoder
                    "vin": the theoretical maximum voltage that the encoder
                        will reach prior to "wrapping". This should
                        theoretically be 5V (after crossing
                        5V goes to 0V, or vice versa). In
                        practice the encoder does not always
                        reach this value before wrapping, which can cause
                        transient spikes in speed at the voltage "wraps".
                    "frame_time": list of the vsync times
                    "dx": angular change, computed during data collection
                The raw data are provided so that the user may compute
                their own speed from source, if desired.

        """

        stim_file = pd.read_pickle(pkl_path)
        frame_times = frame_times[start_index:end_index]

        # occasionally an extra set of frame times are acquired
        # after the rest of the signals. We detect and remove these
        frame_times = sync_utilities.trim_discontiguous_times(frame_times)

        velocities = get_running_df(
                        stim_file,
                        frame_times,
                        self.args['use_lowpass_filter'],
                        self.args['zscore_threshold']
        )

        return velocities

    def _get_behavior_frame_count(
        self,
        pkl_file_path: str
    ) -> int:
        """
        Get the number of frames in a behavior pickle file

        Parameters
        ----------
        pkl_file_path: string
            A path to a behavior pickle file
        """
        data = pd.read_pickle(pkl_file_path)

        return len(data["items"]["behavior"]['intervalsms']) + 1

    def _get_frame_count(
        self,
        pkl_file_path: str
    ) -> int:
        """
        Get the number of frames in a mapping or replay pickle file

        Parameters
        ----------
        pkl_file_path: string
            A path to a mapping or replay pickle file
        """

        data = pd.read_pickle(pkl_file_path)

        return len(data['intervalsms']) + 1

    def _get_frame_counts(
        self
    ) -> list:
        """
        Get the number of frames for each stimulus
        """

        behavior_frame_count = self._get_behavior_frame_count(
            self.args['behavior_pkl_path']
        )

        mapping_frame_count = self._get_frame_count(
            self.args['mapping_pkl_path']
        )

        replay_frames_count = self._get_frame_count(
            self.args['replay_pkl_path']
        )

        return behavior_frame_count, mapping_frame_count, replay_frames_count

    def _get_frame_times(
        self
    ) -> np.ndarray:
        """
        Get the vsync frame times
        """
        sync_data = SyncDataset(self.args['sync_h5_path'])

        return sync_data.get_edges(
            "rising", SyncDataset.FRAME_KEYS, units="seconds"
        )

    def _get_stimulus_starts_and_ends(self) -> list:
        """
        Get the start and stop frame indexes for each stimulus
        """

        (
            behavior_frame_count,
            mapping_frame_count,
            replay_frames_count
        ) = self._get_frame_counts()

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

        # Warning - the 'iscose' line below needs to be refactored
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

        frame_times = self._get_frame_times()

        behavior_velocities = self._extract_dx_info(
            frame_times,
            behavior_start,
            mapping_start,
            self.args['behavior_pkl_path']
        )

        mapping_velocities = self._extract_dx_info(
            frame_times,
            mapping_start,
            replay_start,
            self.args['mapping_pkl_path']
        )

        replay_velocities = self._extract_dx_info(
            frame_times,
            replay_start,
            replay_end,
            self.args['replay_pkl_path']
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

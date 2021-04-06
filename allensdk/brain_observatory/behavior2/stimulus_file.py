from itertools import chain
import pandas as pd
import scipy.stats as sps
import numpy as np

from allensdk.internal.core.lims_utilities import safe_system_path


class StimulusFile:
    def __init__(self, filepath: str):
        self._data = self._read_stimulus_file(filepath=filepath)

    @property
    def data(self):
        return self._data

    @staticmethod
    def from_lims(dbconn, behavior_session_id) -> "StimulusFile":
        """Return the path to the StimulusPickle file for a session.
        :rtype: str
        """
        query = f"""
            SELECT
                stim.storage_directory || stim.filename AS stim_file
            FROM
                well_known_files stim
            WHERE
                stim.attachable_id = {behavior_session_id}
                AND stim.attachable_type = 'BehaviorSession'
                AND stim.well_known_file_type_id IN (
                    SELECT id
                    FROM well_known_file_types
                    WHERE name = 'StimulusPickle');
        """
        filepath = dbconn.fetchone(query, strict=True)
        return StimulusFile(filepath=filepath)

    @staticmethod
    def from_json(dict_repr: dict):
        filepath = dict_repr['stimulus_filepath']
        return StimulusFile(filepath=filepath)

    def get_stimulus_timestamps(self):
        vsyncs = self._data["items"]["behavior"]["intervalsms"]
        cum_sum = np.hstack((0, vsyncs)).cumsum() / 1000.0  # cumulative time
        offset = self._calc_frame_time_offset()
        return cum_sum + offset

    def _calc_frame_time_offset(self) -> float:
        """
        Contained in the behavior "pickle" file is a series of time between
        consecutive vsync frames (`intervalsms`). This information required
        to get the timestamp (via frame number) for events that occured
        outside of a trial(e.g. licks). However, we don't have the value
        in the trial log time stream when the first vsync frame actually
        occured -- so we estimate it with a linear regression (frame
        number x time). All trials in the `trial_log` have events for
        `trial_start` and `trial_end`, so these are used to fit the
        regression. A linear regression is used rather than just
        subtracting the time from the first trial, since there can be some
        jitter given the 60Hz refresh rate.

        Parameters
        ----------
        data: dict
            behavior pickle well-known file data

        Returns
        -------
        float
            Time offset to add to the vsync stream to sync it with the
            `trial_log` time stream. The "zero-th" frame time.
        """
        events = [trial["events"] for trial
                  in self._data["items"]["behavior"]["trial_log"]]
        # First event in `events` is `trial_start`, and last is `trial_end`
        # Event log has following schema:
        #    event name, description: 'enter'/'exit'/'', time, frame number
        # We want last two fields for first and last trial event
        trial_by_frame = list(chain(
            [event[i][-2:] for event in events for i in [0, -1]]))
        times = [trials[0] for trials in trial_by_frame]
        frames = [trials[1] for trials in trial_by_frame]
        time_to_first_vsync = sps.linregress(frames, times).intercept
        return time_to_first_vsync

    @staticmethod
    def _read_stimulus_file(filepath: str):
        filepath = safe_system_path(file_name=filepath)
        return pd.read_pickle(filepath)

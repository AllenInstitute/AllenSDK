from typing import Dict, Optional, Any
from datetime import datetime
import pytz
import uuid

import numpy as np
import pandas as pd

from allensdk.core.exceptions import DataFrameIndexError
from allensdk.api.cache import memoize
from allensdk.brain_observatory.behavior.session_apis.abcs import (
    BehaviorBase)
from allensdk.brain_observatory.behavior.rewards_processing import get_rewards
from allensdk.brain_observatory.behavior.running_processing import (
    get_running_df)
from allensdk.brain_observatory.behavior.stimulus_processing import (
    get_stimulus_presentations, get_stimulus_templates, get_stimulus_metadata)
from allensdk.brain_observatory.running_speed import RunningSpeed
from allensdk.brain_observatory.behavior.metadata_processing import (
    get_task_parameters)
from allensdk.brain_observatory.behavior.sync import frame_time_offset
from allensdk.brain_observatory.behavior.trials_processing import (
    get_trials, get_extended_trials)


class BehaviorDataXforms(BehaviorBase):
    """This class provides methods that transform (xform) 'raw' data provided
    by LIMS data APIs to fill a BehaviorSession.
    """

    @memoize
    def _behavior_stimulus_file(self) -> pd.DataFrame:
        """Helper method to cache stimulus file in memory since it takes about
        a second to load (and is used in many methods).
        """
        return pd.read_pickle(self.get_behavior_stimulus_file())

    def get_behavior_session_uuid(self) -> Optional[int]:
        """Get the universally unique identifier (UUID) number for the
        current behavior session.
        """
        data = self._behavior_stimulus_file()
        behavior_pkl_uuid = data.get("session_uuid")

        behavior_session_id = self.get_behavior_session_id()
        foraging_id = self.get_foraging_id()

        # Sanity check to ensure that pkl file data matches up with
        # the behavior session that the pkl file has been associated with.
        assert_err_msg = (
            f"The behavior session UUID ({behavior_pkl_uuid}) in the "
            f"behavior stimulus *.pkl file "
            f"({self.get_behavior_stimulus_file()}) does "
            f"does not match the foraging UUID ({foraging_id}) for "
            f"behavior session: {behavior_session_id}")
        assert behavior_pkl_uuid == foraging_id, assert_err_msg

        return behavior_pkl_uuid

    def get_licks(self) -> pd.DataFrame:
        """Get lick data from pkl file.
        This function assumes that the first sensor in the list of
        lick_sensors is the desired lick sensor. If this changes we need
        to update to get the proper line.

        Since licks can occur outside of a trial context, the lick times
        are extracted from the vsyncs and the frame number in `lick_events`.
        Since we don't have a timestamp for when in "experiment time" the
        vsync stream starts (from self.get_stimulus_timestamps), we compute
        it by fitting a linear regression (frame number x time) for the
        `start_trial` and `end_trial` events in the `trial_log`, to true
        up these time streams.

        :returns: pd.DataFrame -- A dataframe containing lick timestamps
        """
        # Get licks from pickle file instead of sync
        data = self._behavior_stimulus_file()
        stimulus_timestamps = self.get_stimulus_timestamps()
        lick_frames = (data["items"]["behavior"]["lick_sensors"][0]
                       ["lick_events"])
        lick_times = [stimulus_timestamps[frame] for frame in lick_frames]
        return pd.DataFrame({"time": lick_times})

    def get_rewards(self) -> pd.DataFrame:
        """Get reward data from pkl file, based on pkl file timestamps
        (not sync file).

        :returns: pd.DataFrame -- A dataframe containing timestamps of
            delivered rewards.
        """
        data = self._behavior_stimulus_file()
        offset = frame_time_offset(data)
        # No sync timestamps to rebase on, but do need to align to
        # trial events, so add the offset as the "rebase" function
        return get_rewards(data, lambda x: x + offset)

    def get_running_data_df(self, lowpass=True) -> pd.DataFrame:
        """Get running speed data.

        :returns: pd.DataFrame -- dataframe containing various signals used
            to compute running speed.
        """
        stimulus_timestamps = self.get_stimulus_timestamps()
        data = self._behavior_stimulus_file()
        return get_running_df(data, stimulus_timestamps, lowpass=lowpass)

    def get_running_speed(self, lowpass=True) -> RunningSpeed:
        """Get running speed using timestamps from
        self.get_stimulus_timestamps.

        NOTE: Do not correct for monitor delay.

        :returns: RunningSpeed -- a NamedTuple containing the subject's
            timestamps and running speeds (in cm/s)
        """
        running_data_df = self.get_running_data_df(lowpass=lowpass)
        if running_data_df.index.name != "timestamps":
            raise DataFrameIndexError(
                f"Expected index to be named 'timestamps' but got "
                f"'{running_data_df.index.name}'.")
        return RunningSpeed(timestamps=running_data_df.index.values,
                            values=running_data_df.speed.values)

    def get_stimulus_frame_rate(self) -> float:
        stimulus_timestamps = self.get_stimulus_timestamps()
        return np.round(1 / np.mean(np.diff(stimulus_timestamps)), 0)

    def get_stimulus_presentations(self) -> pd.DataFrame:
        """Get stimulus presentation data.

        NOTE: Uses timestamps that do not account for monitor delay.

        :returns: pd.DataFrame --
            Table whose rows are stimulus presentations
            (i.e. a given image, for a given duration, typically 250 ms)
            and whose columns are presentation characteristics.
        """
        stimulus_timestamps = self.get_stimulus_timestamps()
        data = self._behavior_stimulus_file()
        raw_stim_pres_df = get_stimulus_presentations(
            data, stimulus_timestamps)

        # Fill in nulls for image_name
        # This makes two assumptions:
        #   1. Nulls in `image_name` should be "gratings_<orientation>"
        #   2. Gratings are only present (or need to be fixed) when all
        #      values for `image_name` are null.
        if pd.isnull(raw_stim_pres_df["image_name"]).all():
            if ~pd.isnull(raw_stim_pres_df["orientation"]).all():
                raw_stim_pres_df["image_name"] = (
                    raw_stim_pres_df["orientation"]
                    .apply(lambda x: f"gratings_{x}"))
            else:
                raise ValueError("All values for 'orentation' and 'image_name'"
                                 " are null.")

        stimulus_metadata_df = get_stimulus_metadata(data)

        idx_name = raw_stim_pres_df.index.name
        stimulus_index_df = (
            raw_stim_pres_df
            .reset_index()
            .merge(stimulus_metadata_df.reset_index(), on=["image_name"])
            .set_index(idx_name))
        stimulus_index_df = (
            stimulus_index_df[["image_set", "image_index", "start_time",
                               "phase", "spatial_frequency"]]
            .rename(columns={"start_time": "timestamps"})
            .sort_index()
            .set_index("timestamps", drop=True))
        stim_pres_df = raw_stim_pres_df.merge(
            stimulus_index_df, left_on="start_time", right_index=True,
            how="left")
        if len(raw_stim_pres_df) != len(stim_pres_df):
            raise ValueError("Length of `stim_pres_df` should not change after"
                             f" merge; was {len(raw_stim_pres_df)}, now "
                             f" {len(stim_pres_df)}.")
        return stim_pres_df[sorted(stim_pres_df)]

    def get_stimulus_templates(self) -> Dict[str, np.ndarray]:
        """Get stimulus templates (movies, scenes) for behavior session.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing the stimulus images presented during the
            session. Keys are data set names, and values are 3D numpy arrays.
        """
        data = self._behavior_stimulus_file()
        return get_stimulus_templates(data)

    def get_stimulus_timestamps(self) -> np.ndarray:
        """Get stimulus timestamps (vsyncs) from pkl file. Align to the
        (frame, time) points in the trial events.

        NOTE: Located with behavior_session_id. Does not use the sync_file
        which requires ophys_session_id.

        Returns
        -------
        np.ndarray
            Timestamps associated with stimulus presentations on the monitor
            that do no account for monitor delay.
        """
        data = self._behavior_stimulus_file()
        vsyncs = data["items"]["behavior"]["intervalsms"]
        cum_sum = np.hstack((0, vsyncs)).cumsum() / 1000.0  # cumulative time
        offset = frame_time_offset(data)
        return cum_sum + offset

    def get_task_parameters(self) -> dict:
        """Get task parameters from pkl file.

        Returns
        -------
        dict
            A dictionary containing parameters used to define the task runtime
            behavior.
        """
        data = self._behavior_stimulus_file()
        return get_task_parameters(data)

    def get_trials(self) -> pd.DataFrame:
        """Get trials from pkl file

        Returns
        -------
        pd.DataFrame
            A dataframe containing behavioral trial start/stop times,
            and trial data
        """
        licks = self.get_licks()
        data = self._behavior_stimulus_file()
        rewards = self.get_rewards()
        stimulus_presentations = self.get_stimulus_presentations()
        # Pass a dummy rebase function since we don't have two time streams,
        # and the frame times are already aligned to trial events in their
        # respective getters
        trial_df = get_trials(data, licks, rewards, stimulus_presentations,
                              lambda x: x)

        return trial_df

    def get_extended_trials(self) -> pd.DataFrame:
        """Get extended trials from pkl file

        Returns
        -------
        pd.DataFrame
            A dataframe containing extended behavior trial information.
        """
        data = self._behavior_stimulus_file()
        return get_extended_trials(data)

    @memoize
    def get_experiment_date(self) -> datetime:
        """Return timestamp the behavior stimulus file began recording in UTC
        :rtype: datetime
        """
        data = self._behavior_stimulus_file()
        # Assuming file has local time of computer (Seattle)
        tz = pytz.timezone("America/Los_Angeles")
        return tz.localize(data["start_time"]).astimezone(pytz.utc)

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the session.
        :rtype: dict
        """
        if self.get_behavior_session_uuid() is None:
            bs_uuid = None
        else:
            bs_uuid = uuid.UUID(self.get_behavior_session_uuid())
        metadata = {
            "rig_name": self.get_rig_name(),
            "sex": self.get_sex(),
            "age": self.get_age(),
            "stimulus_frame_rate": self.get_stimulus_frame_rate(),
            "session_type": self.get_stimulus_name(),
            "experiment_datetime": self.get_experiment_date(),
            "reporter_line": self.get_reporter_line(),
            "driver_line": self.get_driver_line(),
            "LabTracks_ID": self.get_external_specimen_name(),
            "full_genotype": self.get_full_genotype(),
            "behavior_session_uuid": bs_uuid,
            "foraging_id": self.get_foraging_id(),
            "behavior_session_id": self.get_behavior_session_id()
        }
        return metadata

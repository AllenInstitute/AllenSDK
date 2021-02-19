import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
import warnings

import numpy as np
import pandas as pd
import pytz
from allensdk.api.cache import memoize
from allensdk.brain_observatory.behavior.metadata_processing import \
    get_task_parameters
from allensdk.brain_observatory.behavior.rewards_processing import get_rewards
from allensdk.brain_observatory.behavior.running_processing import \
    get_running_df
from allensdk.brain_observatory.behavior.session_apis.abcs import (
    BehaviorBase, BehaviorDataExtractorBase)
from allensdk.brain_observatory.behavior.stimulus_processing import (
    get_stimulus_metadata, get_stimulus_presentations, get_stimulus_templates,
    StimulusTemplate)
from allensdk.brain_observatory.behavior.trials_processing import (
    get_extended_trials, get_trials)
from allensdk.brain_observatory.running_speed import RunningSpeed
from allensdk.core.exceptions import DataFrameIndexError


class BehaviorDataTransforms(BehaviorBase):
    """This class provides methods that transform data extracted from
    LIMS or JSON data sources into final data products necessary for
    populating a BehaviorSession.
    """

    def __init__(self, extractor: BehaviorDataExtractorBase):
        self.extractor: BehaviorDataExtractorBase = extractor
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_behavior_session_id(self):
        return self.extractor.get_behavior_session_id()

    @memoize
    def _behavior_stimulus_file(self) -> pd.DataFrame:
        """Helper method to cache stimulus pkl file in memory since it takes
        about a second to load (and is used in many methods).
        """
        return pd.read_pickle(self.extractor.get_behavior_stimulus_file())

    def get_behavior_session_uuid(self) -> Optional[str]:
        """Get the universally unique identifier (UUID) number for the
        current behavior session.
        """
        data = self._behavior_stimulus_file()
        behavior_pkl_uuid = data.get("session_uuid")

        behavior_session_id = self.extractor.get_behavior_session_id()
        foraging_id = self.extractor.get_foraging_id()

        # Sanity check to ensure that pkl file data matches up with
        # the behavior session that the pkl file has been associated with.
        assert_err_msg = (
            f"The behavior session UUID ({behavior_pkl_uuid}) in the "
            f"behavior stimulus *.pkl file "
            f"({self.extractor.get_behavior_stimulus_file()}) does "
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

        :returns: pd.DataFrame
            Two columns: "time", which contains the sync time
            of the licks that occurred in this session and "frame",
            the frame numbers of licks that occurred in this session
        """
        # Get licks from pickle file instead of sync
        data = self._behavior_stimulus_file()
        stimulus_timestamps = self.get_stimulus_timestamps()
        lick_frames = (data["items"]["behavior"]["lick_sensors"][0]
                       ["lick_events"])
        lick_times = [stimulus_timestamps[frame] for frame in lick_frames]
        return pd.DataFrame({"time": lick_times, "frame": lick_frames})

    def get_rewards(self) -> pd.DataFrame:
        """Get reward data from pkl file, based on pkl file timestamps
        (not sync file).

        :returns: pd.DataFrame -- A dataframe containing timestamps of
            delivered rewards.
        """
        data = self._behavior_stimulus_file()
        timestamps = self.get_stimulus_timestamps()
        return get_rewards(data, timestamps)

    def get_running_acquisition_df(self, lowpass=True,
                                   zscore_threshold=10.0) -> pd.DataFrame:
        """Get running speed acquisition data from a behavior pickle file.

        NOTE: Rebases timestamps with the self.get_stimulus_timestamps()
        method which varies between the BehaviorDataTransformer and the
        BehaviorOphysDataTransformer.

        Parameters
        ----------
        lowpass: bool (default=True)
            Whether to apply a 10Hz low-pass filter to the running speed
            data.
        zscore_threshold: float
            The threshold to use for removing outlier running speeds which
            might be noise and not true signal

        Returns
        -------
        pd.DataFrame
            Dataframe with an index of timestamps and the following columns:
                "speed": computed running speed
                "dx": angular change, computed during data collection
                "v_sig": voltage signal from the encoder
                "v_in": the theoretical maximum voltage that the encoder
                    will reach prior to "wrapping". This should
                    theoretically be 5V (after crossing 5V goes to 0V, or
                    vice versa). In practice the encoder does not always
                    reach this value before wrapping, which can cause
                    transient spikes in speed at the voltage "wraps".
        """
        stimulus_timestamps = self.get_stimulus_timestamps()
        data = self._behavior_stimulus_file()
        return get_running_df(data, stimulus_timestamps, lowpass=lowpass,
                              zscore_threshold=zscore_threshold)

    def get_running_speed(self, lowpass=True) -> pd.DataFrame:
        """Get running speed using timestamps from
        self.get_stimulus_timestamps.

        NOTE: Do not correct for monitor delay.

        :returns: pd.DataFrame
            index: timestamps
            speed : subject's running speeds (in cm/s)
        """
        running_data_df = self.get_running_acquisition_df(lowpass=lowpass)
        if running_data_df.index.name != "timestamps":
            raise DataFrameIndexError(
                f"Expected index to be named 'timestamps' but got "
                f"'{running_data_df.index.name}'.")
        return pd.DataFrame({
            "timestamps": running_data_df.index.values,
            "speed": running_data_df.speed.values})

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

        # Sort columns then drop columns which contain only all NaN values
        return stim_pres_df[sorted(stim_pres_df)].dropna(axis=1, how='all')

    def get_stimulus_templates(self) -> Optional[StimulusTemplate]:
        """Get stimulus templates (movies, scenes) for behavior session.

        Returns
        -------
        StimulusTemplate or None if there are no images for the experiment
        """
        pkl = self._behavior_stimulus_file()
        return get_stimulus_templates(pkl=pkl)

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
        return cum_sum

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
        timestamps = self.get_stimulus_timestamps()
        licks = self.get_licks()
        data = self._behavior_stimulus_file()
        rewards = self.get_rewards()

        trial_df = get_trials(data,
                              licks,
                              rewards,
                              timestamps)

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
        """Return the timestamp for when experiment was started in UTC

        NOTE: This method will only get acquisition datetime from
        extractor (data from LIMS) methods. As a sanity check,
        it will also read the acquisition datetime from the behavior stimulus
        (*.pkl) file and raise a warning if the date differs too much from the
        datetime obtained from the behavior stimulus (*.pkl) file.

        :rtype: datetime
        """
        extractor_acq_date = self.extractor.get_experiment_date()

        pkl_data = self._behavior_stimulus_file()
        pkl_raw_acq_date = pkl_data["start_time"]
        if isinstance(pkl_raw_acq_date, datetime):
            pkl_acq_date = pytz.utc.localize(pkl_raw_acq_date)

        elif isinstance(pkl_raw_acq_date, (int, float)):
            # We are dealing with an older pkl file where the acq time is
            # stored as a Unix style timestamp string
            parsed_pkl_acq_date = datetime.fromtimestamp(pkl_raw_acq_date)
            pkl_acq_date = pytz.utc.localize(parsed_pkl_acq_date)
        else:
            pkl_acq_date = None
            warnings.warn(
                "Could not parse the acquisition datetime "
                f"({pkl_raw_acq_date}) found in the following stimulus *.pkl: "
                f"{self.extractor.get_behavior_stimulus_file()}"
            )

        if pkl_acq_date:
            acq_start_diff = (
                extractor_acq_date - pkl_acq_date).total_seconds()
            # If acquisition dates differ by more than an hour
            if abs(acq_start_diff) > 360:
                warnings.warn(
                    "The `date_of_acquisition` field in LIMS "
                    f"({extractor_acq_date}) for behavior session "
                    f"({self.get_behavior_session_id()}) deviates by more "
                    f"than an hour from the `start_time` ({pkl_acq_date}) "
                    "specified in the associated stimulus *.pkl file: "
                    f"{self.extractor.get_behavior_stimulus_file()}"
                )
        return extractor_acq_date

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the session.
        :rtype: dict
        """
        if self.get_behavior_session_uuid() is None:
            bs_uuid = None
        else:
            bs_uuid = uuid.UUID(self.get_behavior_session_uuid())
        metadata = {
            "rig_name": self.extractor.get_rig_name(),
            "sex": self.extractor.get_sex(),
            "age": self.extractor.get_age(),
            "stimulus_frame_rate": self.get_stimulus_frame_rate(),
            "session_type": self.extractor.get_stimulus_name(),
            "experiment_datetime": self.get_experiment_date(),
            "reporter_line": sorted(self.extractor.get_reporter_line()),
            "driver_line": sorted(self.extractor.get_driver_line()),
            "LabTracks_ID": self.extractor.get_external_specimen_name(),
            "full_genotype": self.extractor.get_full_genotype(),
            "behavior_session_uuid": bs_uuid,
            "behavior_session_id": self.extractor.get_behavior_session_id()
        }
        return metadata

import numpy as np
import pandas as pd
import uuid
from datetime import datetime
import pytz

from typing import Dict, Optional, Union, List, Any

from allensdk.core.exceptions import DataFrameIndexError
from allensdk.brain_observatory.behavior.internal.behavior_base import (
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
from allensdk.brain_observatory.behavior.trials_processing import get_trials
from allensdk.internal.core.lims_utilities import safe_system_path
from allensdk.internal.api import PostgresQueryMixin
from allensdk.api.cache import memoize
from allensdk.internal.api import (
    OneResultExpectedError, OneOrMoreResultExpectedError)
from allensdk.core.cache_method_utilities import CachedInstanceMethodMixin
from allensdk.core.authentication import DbCredentials, credential_injector
from allensdk.core.auth_config import (
    LIMS_DB_CREDENTIAL_MAP, MTRAIN_DB_CREDENTIAL_MAP)


class BehaviorDataLimsApi(CachedInstanceMethodMixin, BehaviorBase):
    def __init__(self, behavior_session_id: int,
                 lims_credentials: Optional[DbCredentials] = None,
                 mtrain_credentials: Optional[DbCredentials] = None):
        super().__init__()
        if mtrain_credentials:
            self.mtrain_db = PostgresQueryMixin(
                dbname=mtrain_credentials.dbname, user=mtrain_credentials.user,
                host=mtrain_credentials.host, port=mtrain_credentials.port,
                password=mtrain_credentials.password)
        else:
            self.mtrain_db = (credential_injector(MTRAIN_DB_CREDENTIAL_MAP)
                              (PostgresQueryMixin)())
        if lims_credentials:
            self.lims_db = PostgresQueryMixin(
                dbname=lims_credentials.dbname, user=lims_credentials.user,
                host=lims_credentials.host, port=lims_credentials.port,
                password=lims_credentials.password)
        else:
            self.lims_db = (credential_injector(LIMS_DB_CREDENTIAL_MAP)
                            (PostgresQueryMixin)())

        self.behavior_session_id = behavior_session_id
        ids = self._get_ids()
        self.ophys_experiment_ids = ids.get("ophys_experiment_ids")
        self.ophys_session_id = ids.get("ophys_session_id")
        self.behavior_training_id = ids.get("behavior_training_id")
        self.foraging_id = ids.get("foraging_id")
        self.ophys_container_id = ids.get("ophys_container_id")

    def _get_ids(self) -> Dict[str, Optional[Union[int, List[int]]]]:
        """Fetch ids associated with this behavior_session_id. If there is no
        id, return None.
        :returns: Dictionary of ids with the following keys:
            behavior_training_id: int -- Only if was a training session
            ophys_session_id: int -- None if have behavior_training_id
            ophys_experiment_ids: List[int] -- only if have ophys_session_id
            foraging_id: int
        :rtype: dict
        """
        # Get all ids from the behavior_sessions table
        query = f"""
            SELECT
                ophys_session_id, behavior_training_id, foraging_id
            FROM
                behavior_sessions
            WHERE
                behavior_sessions.id = {self.behavior_session_id};
        """
        ids_response = self.lims_db.select(query)
        if len(ids_response) > 1:
            raise OneResultExpectedError
        ids_dict = ids_response.iloc[0].to_dict()

        #  Get additional ids if also an ophys session
        #     (experiment_id, container_id)
        if ids_dict.get("ophys_session_id"):
            oed_query = f"""
                SELECT id
                FROM ophys_experiments
                WHERE ophys_session_id = {ids_dict["ophys_session_id"]};
                """
            oed = self.lims_db.fetchall(oed_query)
            if len(oed) == 0:
                oed = None

            container_query = f"""
            SELECT DISTINCT
                visual_behavior_experiment_container_id id
            FROM
                ophys_experiments_visual_behavior_experiment_containers
            WHERE
                ophys_experiment_id IN ({",".join(set(map(str, oed)))});
            """
            try:
                container_id = self.lims_db.fetchone(container_query, strict=True)
            except OneResultExpectedError:
                container_id = None

            ids_dict.update({"ophys_experiment_ids": oed,
                             "ophys_container_id": container_id})
        else:
            ids_dict.update({"ophys_experiment_ids": None,
                             "ophys_container_id": None})
        return ids_dict

    def get_behavior_session_id(self) -> int:
        """Getter to be consistent with BehaviorOphysLimsApi."""
        return self.behavior_session_id

    def get_behavior_session_uuid(self) -> Optional[int]:
        data = self._behavior_stimulus_file()
        return data.get("session_uuid")

    def get_behavior_stimulus_file(self) -> str:
        """Return the path to the StimulusPickle file for a session.
        :rtype: str
        """
        query = f"""
            SELECT
                stim.storage_directory || stim.filename AS stim_file
            FROM
                well_known_files stim
            WHERE
                stim.attachable_id = {self.behavior_session_id}
                AND stim.attachable_type = 'BehaviorSession'
                AND stim.well_known_file_type_id IN (
                    SELECT id
                    FROM well_known_file_types
                    WHERE name = 'StimulusPickle');
        """
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def _behavior_stimulus_file(self) -> pd.DataFrame:
        """Helper method to cache stimulus file in memory since it takes about
        a second to load (and is used in many methods).
        """
        return pd.read_pickle(self.get_behavior_stimulus_file())

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

    def get_running_data_df(self) -> pd.DataFrame:
        """Get running speed data.

        :returns: pd.DataFrame -- dataframe containing various signals used
            to compute running speed.
        """
        stimulus_timestamps = self.get_stimulus_timestamps()
        data = self._behavior_stimulus_file()
        return get_running_df(data, stimulus_timestamps)

    def get_running_speed(self) -> RunningSpeed:
        """Get running speed using timestamps from
        self.get_stimulus_timestamps.

        NOTE: Do not correct for monitor delay.

        :returns: RunningSpeed -- a NamedTuple containing the subject's
            timestamps and running speeds (in cm/s)
        """
        running_data_df = self.get_running_data_df()
        if running_data_df.index.name != "timestamps":
            raise DataFrameIndexError(
                f"Expected index to be named 'timestamps' but got "
                "'{running_data_df.index.name}'.")
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
            stimulus_index_df[["image_set", "image_index", "start_time"]]
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

    @memoize
    def get_birth_date(self) -> datetime.date:
        """Returns the birth date of the animal.
        :rtype: datetime.date
        """
        query = f"""
        SELECT d.date_of_birth
        FROM behavior_sessions bs
        JOIN donors d on d.id = bs.donor_id
        WHERE bs.id = {self.behavior_session_id}
        """
        return self.lims_db.fetchone(query, strict=True).date()

    @memoize
    def get_sex(self) -> str:
        """Returns sex of the animal (M/F)
        :rtype: str
        """
        query = f"""
            SELECT g.name AS sex
            FROM behavior_sessions bs
            JOIN donors d ON bs.donor_id = d.id
            JOIN genders g ON g.id = d.gender_id
            WHERE bs.id = {self.behavior_session_id};
            """
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_age(self) -> str:
        """Returns age code of the subject.
        :rtype: str
        """
        query = f"""
            SELECT a.name AS age
            FROM behavior_sessions bs
            JOIN donors d ON d.id = bs.donor_id
            JOIN ages a ON a.id = d.age_id
            WHERE bs.id = {self.behavior_session_id};
        """
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_rig_name(self) -> str:
        """Returns the name of the experimental rig.
        :rtype: str
        """
        query = f"""
            SELECT e.name AS device_name
            FROM behavior_sessions bs
            JOIN equipment e ON e.id = bs.equipment_id
            WHERE bs.id = {self.behavior_session_id};
        """
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_stimulus_name(self) -> str:
        """Returns the name of the stimulus set used for the session.
        :rtype: str
        """
        query = f"""
            SELECT stages.name
            FROM behavior_sessions bs
            JOIN stages ON stages.id = bs.state_id
            WHERE bs.id = '{self.foraging_id}'
        """
        return self.mtrain_db.fetchone(query, strict=True)

    @memoize
    def get_reporter_line(self) -> List[str]:
        """Returns the genotype name(s) of the reporter line(s).
        :rtype: list
        """
        query = f"""
            SELECT g.name AS reporter_line
            FROM behavior_sessions bs
            JOIN donors d ON bs.donor_id=d.id
            JOIN donors_genotypes dg ON dg.donor_id=d.id
            JOIN genotypes g ON g.id=dg.genotype_id
            JOIN genotype_types gt
                ON gt.id=g.genotype_type_id AND gt.name = 'reporter'
            WHERE bs.id={self.behavior_session_id};
        """
        result = self.lims_db.fetchall(query)
        if result is None or len(result) < 1:
            raise OneOrMoreResultExpectedError(
                f"Expected one or more, but received: '{result}' "
                f"from query:\n'{query}'")
        return result

    @memoize
    def get_driver_line(self) -> List[str]:
        """Returns the genotype name(s) of the driver line(s).
        :rtype: list
        """
        query = f"""
            SELECT g.name AS driver_line
            FROM behavior_sessions bs
            JOIN donors d ON bs.donor_id=d.id
            JOIN donors_genotypes dg ON dg.donor_id=d.id
            JOIN genotypes g ON g.id=dg.genotype_id
            JOIN genotype_types gt
                ON gt.id=g.genotype_type_id AND gt.name = 'driver'
            WHERE bs.id={self.behavior_session_id};
        """
        result = self.lims_db.fetchall(query)
        if result is None or len(result) < 1:
            raise OneOrMoreResultExpectedError(
                f"Expected one or more, but received: '{result}' "
                f"from query:\n'{query}'")
        return result

    @memoize
    def get_external_specimen_name(self) -> int:
        """Returns the LabTracks ID
        :rtype: int
        """
        # TODO: Should this even be included?
        # Found sometimes there were entries with NONE which is
        # why they are filtered out; also many entries in the table
        # match the donor_id, which is why used DISTINCT
        query = f"""
            SELECT DISTINCT(sp.external_specimen_name)
            FROM behavior_sessions bs
            JOIN donors d ON bs.donor_id=d.id
            JOIN specimens sp ON sp.donor_id=d.id
            WHERE bs.id={self.behavior_session_id}
            AND sp.external_specimen_name IS NOT NULL;
            """
        return int(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_full_genotype(self) -> str:
        """Return the name of the subject's genotype
        :rtype: str
        """
        query = f"""
                SELECT d.full_genotype
                FROM behavior_sessions bs
                JOIN donors d ON d.id=bs.donor_id
                WHERE bs.id= {self.behavior_session_id};
                """
        return self.lims_db.fetchone(query, strict=True)

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
            "ophys_experiment_id": self.ophys_experiment_ids,
            "experiment_container_id": self.ophys_container_id,
            "stimulus_frame_rate": self.get_stimulus_frame_rate(),
            "session_type": self.get_stimulus_name(),
            "experiment_datetime": self.get_experiment_date(),
            "reporter_line": self.get_reporter_line(),
            "driver_line": self.get_driver_line(),
            "LabTracks_ID": self.get_external_specimen_name(),
            "full_genotype": self.get_full_genotype(),
            "behavior_session_uuid": bs_uuid,
            "foraging_id": self.foraging_id,
            "behavior_session_id": self.behavior_session_id,
            "behavior_training_id": self.behavior_training_id,
        }
        return metadata

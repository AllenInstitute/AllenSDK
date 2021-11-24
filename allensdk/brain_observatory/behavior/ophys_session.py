import datetime
from typing import Any, List, Dict, Optional
import pynwb
import pandas as pd
import numpy as np
import pytz

from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_files import StimulusFile
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    JsonReadableInterface, NwbReadableInterface, \
    LimsReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base \
    .writable_interfaces import \
    NwbWritableInterface
from allensdk.brain_observatory.behavior.data_objects.licks import Licks
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .behavior_metadata.behavior_metadata import \
    BehaviorMetadata, get_expt_description
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.date_of_acquisition import \
    DateOfAcquisitionOphys
from allensdk.brain_observatory.behavior.data_objects.rewards import Rewards
from allensdk.brain_observatory.behavior.data_objects.stimuli.stimuli import \
    Stimuli
from allensdk.brain_observatory.behavior.data_objects.task_parameters import \
    TaskParameters
from allensdk.brain_observatory.behavior.data_objects.trials.trial_table \
    import \
    TrialTable
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.multi_plane_metadata.multi_plane_metadata \
    import \
    MultiplaneMetadata
from allensdk.brain_observatory.behavior.trials_processing import (
    construct_rolling_performance_df, calculate_reward_rate_fix_nans)
from allensdk.brain_observatory.behavior.data_objects import (
    OphysSessionId, StimulusTimestamps, RunningSpeed, RunningAcquisition,
    DataObject
)

from allensdk.core.auth_config import LIMS_DB_CREDENTIAL_MAP
from allensdk.internal.api import db_connection_creator, PostgresQueryMixin



class OphysSession(DataObject, LimsReadableInterface,
                      NwbReadableInterface,
                      JsonReadableInterface, NwbWritableInterface):
    """Represents data from a single Visual Behavior behavior session.
    Initialize by using class methods `from_lims` or `from_nwb_path`.
    """
    def __init__(
        self,
        ophys_session_id: OphysSessionId,
        stimulus_timestamps: StimulusTimestamps,
        running_acquisition: RunningAcquisition,
        raw_running_speed: RunningSpeed,
        running_speed: RunningSpeed,
        #stimuli: Stimuli,
        #task_parameters: TaskParameters,
        #trials: TrialTable,
        metadata: MultiplaneMetadata,
        date_of_acquisition: DateOfAcquisitionOphys
    ):
        super().__init__(name='ophys_session', value=self)

        self._ophys_session_id = ophys_session_id
        self._running_acquisition = running_acquisition
        self._running_speed = running_speed
        self._raw_running_speed = raw_running_speed
        #self._stimuli = stimuli
        self._stimulus_timestamps = stimulus_timestamps
        #self._task_parameters = task_parameters
        self._metadata = metadata
        #self._trials = trials
        self._date_of_acquisition = date_of_acquisition

    # ==================== class and utility methods ======================


    @classmethod
    def from_lims(cls, ophys_session_id: int,
                  lims_db: Optional[PostgresQueryMixin] = None,
                  stimulus_timestamps: Optional[StimulusTimestamps] = None,
                  monitor_delay: Optional[float] = None,
                  date_of_acquisition: Optional[DateOfAcquisitionOphys] = None) \
            -> "OphysSession":
        """

        Parameters
        ----------
        ophys_session_id
            ophys session id
        lims_db
            Database connection. If not provided will create a new one.
        stimulus_timestamps
            Stimulus timestamps. If not provided, will calculate stimulus
            timestamps from stimulus file.
        monitor_delay
            Monitor delay. If not provided, will use an estimate.
            To provide this value, see for example
            allensdk.brain_observatory.behavior.data_objects.stimuli.util.
            calculate_monitor_delay
        date_of_acquisition
            Date of acquisition. If not provided, will read from
            ophys_sessions table.
        Returns
        -------
        `ophysSession` instance
        """
        if lims_db is None:
            lims_db = db_connection_creator(
                fallback_credentials=LIMS_DB_CREDENTIAL_MAP
            )

        metadata = MultiplaneMetadata.from_lims(
            ophys_experiment_id=ophys_session_id, lims_db=lims_db
        )
        sess_id = OphysSessionId.from_lims(
            ophys_experiment_id=ophys_session_id, lims_db=lims_db
        )
        running_acquisition = RunningAcquisition.from_ophys_lims(
            db = lims_db,
            behavior_session_id = sess_id.value
        )
        raw_running_speed = RunningSpeed.from_lims(
            db = lims_db,
            behavior_session_id =sess_id.value,
            filtered=False,
            stimulus_timestamps=stimulus_timestamps
        )
        running_speed = RunningSpeed.from_lims(
            db = lims_db,
            behavior_session_id =sess_id.value,
            stimulus_timestamps=stimulus_timestamps
        )
        date_of_acquisition = DateOfAcquisitionOphys.from_lims(
            ophys_experiment_id=ophys_session_id, 
            lims_db=lims_db
        )
        if monitor_delay is None:
            monitor_delay = cls._get_monitor_delay()



        return OphysSession(
            ophys_session_id=ophys_session_id,
            stimulus_timestamps=stimulus_timestamps,
            metadata=metadata,
            raw_running_speed=raw_running_speed,
            running_acquisition=running_acquisition,
            running_speed=running_speed,
            date_of_acquisition=date_of_acquisition
            #trials=trials,
        )


    def to_nwb(self, add_metadata=False) -> NWBFile:
        """

        Parameters
        ----------
        add_metadata
            Set this to False to prevent adding metadata to the nwb
            instance.
        """
        #TODO: Updates session description and experiment description
        nwbfile = NWBFile(
            session_description='Ophys Session',
            identifier=self._get_identifier(),
            session_start_time=self._date_of_acquisition.value,
            file_create_date=pytz.utc.localize(datetime.datetime.now()),
            institution="Allen Institute for Brain Science",
            keywords=self._get_keywords(),
            experiment_description="ophys session"
        )

        self._stimulus_timestamps.to_nwb(nwbfile=nwbfile)
        self._running_acquisition.to_nwb(nwbfile=nwbfile)
        self._raw_running_speed.to_nwb(nwbfile=nwbfile)
        self._running_speed.to_nwb(nwbfile=nwbfile)
        #self._stimuli.to_nwb(nwbfile=nwbfile)
        #self._task_parameters.to_nwb(nwbfile=nwbfile)
        #self._trials.to_nwb(nwbfile=nwbfile)

        return nwbfile

    def list_data_attributes_and_methods(self) -> List[str]:
        """Convenience method for end-users to list attributes and methods
        that can be called to access data for a BehaviorSession.

        Returns
        -------
        List[str]
            A list of attributes and methods that end-users can access or call
            to get data.
        """
        attrs_and_methods_to_ignore: set = {
            "from_json",
            "from_lims",
            "from_nwb_path",
            "list_data_attributes_and_methods"
        }
        attrs_and_methods_to_ignore.update(dir(NwbReadableInterface))
        attrs_and_methods_to_ignore.update(dir(NwbWritableInterface))
        attrs_and_methods_to_ignore.update(dir(DataObject))
        class_dir = dir(self)
        attrs_and_methods = [
            r for r in class_dir
            if (r not in attrs_and_methods_to_ignore and not r.startswith("_"))
        ]
        return attrs_and_methods

    # ========================= 'get' methods ==========================

    def get_reward_rate(self) -> np.ndarray:
        """ Get the reward rate of the subject for the task calculated over a
        25 trial rolling window and provides a measure of the rewards
        earned per unit time (in units of rewards/minute).

        Returns
        -------
        np.ndarray
            The reward rate (rewards/minute) of the subject for the
            task calculated over a 25 trial rolling window.
        """
        return calculate_reward_rate_fix_nans(
                self.trials,
                self.task_parameters['response_window_sec'][0])

 

    # ====================== properties ========================

    @property
    def ophys(self) -> int:
        """Unique identifier for a ophys session.
        :rtype: int
        """
        return self._ophys_session_id.value

    @property
    def licks(self) -> pd.DataFrame:
        """A dataframe containing lick timestmaps and frames, sampled
        at 60 Hz.

        NOTE: For BehaviorSessions, returned timestamps are not
        aligned to external 'synchronization' reference timestamps.
        Synchronized timestamps are only available for
        BehaviorOphysExperiments.

        Returns
        -------
        np.ndarray
            A dataframe containing lick timestamps.
            dataframe columns:
                timestamps: (float)
                    time of lick, in seconds
                frame: (int)
                    frame of lick

        """
        return self._licks.value

    @property
    def rewards(self) -> pd.DataFrame:
        """Retrieves rewards from data file saved at the end of the
        behavior session.

        NOTE: For BehaviorSessions, returned timestamps are not
        aligned to external 'synchronization' reference timestamps.
        Synchronized timestamps are only available for
        BehaviorOphysExperiments.

        Returns
        -------
        pd.DataFrame
            A dataframe containing timestamps of delivered rewards.
            Timestamps are sampled at 60Hz.

            dataframe columns:
                volume: (float)
                    volume of individual water reward in ml.
                    0.007 if earned reward, 0.005 if auto reward.
                timestamps: (float)
                    time in seconds
                autorewarded: (bool)
                    True if free reward was delivered for that trial.
                    Occurs during the first 5 trials of a session and
                    throughout as needed

        """
        return self._rewards.value

    @property
    def running_speed(self) -> pd.DataFrame:
        """Running speed and timestamps, sampled at 60Hz. By default
        applies a 10Hz low pass filter to the data. To get the
        running speed without the filter, use `raw_running_speed`.


        Returns
        -------
        pd.DataFrame
            Dataframe containing running speed and timestamps
            dataframe columns:
                timestamps: (float)
                    time in seconds
                speed: (float)
                    speed in cm/sec
        """
        return self._running_speed.value

    @property
    def raw_running_speed(self) -> pd.DataFrame:
        """Get unfiltered running speed data. Sampled at 60Hz.


        Returns
        -------
        pd.DataFrame
            Dataframe containing unfiltered running speed and timestamps
            dataframe columns:
                timestamps: (float)
                    time in seconds
                speed: (float)
                    speed in cm/sec
        """
        return self._raw_running_speed.value

    @property
    def stimulus_presentations(self) -> pd.DataFrame:
        """Table whose rows are stimulus presentations (i.e. a given image,
        for a given duration, typically 250 ms) and whose columns are
        presentation characteristics.

        Returns
        -------
        pd.DataFrame
            Table whose rows are stimulus presentations
            (i.e. a given image, for a given duration, typically 250 ms)
            and whose columns are presentation characteristics.

            dataframe columns:
                stimulus_presentations_id [index]: (int)
                    identifier for a stimulus presentation
                    (presentation of an image)
                duration: (float)
                    duration of an image presentation (flash)
                    in seconds (stop_time - start_time). NaN if omitted
                end_frame: (float)
                    image presentation end frame
                image_index: (int)
                    image index (0-7) for a given session,
                    corresponding to each image name
                image_set: (string)
                    image set for this behavior session
                index: (int)
                    an index assigned to each stimulus presentation
                omitted: (bool)
                    True if no image was shown for this stimulus
                    presentation
                start_frame: (int)
                    image presentation start frame
                start_time: (float)
                    image presentation start time in seconds
                stop_time: (float)
                    image presentation end time in seconds
        """
        return self._stimuli.presentations.value

    @property
    def stimulus_templates(self) -> pd.DataFrame:
        """Get stimulus templates (movies, scenes) for behavior session.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame object containing the stimulus images for the
            experiment.

            dataframe columns:
                image_name [index]: (string)
                    name of image presented, if 'omitted'
                    then no image was presented
                unwarped: (array of int)
                    image array of unwarped stimulus image
                warped: (array of int)
                    image array of warped stimulus image

        """
        return self._stimuli.templates.value.to_dataframe()

    @property
    def stimulus_timestamps(self) -> np.ndarray:
        """Timestamps associated with the stimulus presetntation on
        the monitor retrieveddata file saved at the end of the
        ophys session. Sampled at 60Hz.

        Returns
        -------
        np.ndarray
            Timestamps associated with stimulus presentations on the monitor
        """
        return self._stimulus_timestamps.value

 

    @property
    def metadata(self) -> Dict[str, Any]:
        """metadata for a given session

        Returns
        -------
        Dict
            A dictionary containing behavior session specific metadata
            dictionary keys:
                age_in_days: (int)
                    age of mouse in days
                behavior_session_uuid: (int)
                    unique identifier for a behavior session
                behavior_session_id: (int)
                    unique identifier for a behavior session
                cre_line: (string)
                    cre driver line for a transgenic mouse
                date_of_acquisition: (date time object)
                    date and time of experiment acquisition,
                    yyyy-mm-dd hh:mm:ss
                driver_line: (list of string)
                    all driver lines for a transgenic mouse
                equipment_name: (string)
                    identifier for equipment data was collected on
                full_genotype: (string)
                    full genotype of transgenic mouse
                mouse_id: (int)
                    unique identifier for a mouse
                reporter_line: (string)
                    reporter line for a transgenic mouse
                session_type: (string)
                    visual stimulus type displayed during behavior
                    session
                sex: (string)
                    sex of the mouse
                stimulus_frame_rate: (float)
                    frame rate (Hz) at which the visual stimulus is
                    displayed
        """
        return self._get_metadata(behavior_metadata=self._metadata)

    @classmethod
    def _read_data_from_stimulus_file(
            cls, stimulus_file: StimulusFile,
            stimulus_timestamps: StimulusTimestamps,
            trial_monitor_delay: float):
        """Helper method to read data from stimulus file"""
        stimuli = Stimuli.from_stimulus_file(
            stimulus_file=stimulus_file,
            stimulus_timestamps=stimulus_timestamps)
        task_parameters = TaskParameters.from_stimulus_file(
            stimulus_file=stimulus_file)
        return stimuli, task_parameters



    def _get_identifier(self) -> str:
        return str(self._ophys_session_id)

    def _get_session_type(self) -> str:
        return self._metadata.session_type

    @staticmethod
    def _get_keywords():
        """Keywords for NWB file"""
        return ["visual", "behavior", "task"]

    @staticmethod
    def _get_monitor_delay():
        # This is the median estimate across all rigs
        # as discussed in
        # https://github.com/AllenInstitute/AllenSDK/issues/1318
        return 0.02115

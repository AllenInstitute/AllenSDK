from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from pynwb import NWBFile

from allensdk.brain_observatory import sync_utilities
from allensdk.brain_observatory.behavior.behavior_session import \
    BehaviorSession
from allensdk.brain_observatory.ecephys._behavior_ecephys_metadata import \
    BehaviorEcephysMetadata
from allensdk.brain_observatory.ecephys.optotagging import OptotaggingTable
from allensdk.brain_observatory.ecephys.probes import Probes

from allensdk.brain_observatory.behavior.data_files import SyncFile
from allensdk.brain_observatory.behavior.data_objects.licks import Licks
from allensdk.brain_observatory.behavior.data_objects import StimulusTimestamps
from allensdk.brain_observatory.behavior.behavior_session import (
    StimulusFileLookup)
from allensdk.brain_observatory.behavior.data_objects.stimuli.stimuli import (
    Stimuli)
from allensdk.brain_observatory.behavior.data_files.eye_tracking_file import \
    EyeTrackingFile
from allensdk.brain_observatory.behavior.\
    data_files.eye_tracking_metadata_file import EyeTrackingMetadataFile


from allensdk.brain_observatory.behavior.data_objects.eye_tracking \
    .eye_tracking_table import EyeTrackingTable, get_lost_frames


class VBNBehaviorSession(BehaviorSession):
    """
    A class to create the behavior parts of a VBN session,
    performing all of the specialized timestamp calculations
    that implies.
    """

    @classmethod
    def from_lims(cls, behavior_session_id: int,
                  lims_db: Optional[Any] = None,
                  sync_file: Optional[Any] = None,
                  monitor_delay: Optional[float] = None,
                  date_of_acquisition: Optional[Any] = None,
                  skip_eye_tracking=False,
                  eye_tracking_z_threshold: float = 3.0,
                  eye_tracking_dilation_frames: int = 2) \
            -> "VBNBehaviorSession":
        raise NotImplementedError(
                "from_lims is not supported for a VBNBehaviorSession")

    @classmethod
    def _read_stimuli(
            cls,
            stimulus_file_lookup: StimulusFileLookup,
            sync_file: Optional[SyncFile],
            monitor_delay: float,
            stimulus_presentation_columns: Optional[List[str]] = None
    ) -> Stimuli:
        raise NotImplementedError(
            "VBNBehaviorSessions read their stimulus tables from "
            "a precomputed csv file; they should not be computed "
            "on the fly by the AllenSDK")

    @classmethod
    def _read_behavior_stimulus_timestamps(
            cls,
            stimulus_file_lookup: StimulusFileLookup,
            sync_file: Optional[SyncFile],
            monitor_delay: float) -> StimulusTimestamps:
        """
        Assemble the StimulusTimestamps by registering behavior_
        mapping_ and replay_stimulus blocks to a single sync file
        """
        timestamps = StimulusTimestamps.from_multiple_stimulus_blocks(
                sync_file=sync_file,
                list_of_stims=[
                     stimulus_file_lookup.behavior_stimulus_file,
                     stimulus_file_lookup.mapping_stimulus_file,
                     stimulus_file_lookup.replay_stimulus_file],
                stims_of_interest=[0, ],
                monitor_delay=monitor_delay)
        return timestamps

    @classmethod
    def _read_session_timestamps(
            cls,
            stimulus_file_lookup: StimulusFileLookup,
            sync_file: Optional[SyncFile],
            monitor_delay: float) -> StimulusTimestamps:
        """
        Assemble the StimulusTimestamps (with monitor delay) that will
        be associated with this session
        """
        timestamps = StimulusTimestamps.from_multiple_stimulus_blocks(
                sync_file=sync_file,
                list_of_stims=[
                     stimulus_file_lookup.behavior_stimulus_file,
                     stimulus_file_lookup.mapping_stimulus_file,
                     stimulus_file_lookup.replay_stimulus_file],
                stims_of_interest=None,
                monitor_delay=monitor_delay)
        return timestamps

    @classmethod
    def _read_licks(
            cls,
            stimulus_file_lookup: StimulusFileLookup,
            sync_file: Optional[SyncFile]) -> Licks:
        """
        Construct the Licks data object for this session,
        reading the lick times directly from the sync file,
        accepting only those licks that occur during the time
        of the behavior stimulus block
        """
        lick_times = StimulusTimestamps(
                       timestamps=sync_file.data['lick_times'],
                       monitor_delay=0.0)

        # get the timestamps of the behavior stimulus presentations
        behavior_stim_times = cls._read_behavior_stimulus_timestamps(
                                 sync_file=sync_file,
                                 stimulus_file_lookup=stimulus_file_lookup,
                                 monitor_delay=0.0)

        # only accept lick times that are within the temporal bounds of
        # the behavior stimulus presentations
        min_time = behavior_stim_times.value.min()
        max_time = behavior_stim_times.value.max()

        valid = np.logical_and(
                  lick_times.value >= min_time,
                  lick_times.value <= max_time)

        lick_times = lick_times.value[valid]

        behavior_stim = stimulus_file_lookup.behavior_stimulus_file
        behavior_data = behavior_stim.data
        lick_sensor = behavior_data['items']['behavior']['lick_sensors'][0]
        lick_frames = lick_sensor['lick_events']

        if len(lick_frames) != len(lick_times):
            msg = (f"{len(lick_frames)} lick frames; "
                   f"{len(lick_times)} lick timestamps "
                   "in the Sync file. Should be equal")
            raise RuntimeError(msg)

        df = pd.DataFrame({"timestamps": lick_times,
                           "frame": lick_frames})
        return Licks(licks=df)

    @classmethod
    def _read_eye_tracking_table(
            cls,
            eye_tracking_file: EyeTrackingFile,
            eye_tracking_metadata_file: EyeTrackingMetadataFile,
            sync_file: SyncFile,
            z_threshold: float,
            dilation_frames: int) -> EyeTrackingTable:
        """
        Notes
        -----
        more or less copied from
        https://github.com/corbennett/NP_pipeline_QC/blob/6a66f195c4cd6b300776f089773577db542fe7eb/probeSync_qc.py
        """

        eye_metadata = eye_tracking_metadata_file.data
        camera_label = eye_metadata['RecordingReport']['CameraLabel']
        exposure_sync_line_label_dict = {
            'Eye': 'eye_cam_exposing',
            'Face': 'face_cam_exposing',
            'Behavior': 'beh_cam_exposing'}
        camera_line = exposure_sync_line_label_dict[camera_label]

        lost_frames = get_lost_frames(
                        eye_tracking_metadata=eye_tracking_metadata_file)

        frame_times = sync_utilities.get_synchronized_frame_times(
            session_sync_file=sync_file.filepath,
            sync_line_label_keys=(camera_line,),
            drop_frames=lost_frames,
            trim_after_spike=False)

        stimulus_timestamps = StimulusTimestamps(
                                timestamps=frame_times,
                                monitor_delay=0.0)

        return EyeTrackingTable.from_data_file(
                    data_file=eye_tracking_file,
                    stimulus_timestamps=stimulus_timestamps,
                    z_threshold=z_threshold,
                    dilation_frames=dilation_frames,
                    empty_on_fail=False)


class BehaviorEcephysSession(BehaviorSession):
    """
    Represents a session with behavior + ecephys
    """

    @classmethod
    def behavior_data_class(cls):
        """
        Return the class that is used to store the behavior data
        in this BehaviorEcephysSession
        """
        return VBNBehaviorSession

    def __init__(
            self,
            behavior_session: BehaviorSession,
            metadata: BehaviorEcephysMetadata,
            probes: Probes,
            optotagging_table: OptotaggingTable
    ):
        super().__init__(
            behavior_session_id=behavior_session._behavior_session_id,
            date_of_acquisition=behavior_session._date_of_acquisition,
            licks=behavior_session._licks,
            metadata=metadata,
            raw_running_speed=behavior_session._raw_running_speed,
            rewards=behavior_session._rewards,
            running_speed=behavior_session._running_speed,
            running_acquisition=behavior_session._running_acquisition,
            stimuli=behavior_session._stimuli,
            stimulus_timestamps=behavior_session._stimulus_timestamps,
            task_parameters=behavior_session._task_parameters,
            trials=behavior_session._trials,
            eye_tracking_table=behavior_session._eye_tracking,
            eye_tracking_rig_geometry=(
                behavior_session._eye_tracking_rig_geometry)
        )
        self._probes = probes
        self._optotagging_table = optotagging_table

    @property
    def probes(self) -> pd.DataFrame:
        """
        Returns
        -------
        A dataframe with columns
            - id: probe id
            - description: probe name
            - location: probe location
            - lfp_sampling_rate: LFP sampling rate
            - has_lfp_data: Whether this probe has LFP data
        """
        return self._probes.to_dataframe()

    @property
    def optotagging_table(self) -> pd.DataFrame:
        """

        Returns
        -------
        A dataframe with columns
            - start_time: onset of stimulation
            - condition: optical stimulation pattern
            - level: intensity (in volts output to the LED) of stimulation
            - stop_time: stop time of stimulation
            - stimulus_name: stimulus name
            - duration: duration of stimulation
        """
        return self._optotagging_table.value

    @property
    def metadata(self) -> dict:
        behavior_meta = super()._get_metadata(
            behavior_metadata=self._metadata)
        ecephys_meta = {
            'ecephys_session_id': self._metadata.ecephys_session_id
        }
        return {
            **behavior_meta,
            **ecephys_meta
        }

    @property
    def mean_waveforms(self) -> Dict[int, np.ndarray]:
        """

        Returns
        -------
        Dictionary mapping unit id to mean_waveforms for all probes
        """
        return self._probes.mean_waveforms

    @property
    def spike_times(self) -> Dict[int, np.ndarray]:
        """

        Returns
        -------
        Dictionary mapping unit id to spike_times for all probes
        """
        return self._probes.spike_times

    @property
    def spike_amplitudes(self) -> Dict[int, np.ndarray]:
        """

        Returns
        -------
        Dictionary mapping unit id to spike_amplitudes for all probes
        """
        return self._probes.spike_amplitudes

    def get_channels(self, filter_by_validity: bool = True) -> pd.DataFrame:
        """

        Parameters
        ----------
        filter_by_validity: Whether to filter channels based on whether
            the channel is marked as "valid_data"

        Returns
        -------
        `pd.DataFrame` of channels
        """
        return pd.concat([
            p.channels.to_dataframe(filter_by_validity=filter_by_validity)
            for p in self._probes.probes])

    def get_units(self, **kwargs) -> pd.DataFrame:
        """

        Parameters
        ----------
        kwargs: kwargs sent to `Probes.get_units_table`

        Returns
        -------
        `pd.DataFrame` of units detected by all probes
        """
        return self._probes.get_units_table(**kwargs)

    @classmethod
    def from_json(
            cls,
            session_data: dict,
            stimulus_presentation_exclude_columns: Optional[List[str]] = None,
            running_speed_load_from_multiple_stimulus_files: bool = True,
            skip_probes: Optional[List[str]] = None
    ) -> "BehaviorEcephysSession":
        """

        Parameters
        ----------
        session_data: Dict of input data necessary to construct a session
        stimulus_presentation_exclude_columns:  Optional list of columns to
            exclude from stimulus presentations table
        running_speed_load_from_multiple_stimulus_files:
            Whether to load running speed from multiple stimulus files
            If False, will just load from a single behavior stimulus file
        skip_probes: Names of probes to exclude (due to known bad data
            for example)

        Returns
        -------
        Instantiated `BehaviorEcephysSession`
        """

        behavior_session = cls.behavior_data_class().from_json(
            session_data=session_data,
            read_stimulus_presentations_table_from_file=True,
            stimulus_presentation_exclude_columns=(
                stimulus_presentation_exclude_columns),
            sync_file_permissive=True,
            eye_tracking_drop_frames=True,
            running_speed_load_from_multiple_stimulus_files=(
                running_speed_load_from_multiple_stimulus_files)
        )
        probes = Probes.from_json(probes=session_data['probes'],
                                  skip_probes=skip_probes)
        optotagging_table = OptotaggingTable.from_json(dict_repr=session_data)

        return BehaviorEcephysSession(
            behavior_session=behavior_session,
            probes=probes,
            optotagging_table=optotagging_table,
            metadata=BehaviorEcephysMetadata.from_json(dict_repr=session_data)
        )

    def to_nwb(self) -> NWBFile:
        nwbfile = super().to_nwb(
            add_metadata=False,
            include_experiment_description=False,
            stimulus_presentations_stimulus_column_name='stimulus_name')

        self._metadata.to_nwb(nwbfile=nwbfile)
        self._probes.to_nwb(nwbfile=nwbfile)
        self._optotagging_table.to_nwb(nwbfile=nwbfile)
        return nwbfile

    @classmethod
    def from_nwb(
            cls,
            nwbfile: NWBFile,
            **kwargs
    ) -> "BehaviorEcephysSession":
        """

        Parameters
        ----------
        nwbfile
        kwargs: kwargs sent to `BehaviorSession.from_nwb`

        Returns
        -------
        instantiated `BehaviorEcephysSession`
        """
        kwargs['add_is_change_to_stimulus_presentations_table'] = False
        behavior_session = cls.behavior_data_class().from_nwb(
            nwbfile=nwbfile,
            **kwargs
        )
        return BehaviorEcephysSession(
            behavior_session=behavior_session,
            probes=Probes.from_nwb(nwbfile=nwbfile),
            optotagging_table=OptotaggingTable.from_nwb(nwbfile=nwbfile),
            metadata=BehaviorEcephysMetadata.from_nwb(nwbfile=nwbfile)
        )

    def _get_identifier(self) -> str:
        return str(self._metadata.ecephys_session_id)

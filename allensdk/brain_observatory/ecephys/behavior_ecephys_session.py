from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from pynwb import NWBFile

from allensdk.brain_observatory.behavior.behavior_session import \
    BehaviorSession
from allensdk.brain_observatory.ecephys._behavior_ecephys_metadata import \
    BehaviorEcephysMetadata
from allensdk.brain_observatory.ecephys.optotagging import OptotaggingTable
from allensdk.brain_observatory.ecephys.probes import Probes


class BehaviorEcephysSession(BehaviorSession):
    """
    Represents a session with behavior + ecephys
    """
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
        running_speed_load_from_multiple_stimulus_files: Whether to load
            running speed from multiple stimulus files
            If False, will just load from a single behavior stimulus file
        skip_probes: Names of probes to exclude (due to known bad data
            for example)

        Returns
        -------
        Instantiated `BehaviorEcephysSession`
        """
        behavior_session = BehaviorSession.from_json(
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
        behavior_session = BehaviorSession.from_nwb(
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

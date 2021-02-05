import datetime
import uuid

import numpy as np
import pandas as pd
import pytz

from pynwb import NWBHDF5IO, NWBFile

import allensdk.brain_observatory.nwb as nwb
from allensdk.brain_observatory.behavior.metadata_processing import (
    get_expt_description
)
from allensdk.brain_observatory.behavior.session_apis.abcs import (
    BehaviorBase
)
from allensdk.brain_observatory.behavior.schemas import (
    BehaviorTaskParametersSchema, OphysBehaviorMetadataSchema)
from allensdk.brain_observatory.behavior.trials_processing import (
    TRIAL_COLUMN_DESCRIPTION_DICT
)
from allensdk.brain_observatory.nwb.metadata import load_pynwb_extension
from allensdk.brain_observatory.nwb.nwb_api import NwbApi
from allensdk.brain_observatory.nwb.nwb_utils import set_omitted_stop_time

load_pynwb_extension(OphysBehaviorMetadataSchema, 'ndx-aibs-behavior-ophys')
load_pynwb_extension(BehaviorTaskParametersSchema, 'ndx-aibs-behavior-ophys')


class BehaviorNwbApi(NwbApi, BehaviorBase):
    """A data fetching class that serves as an API for fetching 'raw'
    data from an NWB file that is both necessary and sufficient for filling
    a 'BehaviorOphysSession'.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save(self, session_object):

        session_type = str(session_object.metadata['session_type'])

        nwbfile = NWBFile(
            session_description=session_type,
            identifier=str(session_object.behavior_session_id),
            session_start_time=session_object.metadata['experiment_datetime'],
            file_create_date=pytz.utc.localize(datetime.datetime.now()),
            institution="Allen Institute for Brain Science",
            keywords=["visual", "behavior", "task"],
            experiment_description=get_expt_description(session_type)
        )

        # Add stimulus_timestamps to NWB in-memory object:
        nwb.add_stimulus_timestamps(nwbfile,
                                    session_object.stimulus_timestamps)

        # Add running data to NWB in-memory object:
        unit_dict = {'v_sig': 'V', 'v_in': 'V',
                     'speed': 'cm/s', 'timestamps': 's', 'dx': 'cm'}
        nwb.add_running_data_dfs_to_nwbfile(nwbfile,
                                            session_object.running_data_df,
                                            session_object.raw_running_data_df,
                                            unit_dict)

        # Add stimulus template data to NWB in-memory object:
        for name, image_data in session_object.stimulus_templates.items():
            nwb.add_stimulus_template(nwbfile, image_data, name)

            # Add index for this template to NWB in-memory object:
            nwb_template = nwbfile.stimulus_template[name]
            stimulus_index = session_object.stimulus_presentations[
                session_object.stimulus_presentations[
                    'image_set'] == nwb_template.name]
            nwb.add_stimulus_index(nwbfile, stimulus_index, nwb_template)

        # search for omitted rows and add stop_time before writing to NWB file
        set_omitted_stop_time(
            stimulus_table=session_object.stimulus_presentations)

        # Add stimulus presentations data to NWB in-memory object:
        nwb.add_stimulus_presentations(nwbfile,
                                       session_object.stimulus_presentations)

        # Add trials data to NWB in-memory object:
        nwb.add_trials(nwbfile, session_object.trials,
                       TRIAL_COLUMN_DESCRIPTION_DICT)

        # Add licks data to NWB in-memory object:
        if len(session_object.licks) > 0:
            nwb.add_licks(nwbfile, session_object.licks)

        # Add rewards data to NWB in-memory object:
        if len(session_object.rewards) > 0:
            nwb.add_rewards(nwbfile, session_object.rewards)

        # Add metadata to NWB in-memory object:
        nwb.add_metadata(nwbfile, session_object.metadata,
                         behavior_only=True)

        # Add task parameters to NWB in-memory object:
        nwb.add_task_parameters(nwbfile, session_object.task_parameters)

        # Write the file:
        with NWBHDF5IO(self.path, 'w') as nwb_file_writer:
            nwb_file_writer.write(nwbfile)

        return nwbfile

    def get_behavior_session_id(self) -> int:
        return int(self.nwbfile.identifier)

    def get_running_data_df(self,
                            lowpass: bool = True) -> pd.DataFrame:
        """
        Gets the running data df
        Parameters
        ----------
        lowpass: bool
            Whether to return running speed with or without low pass filter
            applied

        Returns
        -------
            pd.DataFrame:
                Dataframe containing various signals used to compute running
                speed, and the filtered or unfiltered speed.
        """
        running_speed = self.get_running_speed(lowpass=lowpass)

        running_data_df = pd.DataFrame({'speed': running_speed.values},
                                       index=pd.Index(running_speed.timestamps,
                                                      name='timestamps'))

        for key in ['v_in', 'v_sig']:
            if key in self.nwbfile.acquisition:
                running_data_df[key] = self.nwbfile.get_acquisition(key).data

        if 'running' in self.nwbfile.processing:
            running = self.nwbfile.processing['running']
            for key in ['dx']:
                if key in running.fields['data_interfaces']:
                    running_data_df[key] = running.get_data_interface(key).data

        return running_data_df[['speed', 'dx', 'v_sig', 'v_in']]

    def get_stimulus_templates(self, **kwargs):
        return {key: val.data[:]
                for key, val in self.nwbfile.stimulus_template.items()}

    def get_stimulus_timestamps(self) -> np.ndarray:
        stim_module = self.nwbfile.processing['stimulus']
        return stim_module.get_data_interface('timestamps').timestamps[:]

    def get_trials(self) -> pd.DataFrame:
        trials = self.nwbfile.trials.to_dataframe()
        if 'lick_events' in trials.columns:
            trials.drop('lick_events', inplace=True, axis=1)
        trials.index = trials.index.rename('trials_id')
        return trials

    def get_licks(self) -> np.ndarray:
        if 'licking' in self.nwbfile.processing:
            licks = (
                self.nwbfile.processing['licking'].get_data_interface('licks'))
            lick_timestamps = licks.timestamps[:]
            return pd.DataFrame({'time': lick_timestamps})
        else:
            return pd.DataFrame({'time': []})

    def get_rewards(self) -> np.ndarray:
        if 'rewards' in self.nwbfile.processing:
            rewards = self.nwbfile.processing['rewards']
            time = rewards.get_data_interface('autorewarded').timestamps[:]
            autorewarded = rewards.get_data_interface('autorewarded').data[:]
            volume = rewards.get_data_interface('volume').data[:]
            return pd.DataFrame({
                'volume': volume, 'timestamps': time,
                'autorewarded': autorewarded}).set_index('timestamps')
        else:
            return pd.DataFrame({
                'volume': [], 'timestamps': [],
                'autorewarded': []}).set_index('timestamps')

    def get_metadata(self) -> dict:

        metadata_nwb_obj = self.nwbfile.lab_meta_data['metadata']
        data = OphysBehaviorMetadataSchema(
            exclude=['experiment_datetime']).dump(metadata_nwb_obj)

        # Add pyNWB Subject metadata to behavior ophys session metadata
        nwb_subject = self.nwbfile.subject
        data['LabTracks_ID'] = int(nwb_subject.subject_id)
        data['sex'] = nwb_subject.sex
        data['age'] = nwb_subject.age
        data['full_genotype'] = nwb_subject.genotype
        data['reporter_line'] = list(nwb_subject.reporter_line)
        data['driver_line'] = list(nwb_subject.driver_line)

        # Add other metadata stored in nwb file to behavior ophys session meta
        data['experiment_datetime'] = self.nwbfile.session_start_time
        data['behavior_session_uuid'] = uuid.UUID(
            data['behavior_session_uuid'])
        return data

    def get_task_parameters(self) -> dict:

        metadata_nwb_obj = self.nwbfile.lab_meta_data['task_parameters']
        data = BehaviorTaskParametersSchema().dump(metadata_nwb_obj)
        return data

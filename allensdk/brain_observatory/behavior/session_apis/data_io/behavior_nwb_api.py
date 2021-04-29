import datetime
import uuid
from typing import Optional

import numpy as np
import pandas as pd
import pytz

from pynwb import NWBHDF5IO, NWBFile

import allensdk.brain_observatory.nwb as nwb
from allensdk.brain_observatory.behavior.metadata.behavior_metadata import (
    get_expt_description, BehaviorMetadata
)
from allensdk.brain_observatory.behavior.session_apis.abcs.\
    session_base.behavior_base import BehaviorBase
from allensdk.brain_observatory.behavior.schemas import (
    BehaviorTaskParametersSchema, OphysBehaviorMetadataSchema)
from allensdk.brain_observatory.behavior.stimulus_processing import \
    StimulusTemplate, StimulusTemplateFactory, is_change_event
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
    a 'BehaviorOphysExperiment'.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._behavior_session_id = None

    def save(self, session_object):

        session_metadata: BehaviorMetadata = \
            session_object.api.get_metadata()

        session_type = str(session_metadata.session_type)

        nwbfile = NWBFile(
            session_description=session_type,
            identifier=str(session_object.behavior_session_id),
            session_start_time=session_metadata.date_of_acquisition,
            file_create_date=pytz.utc.localize(datetime.datetime.now()),
            institution="Allen Institute for Brain Science",
            keywords=["visual", "behavior", "task"],
            experiment_description=get_expt_description(session_type)
        )

        # Add stimulus_timestamps to NWB in-memory object:
        session_object._stimulus_timestamps.to_nwb(nwbfile)

        # Add running acquisition ('dx', 'v_sig', 'v_in') data to NWB
        # This data should be saved to NWB but not accessible directly from
        # Sessions
        session_object._running_acquisition.to_nwb(nwbfile)

        # Add running data to NWB in-memory object:
        session_object._raw_running_speed.to_nwb(nwbfile)
        session_object._running_speed.to_nwb(nwbfile)

        # Add stimulus template data to NWB in-memory object:
        # Use the semi-private _stimulus_templates attribute because it is
        # a StimulusTemplate object. The public stimulus_templates property
        # of the session_object returns a DataFrame.
        session_stimulus_templates = session_object._stimulus_templates
        self._add_stimulus_templates(
            nwbfile=nwbfile,
            stimulus_templates=session_stimulus_templates,
            stimulus_presentations=session_object.stimulus_presentations)

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
        if self._behavior_session_id is None:
            self.get_metadata()
        return self._behavior_session_id

    def get_running_acquisition_df(self):
        raise NotImplementedError()

    def get_running_speed(self):
        raise NotImplementedError()

    def get_stimulus_templates(self, **kwargs) -> Optional[StimulusTemplate]:

        # If we have a session where only gratings were presented
        # there will be no stimulus_template dict in the nwbfile
        if len(self.nwbfile.stimulus_template) == 0:
            return None

        image_set_name = list(self.nwbfile.stimulus_template.keys())[0]
        image_data = list(self.nwbfile.stimulus_template.values())[0]

        image_attributes = [{'image_name': image_name}
                            for image_name in image_data.control_description]
        return StimulusTemplateFactory.from_processed(
            image_set_name=image_set_name, image_attributes=image_attributes,
            warped=image_data.data[:], unwarped=image_data.unwarped[:]
        )

    def get_stimulus_presentations(self) -> pd.DataFrame:
        df = super().get_stimulus_presentations()
        df['is_change'] = is_change_event(stimulus_presentations=df)
        return df

    def get_stimulus_timestamps(self):
        raise NotImplementedError

    def get_trials(self) -> pd.DataFrame:
        trials = self.nwbfile.trials.to_dataframe()
        if 'lick_events' in trials.columns:
            trials.drop('lick_events', inplace=True, axis=1)
        trials.index = trials.index.rename('trials_id')
        return trials

    def get_licks(self) -> np.ndarray:
        if 'licking' in self.nwbfile.processing:
            lick_module = self.nwbfile.processing['licking']
            licks = lick_module.get_data_interface('licks')

            return pd.DataFrame({
                'timestamps': licks.timestamps[:],
                'frame': licks.data[:]
            })
        else:
            return pd.DataFrame({'time': [], 'frame': []})

    def get_rewards(self) -> np.ndarray:
        if 'rewards' in self.nwbfile.processing:
            rewards = self.nwbfile.processing['rewards']
            time = rewards.get_data_interface('autorewarded').timestamps[:]
            autorewarded = rewards.get_data_interface('autorewarded').data[:]
            volume = rewards.get_data_interface('volume').data[:]
            return pd.DataFrame({
                'volume': volume, 'timestamps': time,
                'autorewarded': autorewarded})
        else:
            return pd.DataFrame({
                'volume': [], 'timestamps': [],
                'autorewarded': []})

    def get_metadata(self) -> dict:

        metadata_nwb_obj = self.nwbfile.lab_meta_data['metadata']
        data = OphysBehaviorMetadataSchema(
            exclude=['date_of_acquisition']).dump(metadata_nwb_obj)
        self._behavior_session_id = data["behavior_session_id"]

        # Add pyNWB Subject metadata to behavior session metadata
        nwb_subject = self.nwbfile.subject
        data['mouse_id'] = int(nwb_subject.subject_id)
        data['sex'] = nwb_subject.sex
        data['age_in_days'] = BehaviorMetadata.parse_age_in_days(
            age=nwb_subject.age)
        data['full_genotype'] = nwb_subject.genotype
        data['reporter_line'] = nwb_subject.reporter_line
        data['driver_line'] = sorted(list(nwb_subject.driver_line))
        data['cre_line'] = BehaviorMetadata.parse_cre_line(
            full_genotype=nwb_subject.genotype)
        data['indicator'] = BehaviorMetadata.parse_indicator(
            reporter_line=nwb_subject.reporter_line, warn=True
        )

        # Add other metadata stored in nwb file to behavior session meta
        data['date_of_acquisition'] = self.nwbfile.session_start_time
        data['behavior_session_uuid'] = uuid.UUID(
            data['behavior_session_uuid'])
        return data

    def get_task_parameters(self) -> dict:

        metadata_nwb_obj = self.nwbfile.lab_meta_data['task_parameters']
        data = BehaviorTaskParametersSchema().dump(metadata_nwb_obj)
        return data

    @staticmethod
    def _add_stimulus_templates(nwbfile: NWBFile,
                                stimulus_templates: StimulusTemplate,
                                stimulus_presentations: pd.DataFrame):
        nwb.add_stimulus_template(
            nwbfile=nwbfile, stimulus_template=stimulus_templates)

        # Add index for this template to NWB in-memory object:
        nwb_template = nwbfile.stimulus_template[
            stimulus_templates.image_set_name]
        stimulus_index = stimulus_presentations[
            stimulus_presentations[
                'image_set'] == nwb_template.name]
        nwb.add_stimulus_index(nwbfile, stimulus_index, nwb_template)

        return nwbfile

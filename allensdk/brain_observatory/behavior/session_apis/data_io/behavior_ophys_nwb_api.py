import datetime
import uuid
import warnings

import numpy as np
import pandas as pd
import pytz
import SimpleITK as sitk

from pynwb import NWBHDF5IO, NWBFile

import allensdk.brain_observatory.nwb as nwb
from allensdk.brain_observatory.behavior.metadata_processing import (
    get_expt_description
)
from allensdk.brain_observatory.behavior.session_apis.abcs import (
    BehaviorOphysBase
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


class BehaviorOphysNwbApi(NwbApi, BehaviorOphysBase):
    """A data fetching class that serves as an API for fetching 'raw'
    data from an NWB file that is both necessary and sufficient for filling
    a 'BehaviorOphysSession'.
    """

    def __init__(self, *args, **kwargs):
        self.filter_invalid_rois = kwargs.pop("filter_invalid_rois", False)
        super(BehaviorOphysNwbApi, self).__init__(*args, **kwargs)

    def save(self, session_object):

        session_type = str(session_object.metadata['session_type'])

        nwbfile = NWBFile(
            session_description=session_type,
            identifier=str(session_object.ophys_experiment_id),
            session_start_time=session_object.metadata['experiment_datetime'],
            file_create_date=pytz.utc.localize(datetime.datetime.now()),
            institution="Allen Institute for Brain Science",
            keywords=["2-photon", "calcium imaging", "visual cortex",
                      "behavior", "task"],
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
            stimulus_index = session_object.stimulus_presentations[session_object.stimulus_presentations['image_set'] == nwb_template.name]
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

        # Add max_projection image data to NWB in-memory object:
        nwb.add_max_projection(nwbfile, session_object.max_projection)

        # Add average_image image data to NWB in-memory object:
        nwb.add_average_image(nwbfile, session_object.average_projection)

        # Add segmentation_mask_image image data to NWB in-memory object:
        nwb.add_segmentation_mask_image(nwbfile,
                                        session_object.segmentation_mask_image)

        # Add metadata to NWB in-memory object:
        nwb.add_metadata(nwbfile, session_object.metadata)

        # Add task parameters to NWB in-memory object:
        nwb.add_task_parameters(nwbfile, session_object.task_parameters)

        # Add roi metrics to NWB in-memory object:
        nwb.add_cell_specimen_table(nwbfile,
                                    session_object.cell_specimen_table,
                                    session_object.metadata)

        # Add dff to NWB in-memory object:
        nwb.add_dff_traces(nwbfile, session_object.dff_traces,
                           session_object.ophys_timestamps)

        # Add corrected_fluorescence to NWB in-memory object:
        nwb.add_corrected_fluorescence_traces(
            nwbfile,
            session_object.corrected_fluorescence_traces)

        # Add motion correction to NWB in-memory object:
        nwb.add_motion_correction(nwbfile, session_object.motion_correction)

        # Write the file:
        with NWBHDF5IO(self.path, 'w') as nwb_file_writer:
            nwb_file_writer.write(nwbfile)

        return nwbfile

    def get_ophys_experiment_id(self) -> int:
        return int(self.nwbfile.identifier)

    # TODO: Implement save and load of behavior_session_id to/from NWB file
    def get_behavior_session_id(self) -> int:
        raise NotImplementedError()

    # TODO: Implement save and load of ophys_session_id to/from NWB file
    def get_ophys_session_id(self) -> int:
        raise NotImplementedError()

    # TODO: Implement save and load of eye_tracking_data to/from NWB file
    def get_eye_tracking(self) -> int:
        raise NotImplementedError()

    def get_running_data_df(self, lowpass=True) -> pd.DataFrame:
        """
        Gets the running data df
        Parameters
        ----------
        lowpass: bool
            Whether to return running speed with or without low pass filter applied

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

        for key in ['dx']:
            if ('running' in self.nwbfile.processing) and (key in self.nwbfile.processing['running'].fields['data_interfaces']):
                running_data_df[key] = self.nwbfile.processing['running'].get_data_interface(key).data

        return running_data_df[['speed', 'dx', 'v_sig', 'v_in']]

    def get_ophys_timestamps(self) -> np.ndarray:
        return self.nwbfile.processing['ophys'].get_data_interface('dff').roi_response_series['traces'].timestamps[:]

    def get_stimulus_templates(self, **kwargs):
        return {key: val.data[:]
                for key, val in self.nwbfile.stimulus_template.items()}

    def get_stimulus_timestamps(self) -> np.ndarray:
        return self.nwbfile.processing['stimulus'].get_data_interface('timestamps').timestamps[:]

    def get_trials(self) -> pd.DataFrame:
        trials = self.nwbfile.trials.to_dataframe()
        if 'lick_events' in trials.columns:
            trials.drop('lick_events', inplace=True, axis=1)
        trials.index = trials.index.rename('trials_id')
        return trials

    def get_licks(self) -> np.ndarray:
        if 'licking' in self.nwbfile.processing:
            return pd.DataFrame({'time': self.nwbfile.processing['licking'].get_data_interface('licks')['timestamps'].timestamps[:]})
        else:
            return pd.DataFrame({'time': []})

    def get_rewards(self) -> np.ndarray:
        if 'rewards' in self.nwbfile.processing:
            time = self.nwbfile.processing['rewards'].get_data_interface('autorewarded').timestamps[:]
            autorewarded = self.nwbfile.processing['rewards'].get_data_interface('autorewarded').data[:]
            volume = self.nwbfile.processing['rewards'].get_data_interface('volume').data[:]
            return pd.DataFrame({
                'volume': volume, 'timestamps': time,
                'autorewarded': autorewarded}).set_index('timestamps')
        else:
            return pd.DataFrame({
                'volume': [], 'timestamps': [], 
                'autorewarded': []}).set_index('timestamps')

    def get_max_projection(self, image_api=None) -> sitk.Image:
        return self.get_image('max_projection', 'ophys', image_api=image_api)

    def get_average_projection(self, image_api=None) -> sitk.Image:
        return self.get_image('average_image', 'ophys', image_api=image_api)

    def get_segmentation_mask_image(self, image_api=None) -> sitk.Image:
        return self.get_image('segmentation_mask_image',
                              'ophys', image_api=image_api)

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

        # Add pyNWB OpticalChannel and ImagingPlane metadata to behavior ophys
        # session metadata
        try:
            ophys_module = self.nwbfile.processing['ophys']
        except KeyError:
            warnings.warn("Could not locate 'ophys' module in "
                          "NWB file. The following metadata fields will be "
                          "missing: 'ophys_frame_rate', 'indicator', "
                          "'targeted_structure', 'excitation_lambda', "
                          "'emission_lambda'")
        else:
            image_seg = ophys_module.data_interfaces['image_segmentation']
            imaging_plane = image_seg.plane_segmentations['cell_specimen_table'].imaging_plane
            optical_channel = imaging_plane.optical_channel[0]

            data['ophys_frame_rate'] = imaging_plane.imaging_rate
            data['indicator'] = imaging_plane.indicator
            data['targeted_structure'] = imaging_plane.location
            data['excitation_lambda'] = imaging_plane.excitation_lambda
            data['emission_lambda'] = optical_channel.emission_lambda

        # Add other metadata stored in nwb file to behavior ophys session meta
        data['experiment_datetime'] = self.nwbfile.session_start_time
        data['behavior_session_uuid'] = uuid.UUID(
            data['behavior_session_uuid'])
        return data

    def get_task_parameters(self) -> dict:

        metadata_nwb_obj = self.nwbfile.lab_meta_data['task_parameters']
        data = BehaviorTaskParametersSchema().dump(metadata_nwb_obj)
        return data

    def get_cell_specimen_table(self) -> pd.DataFrame:
        # NOTE: ROI masks are stored in full frame width and height arrays
        df = self.nwbfile.processing['ophys'].data_interfaces['image_segmentation'].plane_segmentations['cell_specimen_table'].to_dataframe()

        # Because pynwb stores this field as "image_mask", it is renamed here
        df = df.rename(columns={'image_mask': 'roi_mask'})

        df.index.rename('cell_roi_id', inplace=True)
        df['cell_specimen_id'] = [None if csid == -1 else csid for csid in df['cell_specimen_id'].values]

        df.reset_index(inplace=True)
        df.set_index('cell_specimen_id', inplace=True)

        if self.filter_invalid_rois:
            df = df[df["valid_roi"]]

        return df

    def get_dff_traces(self) -> pd.DataFrame:
        dff_nwb = self.nwbfile.processing['ophys'].data_interfaces['dff'].roi_response_series['traces']
        # dff traces stored as timepoints x rois in NWB
        # We want rois x timepoints, hence the transpose
        dff_traces = dff_nwb.data[:].T
        number_of_cells, number_of_dff_frames = dff_traces.shape
        num_of_timestamps = len(self.get_ophys_timestamps())
        assert num_of_timestamps == number_of_dff_frames

        df = pd.DataFrame({'dff': dff_traces.tolist()},
                          index=pd.Index(data=dff_nwb.rois.table.id[:],
                                         name='cell_roi_id'))
        cell_specimen_table = self.get_cell_specimen_table()
        df = cell_specimen_table[['cell_roi_id']].join(df, on='cell_roi_id')
        return df

    def get_corrected_fluorescence_traces(self) -> pd.DataFrame:
        corrected_fluorescence_nwb = self.nwbfile.processing['ophys'].data_interfaces['corrected_fluorescence'].roi_response_series['traces']
        # f traces stored as timepoints x rois in NWB
        # We want rois x timepoints, hence the transpose
        f_traces = corrected_fluorescence_nwb.data[:].T
        df = pd.DataFrame({'corrected_fluorescence': f_traces.tolist()},
                          index=pd.Index(data=corrected_fluorescence_nwb.rois.table.id[:],
                                         name='cell_roi_id'))

        cell_specimen_table = self.get_cell_specimen_table()
        df = cell_specimen_table[['cell_roi_id']].join(df, on='cell_roi_id')
        return df

    def get_motion_correction(self) -> pd.DataFrame:
        ophys_module = self.nwbfile.processing['ophys']

        motion_correction_data = {}
        motion_correction_data['x'] = ophys_module.get_data_interface('ophys_motion_correction_x').data[:]
        motion_correction_data['y'] = ophys_module.get_data_interface('ophys_motion_correction_y').data[:]

        return pd.DataFrame(motion_correction_data)

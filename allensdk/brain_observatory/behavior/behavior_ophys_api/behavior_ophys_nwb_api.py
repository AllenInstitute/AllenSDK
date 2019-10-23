import datetime
from pynwb import NWBFile, NWBHDF5IO
import pandas as pd
import allensdk.brain_observatory.nwb as nwb
import numpy as np
import SimpleITK as sitk
import pytz
import uuid
from pandas.util.testing import assert_frame_equal
import os
import math
import numpy as np
import xarray as xr
import pandas as pd


from allensdk.core.lazy_property import LazyProperty
from allensdk.brain_observatory.nwb.nwb_api import NwbApi
from allensdk.brain_observatory.behavior.trials_processing import TRIAL_COLUMN_DESCRIPTION_DICT
from allensdk.brain_observatory.behavior.schemas import OphysBehaviorMetaDataSchema, OphysBehaviorTaskParametersSchema
from allensdk.brain_observatory.nwb.metadata import load_LabMetaData_extension
from allensdk.brain_observatory.behavior.behavior_ophys_api import BehaviorOphysApiBase


load_LabMetaData_extension(OphysBehaviorMetaDataSchema, 'AIBS_ophys_behavior')
load_LabMetaData_extension(OphysBehaviorTaskParametersSchema, 'AIBS_ophys_behavior')


class BehaviorOphysNwbApi(NwbApi, BehaviorOphysApiBase):


    def __init__(self, *args, **kwargs):
        self.filter_invalid_rois = kwargs.pop("filter_invalid_rois", False)
        super(BehaviorOphysNwbApi, self).__init__(*args, **kwargs)


    def save(self, session_object):

        nwbfile = NWBFile(
            session_description=str(session_object.metadata['session_type']),
            identifier=str(session_object.ophys_experiment_id),
            session_start_time=session_object.metadata['experiment_datetime'],
            file_create_date=pytz.utc.localize(datetime.datetime.now())
        )

        # Add stimulus_timestamps to NWB in-memory object:
        nwb.add_stimulus_timestamps(nwbfile, session_object.stimulus_timestamps)

        # Add running data to NWB in-memory object:
        unit_dict = {'v_sig': 'V', 'v_in': 'V', 'speed': 'cm/s', 'timestamps': 's', 'dx': 'cm'}
        nwb.add_running_data_df_to_nwbfile(nwbfile, session_object.running_data_df, unit_dict)

        # Add stimulus template data to NWB in-memory object:
        for name, image_data in session_object.stimulus_templates.items():
            nwb.add_stimulus_template(nwbfile, image_data, name)

            # Add index for this template to NWB in-memory object:
            nwb_template = nwbfile.stimulus_template[name]
            stimulus_index = session_object.stimulus_presentations[session_object.stimulus_presentations['image_set'] == nwb_template.name]
            nwb.add_stimulus_index(nwbfile, stimulus_index, nwb_template)

        # Add stimulus presentations data to NWB in-memory object:
        nwb.add_stimulus_presentations(nwbfile, session_object.stimulus_presentations)

        # Add trials data to NWB in-memory object:
        nwb.add_trials(nwbfile, session_object.trials, TRIAL_COLUMN_DESCRIPTION_DICT)

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
        nwb.add_segmentation_mask_image(nwbfile, session_object.segmentation_mask_image)

        # Add metadata to NWB in-memory object:
        nwb.add_metadata(nwbfile, session_object.metadata)

        # Add task parameters to NWB in-memory object:
        nwb.add_task_parameters(nwbfile, session_object.task_parameters)

        # Add roi metrics to NWB in-memory object:
        nwb.add_cell_specimen_table(nwbfile, session_object.cell_specimen_table)

        # Add dff to NWB in-memory object:
        nwb.add_dff_traces(nwbfile, session_object.dff_traces, session_object.ophys_timestamps)

        # Add corrected_fluorescence to NWB in-memory object:
        nwb.add_corrected_fluorescence_traces(nwbfile, session_object.corrected_fluorescence_traces)

        # Add motion correction to NWB in-memory object:
        nwb.add_motion_correction(nwbfile, session_object.motion_correction)

        # Write the file:
        with NWBHDF5IO(self.path, 'w') as nwb_file_writer:
            nwb_file_writer.write(nwbfile)

        return nwbfile

    def get_running_data_df(self, **kwargs):

        running_speed = self.get_running_speed()

        running_data_df = pd.DataFrame({'speed': running_speed.values},
                                       index=pd.Index(running_speed.timestamps, name='timestamps'))

        for key in ['v_in', 'v_sig']:
            if key in self.nwbfile.acquisition:
                running_data_df[key] = self.nwbfile.get_acquisition(key).data

        for key in ['dx']:
            if ('running' in self.nwbfile.modules) and (key in self.nwbfile.modules['running'].fields['data_interfaces']):
                running_data_df[key] = self.nwbfile.modules['running'].get_data_interface(key).data

        return running_data_df[['speed', 'dx', 'v_sig', 'v_in']]

    def get_stimulus_templates(self, **kwargs):
        return {key: val.data[:] for key, val in self.nwbfile.stimulus_template.items()}

    def get_ophys_timestamps(self) -> np.ndarray:
        return self.nwbfile.modules['two_photon_imaging'].get_data_interface('dff').roi_response_series['traces'].timestamps[:]

    def get_stimulus_timestamps(self) -> np.ndarray:
        return self.nwbfile.modules['stimulus'].get_data_interface('timestamps').timestamps[:]

    def get_trials(self) -> pd.DataFrame:
        trials = self.nwbfile.trials.to_dataframe()
        if 'lick_events' in trials.columns:
            trials.drop('lick_events', inplace=True, axis=1)
        trials.index = trials.index.rename('trials_id')
        return trials

    def get_licks(self) -> np.ndarray:
        if 'licking' in self.nwbfile.modules:
            return pd.DataFrame({'time': self.nwbfile.modules['licking'].get_data_interface('licks')['timestamps'].timestamps[:]})
        else:
            return pd.DataFrame({'time': []})

    def get_rewards(self) -> np.ndarray:
        if 'rewards' in self.nwbfile.modules:
            time = self.nwbfile.modules['rewards'].get_data_interface('autorewarded').timestamps[:]
            autorewarded = self.nwbfile.modules['rewards'].get_data_interface('autorewarded').data[:]
            volume = self.nwbfile.modules['rewards'].get_data_interface('volume').data[:]
            return pd.DataFrame({'volume': volume, 'timestamps': time, 'autorewarded': autorewarded}).set_index('timestamps')
        else:
            return pd.DataFrame({'volume': [], 'timestamps': [], 'autorewarded': []}).set_index('timestamps')

    def get_max_projection(self, image_api=None) -> sitk.Image:
        return self.get_image('max_projection', 'two_photon_imaging', image_api=image_api)

    def get_average_projection(self, image_api=None) -> sitk.Image:
        return self.get_image('average_image', 'two_photon_imaging', image_api=image_api)

    def get_segmentation_mask_image(self, image_api=None) -> sitk.Image:
        return self.get_image('segmentation_mask_image', 'two_photon_imaging', image_api=image_api)

    def get_metadata(self) -> dict:

        metadata_nwb_obj = self.nwbfile.lab_meta_data['metadata']
        data = OphysBehaviorMetaDataSchema(exclude=['experiment_datetime']).dump(metadata_nwb_obj)
        experiment_datetime = metadata_nwb_obj.experiment_datetime
        data['experiment_datetime'] = OphysBehaviorMetaDataSchema().load({'experiment_datetime': experiment_datetime}, partial=True)['experiment_datetime']            
        data['behavior_session_uuid'] = uuid.UUID(data['behavior_session_uuid'])
        return data

    def get_task_parameters(self) -> dict:

        metadata_nwb_obj = self.nwbfile.lab_meta_data['task_parameters']
        data = OphysBehaviorTaskParametersSchema().dump(metadata_nwb_obj)
        return data

    def get_cell_specimen_table(self) -> pd.DataFrame:
        df = self.nwbfile.modules['two_photon_imaging'].data_interfaces['image_segmentation'].plane_segmentations['cell_specimen_table'].to_dataframe()
        df.index.rename('cell_roi_id', inplace=True)
        df['cell_specimen_id'] = [None if csid == -1 else csid for csid in df['cell_specimen_id'].values]
        df['image_mask'] = [mask.astype(bool) for mask in df['image_mask'].values]
        df.reset_index(inplace=True)
        df.set_index('cell_specimen_id', inplace=True)

        if self.filter_invalid_rois:
            df = df[df["valid_roi"]]

        return df

    def get_dff_traces(self) -> pd.DataFrame:
        dff_nwb = self.nwbfile.modules['two_photon_imaging'].data_interfaces['dff'].roi_response_series['traces']
        dff_traces = dff_nwb.data[:]
        number_of_cells, number_of_dff_frames = dff_traces.shape
        num_of_timestamps = len(self.get_ophys_timestamps())
        assert num_of_timestamps == number_of_dff_frames
        
        df = pd.DataFrame({'dff': [x for x in dff_traces]}, index=pd.Index(data=dff_nwb.rois.table.id[:], name='cell_roi_id'))
        cell_specimen_table = self.get_cell_specimen_table()
        df = cell_specimen_table[['cell_roi_id']].join(df, on='cell_roi_id')
        return df

    def get_corrected_fluorescence_traces(self) -> pd.DataFrame:
        corrected_fluorescence_nwb = self.nwbfile.modules['two_photon_imaging'].data_interfaces['corrected_fluorescence'].roi_response_series['traces']
        df = pd.DataFrame({'corrected_fluorescence': [x for x in corrected_fluorescence_nwb.data[:]]},
                             index=pd.Index(data=corrected_fluorescence_nwb.rois.table.id[:], name='cell_roi_id'))

        cell_specimen_table = self.get_cell_specimen_table()
        df = cell_specimen_table[['cell_roi_id']].join(df, on='cell_roi_id')
        return df

    def get_motion_correction(self) -> pd.DataFrame:

        motion_correction_data = {}
        motion_correction_data['x'] = self.nwbfile.modules['motion_correction'].get_data_interface('x').data[:]
        motion_correction_data['y'] = self.nwbfile.modules['motion_correction'].get_data_interface('y').data[:]

        return pd.DataFrame(motion_correction_data)


def equals(A, B, reraise=False):

    field_set = set()
    for key, val in A.__dict__.items():
        if isinstance(val, LazyProperty):
            field_set.add(key)
    for key, val in B.__dict__.items():
        if isinstance(val, LazyProperty):
            field_set.add(key)

    try:
        for field in sorted(field_set):
            x1, x2 = getattr(A, field), getattr(B, field)
            err_msg = f"{field} on {A} did not equal {field} on {B} (\n{x1} vs\n{x2}\n)"
            compare_fields(x1, x2, err_msg)

    except NotImplementedError as e:
        A_implements_get_field = hasattr(A.api, getattr(type(A), field).getter_name)
        B_implements_get_field = hasattr(B.api, getattr(type(B), field).getter_name)
        assert A_implements_get_field == B_implements_get_field == False

    except (AssertionError, AttributeError) as e:
        if reraise:
            raise
        return False

    return True



def compare_fields(x1, x2, err_msg=""):
    if isinstance(x1, pd.DataFrame):
        try:
            assert_frame_equal(x1, x2, check_like=True)
        except:
            print(err_msg)
            raise
    elif isinstance(x1, np.ndarray):
        np.testing.assert_array_almost_equal(x1, x2, err_msg=err_msg)
    elif isinstance(x1, xr.DataArray):
        xr.testing.assert_allclose(x1, x2)
    elif isinstance(x1, (list,)):
        assert x1 == x2, err_msg
    elif isinstance(x1, (sitk.Image,)):
        assert x1.GetSize() == x2.GetSize(), err_msg
        assert x1 == x2, err_msg
    elif isinstance(x1, (dict,)):
        for key in set(x1.keys()).union(set(x2.keys())):
            key_err_msg = f"mismatch when checking key {key}. {err_msg}"

            if isinstance(x1[key], (np.ndarray,)):
                np.testing.assert_array_almost_equal(x1[key], x2[key], err_msg=key_err_msg)
            elif isinstance(x1[key], (float,)):
                if math.isnan(x1[key]) or math.isnan(x2[key]):
                    assert math.isnan(x1[key]) and math.isnan(x2[key]), key_err_msg
                else:
                    assert x1[key] == x2[key], key_err_msg
            else:
                assert x1[key] == x2[key], key_err_msg

    else:
        assert x1 == x2, err_msg
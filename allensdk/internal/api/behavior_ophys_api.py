import matplotlib.image as mpimg  # NOQA: E402
import numpy as np
import h5py
import pandas as pd

from allensdk.api.cache import memoize
from allensdk.internal.api.ophys_lims_api import OphysLimsApi
from allensdk.brain_observatory.behavior.sync import get_sync_data
from allensdk.brain_observatory.behavior.roi_processing import get_roi_metrics, get_roi_masks
from visual_behavior.translator import foraging2

class BehaviorOphysLimsApi(OphysLimsApi):

    @memoize
    def get_sync_data(self, ophys_experiment_id=None, use_acq_trigger=False):
        sync_path = self.get_sync_file(ophys_experiment_id=ophys_experiment_id)
        return get_sync_data(sync_path, use_acq_trigger=use_acq_trigger)


    @memoize
    def get_stimulus_timestamps(self, ophys_experiment_id=None, use_acq_trigger=False):
        return self.get_sync_data(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)['stimulus_frames']


    @memoize
    def get_ophys_timestamps(self, ophys_experiment_id=None, use_acq_trigger=False):
        return self.get_sync_data(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)['ophys_frames']


    @memoize
    def get_experiment_container_id(self, ophys_experiment_id=None):
        query = '''
                SELECT visual_behavior_experiment_container_id 
                FROM ophys_experiments_visual_behavior_experiment_containers 
                WHERE ophys_experiment_id= {};
                '''.format(ophys_experiment_id)        
        return self.fetchone(query, strict=False)


    @memoize
    def get_behavior_stimulus_file(self, ophys_experiment_id=None):
        query = '''
                SELECT stim.storage_directory || stim.filename AS stim_file
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN behavior_sessions bs ON bs.ophys_session_id=os.id
                LEFT JOIN well_known_files stim ON stim.attachable_id=bs.id AND stim.attachable_type = 'BehaviorSession' AND stim.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'StimulusPickle')
                WHERE oe.id= {};
                '''.format(ophys_experiment_id)
        return self.fetchone(query, strict=True)


    def get_behavior_session_uuid(self, ophys_experiment_id=None):
        behavior_stimulus_file = self.get_behavior_stimulus_file(ophys_experiment_id=ophys_experiment_id)
        data = pd.read_pickle(behavior_stimulus_file)
        return data['session_uuid']

    @memoize
    def get_stimulus_frame_rate(self, ophys_experiment_id=None, use_acq_trigger=False):
        stimulus_timestamps = self.get_stimulus_timestamps(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)
        return np.round(1 / np.mean(np.diff(stimulus_timestamps)), 0)


    @memoize
    def get_ophys_frame_rate(self, ophys_experiment_id=None, use_acq_trigger=False):
        ophys_timestamps = self.get_ophys_timestamps(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)
        return np.round(1 / np.mean(np.diff(ophys_timestamps)), 0)


    @memoize
    def get_metadata(self, ophys_experiment_id=None, use_acq_trigger=False):
        
        metadata = {}
        metadata['ophys_experiment_id'] = ophys_experiment_id
        metadata['experiment_container_id'] = self.get_experiment_container_id(ophys_experiment_id=ophys_experiment_id)
        metadata['ophys_frame_rate'] = self.get_ophys_frame_rate(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)
        metadata['stimulus_frame_rate'] = self.get_stimulus_frame_rate(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)
        metadata['targeted_structure'] = self.get_targeted_structure(ophys_experiment_id)
        metadata['imaging_depth'] = self.get_imaging_depth(ophys_experiment_id)
        metadata['session_type'] = self.get_stimulus_name(ophys_experiment_id)
        metadata['experiment_date'] = self.get_experiment_date(ophys_experiment_id)
        metadata['reporter_line'] = self.get_reporter_line(ophys_experiment_id)
        metadata['driver_line'] = self.get_driver_line(ophys_experiment_id)
        metadata['LabTracks_ID'] = self.get_LabTracks_ID(ophys_experiment_id)
        metadata['full_genotype'] = self.get_full_genotype(ophys_experiment_id)
        metadata['behavior_session_uuid'] = self.get_behavior_session_uuid(ophys_experiment_id)

        return metadata


    @memoize
    def get_dff_traces(self, ophys_experiment_id=None, use_acq_trigger=False):
        dff_path = self.get_dff_file(ophys_experiment_id=ophys_experiment_id)
        g = h5py.File(dff_path)
        dff_traces = np.asarray(g['data'])
        g.close()

        cell_roi_id_list = self.get_cell_roi_ids(ophys_experiment_id=ophys_experiment_id)
        df = pd.DataFrame({'cell_roi_id':cell_roi_id_list, 'dff':list(dff_traces)})
        return df


    @memoize
    def get_roi_metrics(self, ophys_experiment_id=None):
        input_extract_traces_file = self.get_input_extract_traces_file(ophys_experiment_id=ophys_experiment_id)
        objectlist_file = self.get_objectlist_file(ophys_experiment_id=ophys_experiment_id)
        return get_roi_metrics(input_extract_traces_file, ophys_experiment_id, objectlist_file)['unfiltered']


    @memoize
    def get_roi_masks(self, ophys_experiment_id=None):
        roi_metrics = self.get_roi_metrics( ophys_experiment_id=ophys_experiment_id)
        input_extract_traces_file = self.get_input_extract_traces_file(ophys_experiment_id=ophys_experiment_id)
        return get_roi_masks(roi_metrics, input_extract_traces_file)


    @memoize
    def get_core_data(self, ophys_experiment_id=None, use_acq_trigger=False):
        stim_filepath = self.get_behavior_stimulus_file(ophys_experiment_id=ophys_experiment_id)
        pkl = pd.read_pickle(stim_filepath)
        stimulus_timestamps = self.get_stimulus_timestamps(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)
        core_data = foraging2.data_to_change_detection_core(pkl, time=stimulus_timestamps)
        return core_data


    @memoize
    def get_running_speed(self, ophys_experiment_id=None, use_acq_trigger=False):
        return self.get_core_data(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)['running']
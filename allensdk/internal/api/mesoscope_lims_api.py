import pandas as pd
import logging
import json
import os
import uuid
import matplotlib.image as mpimg


from allensdk.api.cache import memoize
from allensdk.internal.core.lims_utilities import safe_system_path

from allensdk.brain_observatory.behavior.image_api import ImageApi
from allensdk.brain_observatory.behavior.sync import get_stimulus_rebase_function
from allensdk.brain_observatory.behavior.trials_processing import get_trials
from allensdk.brain_observatory.sync_dataset import Dataset as SyncDataset

from . import PostgresQueryMixin
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi

logger = logging.getLogger(__name__)


class MesoscopeSessionLimsApi(PostgresQueryMixin):

    def __init__(self, session_id):
        self.session_id = session_id
        self.experiment_ids = None
        self.pairs = None
        self.splitting_json = None
        self.session_folder = None
        self.session_df = None
        self.sync_path = None
        self.planes_timestamps = None
        super().__init__()

    def get_well_known_file(self, file_type):
        """Gets a well_known_file's location"""
        query = ' '.join(['SELECT wkf.storage_directory, wkf.filename FROM well_known_files wkf',
                           'JOIN well_known_file_types wkft',
                           'ON wkf.well_known_file_type_id = wkft.id',
                           'WHERE',
                           'attachable_id = {}'.format(self.session_id),
                           'AND wkft.name = \'{}\''.format(file_type)])

        query = query.format(self.session_id)
        filepath = pd.read_sql(query, self.get_connection())
        return filepath

    def get_session_id(self):
        return self.session_id

    def get_session_experiments(self):
        query = ' '.join((
            "SELECT oe.id as experiment_id",
            "FROM ophys_experiments oe",
            "WHERE oe.ophys_session_id = {}"
        ))
        self.experiment_ids = pd.read_sql(query.format(self.get_session_id()), self.get_connection())
        return self.experiment_ids

    def get_session_folder(self):
        _session = pd.DataFrame(self.get_session_df())
        session_folder = _session['session_folder']
        self.session_folder = safe_system_path(session_folder.values[0])
        return self.session_folder

    def get_session_df(self):
        query = ' '.join(("SELECT oe.id as experiment_id, os.id as session_id",
                    ", os.storage_directory as session_folder, oe.storage_directory as experiment_folder",
                    ", sp.name as specimen",
                    ", os.date_of_acquisition as date",
                    ", imaging_depths.depth as depth",
                    ", st.acronym as structure",
                    ", os.parent_session_id as parent_id",
                    ", oe.workflow_state",
                    ", os.stimulus_name as stimulus",
                    " FROM ophys_experiments oe",
                    "join ophys_sessions os on os.id = oe.ophys_session_id "
                    "join specimens sp on sp.id = os.specimen_id "
                    "join projects p on p.id = os.project_id "
                    "join imaging_depths on imaging_depths.id = oe.imaging_depth_id "
                    "join structures st on st.id = oe.targeted_structure_id "
                    " WHERE p.code in ('MesoscopeDevelopment', 'VisualBehaviorMultiscope') "
                    " AND oe.workflow_state in ('processing', 'qc', 'passed', 'failed') "
                    " AND os.workflow_state ='uploaded' "
                    " AND os.id='{}' ",
        ))
        query = query.format(self.get_session_id())
        self.session_df = pd.read_sql(query, self.get_connection())
        return self.session_df

    def get_splitting_json(self):
        session_folder = self.get_session_folder()
        """this info should not be read form splitting json, but right now this info is not stored in the database"""
        json_path = os.path.join(session_folder, f"MESOSCOPE_FILE_SPLITTING_QUEUE_{self.session_id}_input.json")
        self.splitting_json = safe_system_path(json_path)
        if not os.path.isfile(self.splitting_json):
            logger.error("Unable to find splitting json for session: {}".format(self.session_id))
        return self.splitting_json

    def get_paired_experiments(self):
        splitting_json = self.get_splitting_json()
        self.pairs = []
        with open(splitting_json, "r") as f:
            data = json.load(f)
        for pg in data.get("plane_groups", []):
            self.pairs.append([p["experiment_id"] for p in pg.get("ophys_experiments", [])])
        return self.pairs

    def get_sync_file(self):
            sync_file_df = self.get_well_known_file(file_type='OphysRigSync')
            sync_file_dir = safe_system_path(sync_file_df['storage_directory'].values[0])
            sync_file_name = sync_file_df['filename'].values[0]
            return os.path.join(sync_file_dir, sync_file_name)

    def get_sync_data(self):

        # let's fix line labels where they are off #

        sync_file = self.get_sync_file()
        sync_dataset = SyncDataset(sync_file)

        wrong_labels = ['vsync_2p', 'photodiode', 'cam1', 'cam2']
        correct_labels = ['2p_vsync', 'stim_photodiode', 'cam1_exposure', 'cam2_exposure']

        line_labels = sync_dataset.line_labels

        for line in line_labels:
            if line in wrong_labels :
                index = sync_dataset.line_labels.index(line)
                sync_dataset.line_labels.remove(line)
                sync_dataset.line_labels.insert(index, correct_labels[index])
        meta_data = sync_dataset.meta_data
        sample_freq = meta_data['ni_daq']['counter_output_freq']

        # 2P vsyncs
        vs2p_r = sync_dataset.get_rising_edges('2p_vsync')
        vs2p_f = sync_dataset.get_falling_edges('2p_vsync')  # new sync may be able to do units = 'sec', so conversion can be skipped
        frames_2p = vs2p_r / sample_freq
        vs2p_fsec = vs2p_f / sample_freq

        # use rising edge for Scientifica and Mesoscope falling edge for Nikon http://confluence.corp.alleninstitute.org/display/IT/Ophys+Time+Sync
        stimulus_times_no_monitor_delay = sync_dataset.get_rising_edges('stim_vsync') / sample_freq

        if 'lick_times' in meta_data['line_labels']:
            lick_times = sync_dataset.get_rising_edges('lick_1') / sample_freq
        elif 'lick_sensor' in meta_data['line_labels']:
            lick_times = sync_dataset.get_rising_edges('lick_sensor') / sample_freq
        else:
            lick_times = None
        if '2p_trigger' in meta_data['line_labels']:
            trigger = sync_dataset.get_rising_edges('2p_trigger') / sample_freq
        elif 'acq_trigger' in meta_data['line_labels']:
            trigger = sync_dataset.get_rising_edges('acq_trigger') / sample_freq
        if 'stim_photodiode' in meta_data['line_labels']:
            a = sync_dataset.get_rising_edges('stim_photodiode') / sample_freq
            b = sync_dataset.get_falling_edges('stim_photodiode') / sample_freq
            stim_photodiode = sorted(list(a)+list(b))
        elif 'photodiode' in meta_data['line_labels']:
            a = sync_dataset.get_rising_edges('photodiode') / sample_freq
            b = sync_dataset.get_falling_edges('photodiode') / sample_freq
            stim_photodiode = sorted(list(a)+list(b))
        if 'cam1_exposure' in meta_data['line_labels']:
            eye_tracking = sync_dataset.get_rising_edges('cam1_exposure') / sample_freq
        elif 'eye_tracking' in meta_data['line_labels']:
            eye_tracking = sync_dataset.get_rising_edges('eye_tracking') / sample_freq
        if 'cam2_exposure' in meta_data['line_labels']:
            behavior_monitoring = sync_dataset.get_rising_edges('cam2_exposure') / sample_freq
        elif 'behavior_monitoring' in meta_data['line_labels']:
            behavior_monitoring = sync_dataset.get_rising_edges('behavior_monitoring') / sample_freq

        sync_data = {'ophys_frames': frames_2p,
                     'lick_times': lick_times,
                     'ophys_trigger': trigger,
                     'eye_tracking': eye_tracking,
                     'behavior_monitoring': behavior_monitoring,
                     'stim_photodiode': stim_photodiode,
                     'stimulus_times_no_delay': stimulus_times_no_monitor_delay,
                     }

        return sync_data


    def split_session_timestamps(self):

        #this needs a check for dropped frames: compare timestamps with scanimage header's timestamps.

        timestamps = self.get_sync_data()['ophys_frames']
        planes_timestamps = pd.DataFrame(columns= ['plane_id', 'ophys_timestamps'], index = range(len(self.get_session_experiments())))
        pairs = self.get_paired_experiments()
        i = 0
        for pair in range(len(pairs)):
            planes_timestamps['plane_id'][i] = pairs[pair][0]
            planes_timestamps['plane_id'][i+1] = pairs[pair][1]
            planes_timestamps['ophys_timestamps'][i] = planes_timestamps['ophys_timestamps'][i+1] = timestamps[pair::len(pairs)]
            i += 2
        self.planes_timestamps = planes_timestamps
        return self.planes_timestamps


class MesoscopePlaneLimsApi(BehaviorOphysLimsApi):

    def __init__(self, experiment_id, session):
        self.experiment_id = experiment_id
        self.session = session
        self.session_id = None
        self.experiment_df = None
        self.ophys_timestamps = None
        super().__init__(experiment_id)

    def get_ophys_timestamps(self):
        if not self.session_id :
            self.get_ophys_session_id()

        plane_timestamps = self.session.get_plane_timestamps(self.ophys_experiment_id)
        self.ophys_timestamps = plane_timestamps
        return self.ophys_timestamps

    def get_experiment_df(self):

        api = PostgresQueryMixin()
        query = ''' 
                SELECT 
                
                oe.id as experiment_id, 
                os.id as session_id, 
                oe.storage_directory as experiment_folder,
                sp.name as specimen,
                os.date_of_acquisition as date,
                imaging_depths.depth as depth,
                st.acronym as structure,
                os.parent_session_id as parent_id,
                oe.workflow_state as workflow_state,
                os.stimulus_name as stimulus
                
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON os.id = oe.ophys_session_id 
                JOIN specimens sp ON sp.id = os.specimen_id  
                JOIN imaging_depths ON imaging_depths.id = oe.imaging_depth_id 
                JOIN structures st ON st.id = oe.targeted_structure_id 
                
                AND oe.id='{}'
                '''

        query = query.format(self.get_ophys_experiment_id())
        self.experiment_df = pd.read_sql(query, api.get_connection())
        return self.experiment_df

    def get_ophys_session_id(self):
        return self.session.session_id

    @memoize
    def get_metadata(self):

        metadata = super().get_metadata()
        metadata['ophys_experiment_id'] = self.get_ophys_experiment_id()
        metadata['experiment_container_id'] = self.get_experiment_container_id()
        metadata['ophys_frame_rate'] = self.get_ophys_frame_rate()
        metadata['stimulus_frame_rate'] = self.get_stimulus_frame_rate()
        metadata['targeted_structure'] = self.get_targeted_structure()
        metadata['imaging_depth'] = self.get_imaging_depth() #this is redefined below
        metadata['session_type'] = self.get_stimulus_name()
        metadata['experiment_datetime'] = self.get_experiment_date()
        metadata['reporter_line'] = self.get_reporter_line()
        metadata['driver_line'] = self.get_driver_line()
        metadata['LabTracks_ID'] = self.get_external_specimen_name()
        metadata['full_genotype'] = self.get_full_genotype()
        metadata['behavior_session_uuid'] = uuid.UUID(self.get_behavior_session_uuid())

        return metadata

    @memoize
    def get_imaging_depth(self):
        query = '''
                SELECT id.depth
                FROM ophys_experiments oe
                JOIN imaging_depths id ON id.id = oe.imaging_depth_id 
                WHERE oe.id= {};
                '''.format(self.get_ophys_experiment_id())
        return self.fetchone(query, strict=True)

    @memoize
    def get_max_projection(self, image_api=None):

        if image_api is None:
            image_api = ImageApi

        maxInt_a13_file = self.get_max_projection_file()
        if (self.get_surface_2p_pixel_size_um() == 0) :
            pixel_size = 400/512
        else : pixel_size = self.get_surface_2p_pixel_size_um()
        max_projection = mpimg.imread(maxInt_a13_file)
        return image_api.serialize(max_projection, [pixel_size / 1000., pixel_size / 1000.], 'mm')


    @memoize
    def get_average_projection(self, image_api=None):

        if image_api is None:
            image_api = ImageApi

        avgint_a1X_file = self.get_average_intensity_projection_image_file()
        if (self.get_surface_2p_pixel_size_um() == 0) :
            pixel_size = 400/512
        else : pixel_size = self.get_surface_2p_pixel_size_um()
        average_image = mpimg.imread(avgint_a1X_file)
        return image_api.serialize(average_image, [pixel_size / 1000., pixel_size / 1000.], 'mm')

    @memoize
    def get_segmentation_mask_image(self, image_api=None):

        if image_api is None:
            image_api = ImageApi

        segmentation_mask_image_file = self.get_segmentation_mask_image_file()
        if (self.get_surface_2p_pixel_size_um() == 0) :
            pixel_size = 400/512
        else : pixel_size = self.get_surface_2p_pixel_size_um()
        segmentation_mask_image = mpimg.imread(segmentation_mask_image_file)
        return image_api.serialize(segmentation_mask_image, [pixel_size / 1000., pixel_size / 1000.], 'mm')

    def get_licks(self):
        lick_times = self.get_sync_data()['lick_times']
        licks_df = pd.DataFrame({'time': lick_times})
        if licks_df.empty :
            behavior_stimulus_file = self.get_behavior_stimulus_file()
            data = pd.read_pickle(behavior_stimulus_file)
            lick_frames = data['items']['behavior']['lick_sensors'][0]['lick_events']
            stimulus_timestamps_no_monitor_delay = self.get_sync_data()['stimulus_times_no_delay'][:-1]
            lick_times  = stimulus_timestamps_no_monitor_delay[lick_frames]
            licks_df = pd.DataFrame({'time': lick_times})
        return licks_df

    @memoize
    def get_rewards(self):
        stimulus_timestamps_no_monitor_delay = self.get_sync_data()['stimulus_times_no_delay'][:-1]
        behavior_stimulus_file = self.get_behavior_stimulus_file()
        data = pd.read_pickle(behavior_stimulus_file)
        rebase_function = get_stimulus_rebase_function(data, stimulus_timestamps_no_monitor_delay)
        trial_df = pd.DataFrame(data["items"]["behavior"]['trial_log'])
        rewards_dict = {'volume': [], 'timestamps': [], 'autorewarded': []}
        for idx, trial in trial_df.iterrows():
            rewards = trial["rewards"]  # as i write this there can only ever be one reward per trial
            if rewards:
                rewards_dict["volume"].append(rewards[0][0])
                rewards_dict["timestamps"].append(rebase_function(rewards[0][1]))
                rewards_dict["autorewarded"].append('auto_rewarded' in trial['trial_params'])
        df = pd.DataFrame(rewards_dict).set_index('timestamps', drop=True)
        return df

    @memoize
    def get_trials(self):
        licks = self.get_licks()
        behavior_stimulus_file = self.get_behavior_stimulus_file()
        data = pd.read_pickle(behavior_stimulus_file)
        rewards = self.get_rewards()
        stimulus_timestamps_no_monitor_delay = self.get_sync_data()['stimulus_times_no_delay'][:-1]
        rebase_function = get_stimulus_rebase_function(data, stimulus_timestamps_no_monitor_delay)
        trial_df = get_trials(data, licks, rewards, rebase_function)
        return trial_df




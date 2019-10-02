from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi
from allensdk.brain_observatory.behavior.image_api import ImageApi
from allensdk.brain_observatory.behavior.sync import get_stimulus_rebase_function
from allensdk.brain_observatory.behavior.trials_processing import get_trials
from allensdk.brain_observatory.mesoscope.sync import get_sync_data
import uuid
import matplotlib.image as mpimg
from allensdk.api.cache import memoize
import pandas as pd
import logging
from . import PostgresQueryMixin
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

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
        lick_times = get_sync_data(self)['lick_times']
        licks_df = pd.DataFrame({'time': lick_times})
        if licks_df.empty :
            behavior_stimulus_file = self.get_behavior_stimulus_file()
            data = pd.read_pickle(behavior_stimulus_file)
            lick_frames = data['items']['behavior']['lick_sensors'][0]['lick_events']
            stimulus_timestamps_no_monitor_delay = get_sync_data(self)['stimulus_times_no_delay'][:-1]
            lick_times  = stimulus_timestamps_no_monitor_delay[lick_frames]
            licks_df = pd.DataFrame({'time': lick_times})
        return licks_df

    @memoize
    def get_rewards(self):
        stimulus_timestamps_no_monitor_delay = get_sync_data(self)['stimulus_times_no_delay'][:-1]
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
        stimulus_timestamps_no_monitor_delay = get_sync_data(self)['stimulus_times_no_delay'][:-1]
        rebase_function = get_stimulus_rebase_function(data, stimulus_timestamps_no_monitor_delay)
        trial_df = get_trials(data, licks, rewards, rebase_function)
        return trial_df



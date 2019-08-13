import os
import pandas as pd
import numpy as np
import json

from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import BehaviorOphysNwbApi
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.core.lazy_property import LazyProperty
from allensdk.brain_observatory.behavior.trials_processing import calculate_reward_rate

csv_io = {
    'reader': lambda path: pd.read_csv(path, index_col='Unnamed: 0'),
    'writer': lambda path, df: df.to_csv(path)
}

cache_json_example = {'manifest_path': '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/visual_behavior_data_manifest.csv',
                      'nwb_base_dir': '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/nwb_files',
                      'analysis_files_base_dir': '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/analysis_files',
                      'analysis_files_metadata_path':'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/analysis_files_metadata.json',
                      }

class BehaviorProjectCache(object):

    def __init__(self, cache_json):
        manifest = csv_io['reader'](cache_json['manifest_path'])
        self.manifest = manifest[[
            'ophys_experiment_id',
            'container_id',
            'full_genotype',
            'imaging_depth',
            'targeted_structure',
            'stage_name',
            'animal_name',
            'sex',
            'date_of_acquisition',
            'retake_number'
        ]]
        self.nwb_base_dir = cache_json['nwb_base_dir']
        self.analysis_files_base_dir = cache_json['analysis_files_base_dir']

        if 'analysis_files_metadata_path' in cache_json:
            self.analysis_files_metadata = self.get_analysis_files_metadata(cache_json['analysis_files_metadata_path'])
        else:
            print('Warning! No metadata supplied for analysis files. Set analysis_files_metadata_path to point at the json file containing the metadata')
            self.analysis_files_metadata = None

    def get_analysis_files_metadata(self, path):
        with open(path, 'r') as metadata_path:
            metadata = json.load(metadata_path)
        return metadata

    def get_nwb_filepath(self, experiment_id):
        return os.path.join(self.nwb_base_dir, 'behavior_ophys_session_{}.nwb'.format(experiment_id))

    def get_trial_response_df_path(self, experiment_id):
        return os.path.join(self.analysis_files_base_dir, 'trial_response_df_{}.h5'.format(experiment_id))

    def get_flash_response_df_path(self, experiment_id):
        return os.path.join(self.analysis_files_base_dir, 'flash_response_df_{}.h5'.format(experiment_id))

    def get_extended_stimulus_presentations_df(self, experiment_id):
        return os.path.join(self.analysis_files_base_dir, 'extended_stimulus_presentations_df_{}.h5'.format(experiment_id))

    def get_session(self, experiment_id):
        nwb_path = self.get_nwb_filepath(experiment_id)
        trial_response_df_path = self.get_trial_response_df_path(experiment_id)
        flash_response_df_path = self.get_flash_response_df_path(experiment_id)
        extended_stim_df_path = self.get_extended_stimulus_presentations_df(experiment_id)
        api = ExtendedNwbApi(nwb_path, trial_response_df_path, flash_response_df_path, extended_stim_df_path)
        session = ExtendedBehaviorSession(api)
        return session 

    def get_container_sessions(self, container_id):
        # TODO: Instead return a dict with stage name as key
        container_manifest = self.manifest.groupby('container_id').get_group(container_id)
        return [self.get_session(experiment_id) for experiment_id in container_manifest['ophys_experiment_id'].values]


class ExtendedNwbApi(BehaviorOphysNwbApi):
    
    def __init__(self, nwb_path, trial_response_df_path, flash_response_df_path, extended_stimulus_presentations_df_path):
        super(ExtendedNwbApi, self).__init__(path=nwb_path, filter_invalid_rois=True)
        self.trial_response_df_path = trial_response_df_path
        self.flash_response_df_path = flash_response_df_path
        self.extended_stimulus_presentations_df_path = extended_stimulus_presentations_df_path

    def get_trial_response_df(self):
        tdf = pd.read_hdf(self.trial_response_df_path, key='df')
        tdf.reset_index(level=1, inplace=True)
        # tdf.insert(loc=0, column='cell_specimen_id', value=tdf.index.values)
        tdf['cell_specimen_id'] = tdf.index.values #add this as a column to the end
        tdf.drop(columns=['cell_roi_id'], inplace=True)
        return tdf

    def get_flash_response_df(self):
        fdf = pd.read_hdf(self.flash_response_df_path, key='df')
        fdf.reset_index(level=1, inplace=True)
        # fdf.insert(loc=0, column='cell_specimen_id', value=fdf.index.values)
        fdf['cell_specimen_id'] = fdf.index.values #add this as a column to the end
        fdf.drop(columns=['image_name', 'cell_roi_id'], inplace=True)
        fdf = fdf.join(self.get_stimulus_presentations(), on='flash_id', how='left')
        return fdf

    def get_extended_stimulus_presentations_df(self):
        return pd.read_hdf(self.extended_stimulus_presentations_df_path, key='df')

    def get_task_parameters(self):
        # The task parameters are incorrect. See: https://github.com/AllenInstitute/AllenSDK/issues/637
        # We need to hard-code the omitted flash fraction and stimulus duration here. 
        task_parameters = super(ExtendedNwbApi, self).get_task_parameters()
        task_parameters['omitted_flash_fraction'] = 0.05
        task_parameters['stimulus_duration_sec'] = 0.25
        return task_parameters

    def get_trials(self):
        trials = super(ExtendedNwbApi, self).get_trials()
        stimulus_presentations = super(ExtendedNwbApi, self).get_stimulus_presentations()

        # Note: everything between dashed lines is a patch to deal with timing issues in the AllenSDK
        # This should be removed in the future after issues #876 and #802 are fixed.
        # -------------------------------------------------------------------------------------------------
        def get_next_flash(timestamp):
            # gets start_time of next stimulus after timestamp in stimulus_presentations 
            query = stimulus_presentations.query('start_time >= @timestamp')
            if len(query) > 0:
                return query.iloc[0]['start_time']
            else:
                return None
        trials['change_time'] = trials['change_time'].map(lambda x:get_next_flash(x))

        ### This method can lead to a NaN change time for any trials at the end of the session.
        ### However, aborted trials at the end of the session also don't have change times. 
        ### The safest method seems like just droping any trials that aren't covered by the stimulus_presentations
        #Using start time in case last stim is omitted
        last_stimulus_presentation = stimulus_presentations.iloc[-1]['start_time']
        trials = trials[np.logical_not(trials['stop_time'] > last_stimulus_presentation)]

        def recalculate_response_latency(row):
            # recalculates response latency based on corrected change time and first lick time
            if len(row['lick_times'] > 0) and not pd.isnull(row['change_time']):
                return row['lick_times'][0] - row['change_time']
            else:
                return np.nan
        trials['response_latency'] = trials.apply(recalculate_response_latency,axis=1)
        # -------------------------------------------------------------------------------------------------

        # asserts that every change time exists in the stimulus_presentations table
        for change_time in trials[trials['change_time'].notna()]['change_time']:
            assert change_time in stimulus_presentations['start_time'].values

        # Reorder / drop some columns to make more sense to students
        trials = trials[[
            'initial_image_name',
            'change_image_name',
            'change_time',
            'lick_times',
            'response_latency',
            'reward_time',
            'go',
            'catch',
            'hit',
            'miss',
            'false_alarm',
            'correct_reject',
            'aborted',
            'auto_rewarded',
            'reward_volume',
            'start_time',
            'stop_time',
            'trial_length'
        ]]


        trials['reward_rate'] = calculate_reward_rate(
            response_latency=trials.response_latency,
            starttime=trials.start_time,
            window=.75,
            trial_window=25,
            initial_trials=10
        )

        # Response_binary is just whether or not they responded - e.g. true for hit or FA. 
        hit = trials['hit'].values
        fa = trials['false_alarm'].values
        trials['response_binary'] = np.logical_or(hit, fa)

        return trials

    def get_stimulus_presentations(self):
        stimulus_presentations = super(ExtendedNwbApi, self).get_stimulus_presentations()
        extended_stimulus_presentations = self.get_extended_stimulus_presentations_df()
        extended_stimulus_presentations = extended_stimulus_presentations.drop(columns = ['omitted'])
        stimulus_presentations = stimulus_presentations.join(extended_stimulus_presentations)

        # Reorder the columns returned to make more sense to students
        stimulus_presentations = stimulus_presentations[[
            'image_name',
            'image_index',
            'start_time',
            'stop_time',
            'omitted',
            'change',
            'duration',
            'licks',
            'rewards',
            'running_speed',
            'index',
            'time_from_last_lick',
            'time_from_last_reward',
            'time_from_last_change',
            'block_index',
            'image_block_repetition',
            'repeat_within_block',
            'image_set'
        ]]

        # Rename some columns to make more sense to students
        stimulus_presentations = stimulus_presentations.rename(
            columns={'index':'absolute_flash_number'})
        return stimulus_presentations

    def get_stimulus_templates(self):
        stimulus_templates = super(ExtendedNwbApi, self).get_stimulus_templates()
        return stimulus_templates[list(stimulus_templates.keys())[0]]


class ExtendedBehaviorSession(BehaviorOphysSession):

    def __init__(self, api):

        super(ExtendedBehaviorSession, self).__init__(api)
        self.api = api

        self.trial_response_df = LazyProperty(self.get_trial_response_df)
        self.flash_response_df = LazyProperty(self.api.get_flash_response_df)
        self.image_index = LazyProperty(self.get_stimulus_index)

    def get_trial_response_df(self):
        trial_response_df = self.api.get_trial_response_df()
        return trial_response_df

    def get_stimulus_index(self):
        return self.stimulus_presentations.groupby('image_index').apply(lambda group: group['image_name'].unique()[0])



#  def test_project_cache():
if __name__=='__main__':

    cache_test_base = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019'
    cache_paths = {
        'manifest_path': os.path.join(cache_test_base, 'visual_behavior_data_manifest.csv'),
        'nwb_base_dir': os.path.join(cache_test_base, 'nwb_files'),
        'analysis_files_base_dir': os.path.join(cache_test_base, 'extra_files_final'),
        'analysis_files_metadata_path': os.path.join(cache_test_base, 'analysis_files_metadata.json')
    }

    cache = BehaviorProjectCache(cache_paths)
    session = cache.get_session(846487947)

    # Test trials extra columns
    for new_key in ['reward_rate', 'response_binary']:
        assert new_key in session.trials.keys()

    # Test stimulus_presentations extra columns
    for new_key in [
            'absolute_flash_number',
            'time_from_last_lick',
            'time_from_last_reward',
            'time_from_last_change',
            'block_index',
            'image_block_repetition',
            'repeat_within_block']:
        assert new_key in session.stimulus_presentations.keys()

    # Test trial response df
    trial_response = session.trial_response_df

    #  test_pairs = 
    #### These need to use almost_equal
    #  assert trial_response.loc[(915133315, 1)]['mean_response'] == 0.0440427
    #  assert trial_response.loc[(915133315, 1)]['baseline_response'] == 0.0640069
    #  assert trial_response.loc[(915133315, 1)]['p_value'] == 0.247066
    #  assert trial_response.loc[(915133315, 1)]['go'] == False
    #  assert trial_response.loc[(915133315, 1)]['initial_image_name'] == 'im000'
    #  

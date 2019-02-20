from allensdk.api.cache import memoize

from . import PostgresQueryMixin

class BehaviorLimsApi(PostgresQueryMixin):

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


    @memoize
    def get_core_data(self, ophys_session_id=None):
        stim_filepath = self.filepath_api.get_behavior_stimulus_file(ophys_experiment_id = ophys_experiment_id)
        pkl = pd.read_pickle(stim_filepath)




        stimulus_timestamps = self.get_stimulus_timestamps(*args, ophys_experiment_id=ophys_experiment_id, **kwargs)
        try:
            core_data = foraging.data_to_change_detection_core(pkl, time=stimulus_timestamps)
        except KeyError:
            core_data = foraging2.data_to_change_detection_core(pkl, time=stimulus_timestamps)
        return core_data


# import pandas as pd

# from visual_behavior.translator import foraging2, foraging
# from allensdk.api.cache import memoize

# from .lims_behavior_api import LimsBehaviorAPI

# class BehaviorApi(object):

#     def __init__(self, filepath_api=None):
        
#         if filepath_api is None:
#             self.filepath_api = LimsBehaviorAPI()
#         else:
#             self.filepath_api = filepath_api

#     @memoize
#     def get_sync_data(self, session_id=None):
#         sync_path = self.get_sync_file(session_id)
#         return get_sync_data(sync_path)


#     @memoize
#     def get_core_data(self, session_id=None):
#         stim_filepath = self.filepath_api.get_behavior_stimulus_file(ophys_experiment_id=ophys_experiment_id)
#         pkl = pd.read_pickle(stim_filepath)




#         stimulus_timestamps = self.get_stimulus_timestamps(*args, ophys_experiment_id=ophys_experiment_id, **kwargs)
#         try:
#             core_data = foraging.data_to_change_detection_core(pkl, time=stimulus_timestamps)
#         except KeyError:
#             core_data = foraging2.data_to_change_detection_core(pkl, time=stimulus_timestamps)
#         return core_data
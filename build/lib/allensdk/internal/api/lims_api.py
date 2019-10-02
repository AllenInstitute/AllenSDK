import pandas as pd

from allensdk.internal.api import PostgresQueryMixin
from allensdk.internal.core.lims_utilities import safe_system_path


class LimsApi(PostgresQueryMixin):

    def __init__(self):
        super().__init__()

    def get_experiment_id(self):
        return self.experiment_id

    def get_behavior_tracking_video_filepath_df(self):
        query = '''
                SELECT wkf.storage_directory || wkf.filename AS raw_behavior_tracking_video_filepath, attachable_type 
                FROM well_known_files wkf WHERE wkf.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'RawBehaviorTrackingVideo')
                '''
        return pd.read_sql(query, self.get_connection())

    def get_eye_tracking_video_filepath_df(self):
        query = '''
                SELECT wkf.storage_directory || wkf.filename AS raw_behavior_tracking_video_filepath, attachable_type 
                FROM well_known_files wkf WHERE wkf.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'RawEyeTrackingVideo')
                '''
        return pd.read_sql(query, self.get_connection())


if __name__ == "__main__":

    api = LimsApi()
    for ii in range(5):
        print(api.get_eye_tracking_video_filepath_df().loc[ii].raw_behavior_tracking_video_filepath)

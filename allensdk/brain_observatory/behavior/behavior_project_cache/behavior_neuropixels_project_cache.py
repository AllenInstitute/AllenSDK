from allensdk.brain_observatory.behavior.behavior_project_cache.\
    project_apis.data_io import VisualBehaviorNeuropixelsProjectCloudApi
from allensdk.brain_observatory.behavior.behavior_project_cache.\
    project_cache_base import ProjectCacheBase


class VisualBehaviorNeuropixelsProjectCache(ProjectCacheBase):

    PROJECT_NAME = "visual-behavior-ecephys"
    BUCKET_NAME = "visual-behavior-ecephys-data"

    def __init__(
            self,
            fetch_api: VisualBehaviorNeuropixelsProjectCloudApi,
            fetch_tries: int = 2,
            ):
        """ Entrypoint for accessing Visual Behavior Neuropixels data.

        Supports access to metadata tables:
        get_ecephys_session_table()
        get_behavior_session_table()
        get_probe_table()
        get_channel_table()
        get_unit_table

        Provides methods for instantiating session objects
        from the nwb files:
        get_ecephys_session() to load BehaviorEcephysSession
        get_behavior_sesion() to load BehaviorSession

        Provides tools for downloading data:

        Will download data from the s3 bucket if session nwb file is not
        in the local cache, othwerwise will use file from the cache.

        """
        super().__init__(fetch_api=fetch_api, fetch_tries=fetch_tries)

    @classmethod
    def cloud_api_class(cls):
        return VisualBehaviorNeuropixelsProjectCloudApi

    def get_ecephys_session_table(self):
        return self.fetch_api.get_ecephys_session_table(),

    def get_behavior_session_table(self):
        return self.fetch_api.get_behavior_session_table(),

    def get_probe_table(self):
        self.fetch_api.get_probe_table(),

    def get_channel_table(self):
        return self.fetch_api.get_channel_table(),

    def get_unit_table(self):
        return self.fetch_api.get_unit_table(),

    def get_ecephys_session(self, ecephys_session_id: int):
        return self.fetch_api.get_ecephys_session(ecephys_session_id)

    def get_behavior_session(self, behavior_session_id: int):
        return self.fetch_api.get_behavior_session(behavior_session_id)

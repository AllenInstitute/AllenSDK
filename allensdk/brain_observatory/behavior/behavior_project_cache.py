import pandas as pd
from allensdk.api.cache import Cache
from allensdk.brain_observatory.ecephys.ecephys_project_cache import call_caching
from allensdk.brain_observatory.ecephys.file_promise import write_from_stream
from allensdk.brain_observatory.behavior.behavior_project_api.behavior_project_fixed_api import BehaviorProjectFixedApi
from allensdk.brain_observatory.behavior.behavior_project_api.behavior_project_lims_api import BehaviorProjectLimsApi
from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import BehaviorOphysNwbApi, ExtendedBehaviorOphysNwbApi
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession, ExtendedBehaviorOphysSession

csv_io = {
    'reader': lambda path: pd.read_csv(path, index_col='Unnamed: 0'),
    'writer': lambda path, df: df.to_csv(path)
}

class BehaviorProjectCache(Cache):

    SESSIONS_KEY = 'sessions'

    NWB_DIR_KEY = 'nwb_files'
    SESSION_NWB_KEY = 'session_nwb'

    ANALYSIS_FILES_DIR_KEY = 'analysis_files'
    TRIAL_RESPONSE_KEY = 'trial_response'
    FLASH_RESPONSE_KEY = 'flash_response'
    EXTENDED_STIM_COLUMNS_KEY = 'extended_stimulus_presentations'
    EXPOSURE_NUMBER_KEY = 'exposure_number'

    MANIFEST_VERSION = '0.0.1'

    def __init__(self, fetch_api, **kwargs):
        
        kwargs['manifest'] = kwargs.get('manifest', 'behavior_project_manifest.json')
        kwargs['version'] = kwargs.get('version', self.MANIFEST_VERSION)

        super(BehaviorProjectCache, self).__init__(**kwargs)
        self.fetch_api = fetch_api

    def get_sessions(self):
        path = self.get_cache_path(None, self.SESSIONS_KEY)
        return call_caching(self.fetch_api.get_sessions, path=path, strategy='lazy', writer=csv_io["writer"], reader=csv_io['reader'])

    def get_session_data(self, experiment_id):
        # TODO: Use session ID here? will need to specify per-plane if we want to support mesoscope though, which would be experiment ID
        # Although, it would be better if a session was really a session, and had n Planes attached (each currently a different 'experiment')
        nwb_path = self.get_cache_path(None, self.SESSION_NWB_KEY, experiment_id)
        trial_response_path = self.get_cache_path(None, self.TRIAL_RESPONSE_KEY, experiment_id)
        flash_response_path = self.get_cache_path(None, self.FLASH_RESPONSE_KEY, experiment_id)
        extended_stim_path = self.get_cache_path(None, self.EXTENDED_STIM_COLUMNS_KEY, experiment_id)

        #  fetch_api_fns = zip(
        #      [self.fetch_api.get_session_nwb,
        #       self.fetch_api.get_session_trial_response,
        #       self.fetch_api.get_session_flash_response,
        #       self.fetch_api.get_session_extended_stimulus_columns],
        #      [nwb_path, 
        #       trial_response_path,
        #       flash_response_path,
        #       extended_stim_path]
        #  )
        fetch_api_fns = zip(
            [self.fetch_api.get_session_nwb],
            [nwb_path]
        )

        for fetch_api_fn, path in fetch_api_fns:
            call_caching(fetch_api_fn, 
                         path, 
                         experiment_id=experiment_id, 
                         strategy='lazy',
                         writer=write_from_stream)

        session_api = BehaviorOphysNwbApi(
            path=nwb_path,
        )
        return BehaviorOphysSession(api=session_api)

    def add_manifest_paths(self, manifest_builder):
        manifest_builder = super(BehaviorProjectCache, self).add_manifest_paths(manifest_builder)
                                  
        manifest_builder.add_path(
            self.SESSIONS_KEY, 'sessions.csv', parent_key='BASEDIR', typename='file'
        )

        manifest_builder.add_path(
            self.NWB_DIR_KEY, 'nwb_files', parent_key='BASEDIR', typename='dir'
        )

        manifest_builder.add_path(
            self.ANALYSIS_FILES_DIR_KEY, 'analysis_files', parent_key='BASEDIR', typename='dir'
        )

        manifest_builder.add_path(
            self.SESSION_NWB_KEY, 'behavior_ophys_session_%d.nwb', parent_key=self.NWB_DIR_KEY, typename='file'
        )

        manifest_builder.add_path(
            self.TRIAL_RESPONSE_KEY, 'trial_response_df_%d.h5', parent_key=self.ANALYSIS_FILES_DIR_KEY, typename='file'
        )
        
        manifest_builder.add_path(
            self.FLASH_RESPONSE_KEY, 'flash_response_df_%d.h5', parent_key=self.ANALYSIS_FILES_DIR_KEY, typename='file'
        )

        manifest_builder.add_path(
            self.EXTENDED_STIM_COLUMNS_KEY, 'extended_stimulus_presentations_df_%d.h5', parent_key=self.ANALYSIS_FILES_DIR_KEY, typename='file'
        )

        manifest_builder.add_path(
            self.EXPOSURE_NUMBER_KEY, 'exposure_number_%d.h5', parent_key=self.ANALYSIS_FILES_DIR_KEY, typename='file'
        )


        return manifest_builder

    @classmethod
    def from_lims(cls, lims_kwargs=None, **kwargs):
        lims_kwargs = {} if lims_kwargs is None else lims_kwargs
        return cls(
            fetch_api=BehaviorProjectLimsApi.default(**lims_kwargs), 
            **kwargs
        )

    #  @classmethod
    #  def from_warehouse(cls, warehouse_kwargs=None, **kwargs):
    #      warehouse_kwargs = {} if warehouse_kwargs is None else warehouse_kwargs
    #      return cls(
    #          fetch_api=EcephysProjectWarehouseApi.default(**warehouse_kwargs), 
    #          **kwargs
    #      )

    @classmethod
    def fixed(cls, **kwargs):
        return cls(fetch_api=BehaviorProjectFixedApi(), **kwargs)

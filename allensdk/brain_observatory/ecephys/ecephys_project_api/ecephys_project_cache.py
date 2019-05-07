from allensdk.api.cache import Cache

from .ecephys_project_lims_api import EcephysProjectLimsApi
from .ecephys_project_api import EcephysProjectApi


class EcephysProjectCache(Cache):

    SESSIONS_KEY = 'sessions'
    MANIFEST_VERSION = '0.1.0'

    def __init__(self, fetch_api, **kwargs):
        
        kwargs['manifest'] = kwargs.get('manifest', 'ecephys_project_manifest.json')
        kwargs['version'] = kwargs.get('version', self.MANIFEST_VERSION)

        super(EcephysProjectCache, self).__init__(**kwargs)
        self.fetch_api = fetch_api

    def get_sessions(self):
        path = self.get_cache_path(None, self.SESSIONS_KEY, )
        sessions = self.fetch_api.get_sessions(path)
        return filter_sessions(sessions)

    def add_manifest_paths(self, manifest_builder):
        manifest_builder = super(EcephysProjectCache, self).add_manifest_paths(manifest_builder)
                                  
        manifest_builder.add_path(
            self.SESSIONS_KEY, 'sessions.csv', parent_key='BASEDIR', typename='file'
        )

        return manifest_builder

    @classmethod
    def from_lims(cls, lims_kwargs=None, **kwargs):
        lims_kwargs = {} if lims_kwargs is None else lims_kwargs
        return cls(fetch_api=EcephysProjectLimsApi(**lims_kwargs), **kwargs)

def filter_sessions(sessions):
    return sessions
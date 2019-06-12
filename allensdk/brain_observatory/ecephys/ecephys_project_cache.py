import functools
from pathlib import Path

import pandas as pd

from allensdk.api.cache import Cache

from allensdk.brain_observatory.ecephys.ecephys_project_api import EcephysProjectLimsApi, EcephysProjectWarehouseApi
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession


csv_io = {
    'reader': lambda path: pd.read_csv(path, index_col='id'),
    'writer': lambda path, df: df.to_csv(path)
}


def call_caching(fn, path, strategy=None, pre=lambda d: d, writer=None, reader=None, post=None, *args, **kwargs):
    fn = functools.partial(fn, *args, **kwargs)
    try:
        return Cache.cacher(fn, path=path, strategy=strategy, pre=pre, writer=writer, reader=reader, post=post)
    except:
        Path(path).unlink
        raise


class EcephysProjectCache(Cache):

    SESSIONS_KEY = 'sessions'
    PROBES_KEY = 'probes'
    CHANNELS_KEY = 'channels'
    UNITS_KEY = 'units'
    SESSION_DIR_KEY = 'session_data'
    SESSION_NWB_KEY = 'session_nwb'

    MANIFEST_VERSION = '0.1.0'

    def __init__(self, fetch_api, **kwargs):
        
        kwargs['manifest'] = kwargs.get('manifest', 'ecephys_project_manifest.json')
        kwargs['version'] = kwargs.get('version', self.MANIFEST_VERSION)

        super(EcephysProjectCache, self).__init__(**kwargs)
        self.fetch_api = fetch_api

    def get_sessions(self):
        path = self.get_cache_path(None, self.SESSIONS_KEY)
        return call_caching(self.fetch_api.get_sessions, path=path, strategy='lazy', **csv_io)

    def get_probes(self):
        path = self.get_cache_path(None, self.PROBES_KEY)
        return call_caching(self.fetch_api.get_probes, path, strategy='lazy', **csv_io)

    def get_channels(self):
        path = self.get_cache_path(None, self.CHANNELS_KEY)
        return call_caching(self.fetch_api.get_channels, path, strategy='lazy', **csv_io)

    def get_units(self):
        path = self.get_cache_path(None, self.UNITS_KEY)
        return call_caching(self.fetch_api.get_units, path, strategy='lazy', **csv_io)

    def get_session_data(self, session_id):
        path = self.get_cache_path(None, self.SESSION_NWB_KEY, session_id, session_id)

        def writer(_path, data):
            with open(_path, 'wb') as writer:
                for chunk in data:
                    writer.write(chunk)

        return call_caching(
            self.fetch_api.get_session_data, 
            path, 
            session_id=session_id, 
            strategy='lazy',
            writer=writer,
            reader=EcephysSession.from_nwb_path
        )

    def add_manifest_paths(self, manifest_builder):
        manifest_builder = super(EcephysProjectCache, self).add_manifest_paths(manifest_builder)
                                  
        manifest_builder.add_path(
            self.SESSIONS_KEY, 'sessions.csv', parent_key='BASEDIR', typename='file'
        )

        manifest_builder.add_path(
            self.PROBES_KEY, 'probes.csv', parent_key='BASEDIR', typename='file'
        )

        manifest_builder.add_path(
            self.CHANNELS_KEY, 'channels.csv', parent_key='BASEDIR', typename='file'
        )

        manifest_builder.add_path(
            self.UNITS_KEY, 'units.csv', parent_key='BASEDIR', typename='file'
        )

        manifest_builder.add_path(
            self.SESSION_DIR_KEY, 'session_%d', parent_key='BASEDIR', typename='dir'
        )

        manifest_builder.add_path(
            self.SESSION_NWB_KEY, 'session_%d.nwb', parent_key=self.SESSION_DIR_KEY, typename='file'
        )

        return manifest_builder

    @classmethod
    def from_lims(cls, lims_kwargs=None, **kwargs):
        lims_kwargs = {} if lims_kwargs is None else lims_kwargs
        return cls(
            fetch_api=EcephysProjectLimsApi.default(**lims_kwargs), 
            **kwargs
        )

    @classmethod
    def from_warehouse(cls, warehouse_kwargs=None, **kwargs):
        warehouse_kwargs = {} if warehouse_kwargs is None else warehouse_kwargs
        return cls(
            fetch_api=EcephysProjectWarehouseApi.default(**warehouse_kwargs), 
            **kwargs
        )

import functools
from pathlib import Path
import ast

import pandas as pd
import SimpleITK as sitk
import h5py

from allensdk.api.cache import Cache

from allensdk.brain_observatory.ecephys.ecephys_project_api import EcephysProjectLimsApi, EcephysProjectWarehouseApi, EcephysProjectFixedApi
from allensdk.brain_observatory.ecephys.ecephys_session_api import EcephysNwbSessionApi
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
from allensdk.brain_observatory.ecephys.file_promise import FilePromise, read_nwb, write_from_stream
from allensdk.brain_observatory.ecephys import get_unit_filter_value


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
    PROBE_LFP_NWB_KEY = "probe_lfp_nwb"

    NATURAL_MOVIE_DIR_KEY = "movie_dir"
    NATURAL_MOVIE_KEY = "natural_movie"

    NATURAL_SCENE_DIR_KEY = "natural_scene_dir"
    NATURAL_SCENE_KEY = "natural_scene"

    MANIFEST_VERSION = '0.2.0'

    def __init__(self, fetch_api, **kwargs):
        
        kwargs['manifest'] = kwargs.get('manifest', 'ecephys_project_manifest.json')
        kwargs['version'] = kwargs.get('version', self.MANIFEST_VERSION)

        super(EcephysProjectCache, self).__init__(**kwargs)
        self.fetch_api = fetch_api


    def get_sessions(self):
        path = self.get_cache_path(None, self.SESSIONS_KEY)
        def reader(path):
            response = pd.read_csv(path, index_col='id')
            if "structure_acronyms" in response.columns: #  unfortunately, structure_acronyms is a list of str
                response["structure_acronyms"] = [ast.literal_eval(item) for item in response["structure_acronyms"]]
            return response
        return call_caching(self.fetch_api.get_sessions, path=path, strategy='lazy', writer=csv_io["writer"], reader=reader)

    def get_probes(self):
        path = self.get_cache_path(None, self.PROBES_KEY)
        return call_caching(self.fetch_api.get_probes, path, strategy='lazy', **csv_io)

    def get_channels(self):
        path = self.get_cache_path(None, self.CHANNELS_KEY)
        return call_caching(self.fetch_api.get_channels, path, strategy='lazy', **csv_io)

    def get_units(self, annotate=False, **kwargs):
        """ Reports a table consisting of all sorted units across the entire extracellular electrophysiology project.

        Parameters
        ----------
        annotate : bool, optional
            If True, the returned table of units will be merged with channel, probe, and session information.

        Returns
        -------
        pd.DataFrame : 
            each row describes a single sorted unit

        """

        path = self.get_cache_path(None, self.UNITS_KEY)
        get_units = functools.partial(
            self.fetch_api.get_units, 
            amplitude_cutoff_maximum=None, # pull down all the units to csv and filter on the way out
            presence_ratio_minimum=None, 
            isi_violations_maximum=None
        )
        units = call_caching(get_units, path, strategy='lazy', **csv_io)

        if annotate:
            channels = self.get_channels().drop(columns=["unit_count"])
            probes = self.get_probes().drop(columns=["unit_count", "channel_count"])
            sessions = self.get_sessions().drop(columns=["probe_count", "unit_count", "channel_count", "structure_acronyms"])

            units = pd.merge(units, channels, left_on='peak_channel_id', right_index=True, suffixes=['_unit', '_channel'])
            units = pd.merge(units, probes, left_on='ecephys_probe_id', right_index=True, suffixes=['_unit', '_probe'])
            units = pd.merge(units, sessions, left_on='ecephys_session_id', right_index=True, suffixes=['_unit', '_session'])

        units =units[
            (units["amplitude_cutoff"] <= get_unit_filter_value("amplitude_cutoff_maximum", **kwargs))
            & (units["presence_ratio"] >= get_unit_filter_value("presence_ratio_minimum", **kwargs))
            & (units["isi_violations"] <= get_unit_filter_value("isi_violations_maximum", **kwargs))
        ]
        
        return units


    def get_session_data(self, session_id):
        path = self.get_cache_path(None, self.SESSION_NWB_KEY, session_id, session_id)

        probes = self.get_probes()
        probe_ids = probes[probes["ecephys_session_id"] == session_id].index.values
        
        probe_promises = {
            probe_id: FilePromise(
                source=functools.partial(self.fetch_api.get_probe_lfp_data, probe_id),
                path=Path(self.get_cache_path(None, self.PROBE_LFP_NWB_KEY, session_id, probe_id)),
                reader=read_nwb
            )
            for probe_id in probe_ids
        }

        call_caching(
            self.fetch_api.get_session_data, 
            path, 
            session_id=session_id, 
            strategy='lazy',
            writer=write_from_stream,
        )

        session_api = EcephysNwbSessionApi(path=path, probe_lfp_paths=probe_promises)
        return EcephysSession(api=session_api)

    def get_natural_movie_template(self, number):
        path = self.get_cache_path(None, self.NATURAL_MOVIE_KEY, number)

        def reader(path):
            with h5py.File(path, "r") as fil:
                return fil["data"][:]

        return call_caching(
            self.fetch_api.get_natural_movie_template,
            path,
            number=number,
            strategy="lazy",
            writer=write_from_stream,
            reader=reader
        )

    def get_natural_scene_template(self, number):
        path = self.get_cache_path(None, self.NATURAL_SCENE_KEY, number)

        def reader(path):
            return sitk.GetArrayFromImage(sitk.ReadImage(path))

        return call_caching(
            self.fetch_api.get_natural_scene_template, 
            path, 
            number=number, 
            strategy="lazy",
            writer=write_from_stream,
            reader=reader
        )

    def get_all_stimulus_sets(self, **session_kwargs):
        return self._get_all_values("session_type", self.get_sessions, **session_kwargs)

    def get_all_genotypes(self, **session_kwargs):
        return self._get_all_values("genotype", self.get_sessions, **session_kwargs)

    def get_all_recorded_structures(self, **channel_kwargs):
        return self._get_all_values("manual_structure_acronym", self.get_channels, **channel_kwargs)

    def get_all_project_codes(self):
        return self._get_all_values("project_code", self.get_sessions, **session_kwargs)

    def get_all_ages(self):
        return self._get_all_values("age", self.get_sessions, **session_kwargs)
    
    def get_all_genders(self):
        return self._get_all_values("gender", self.get_sessions, **session_kwargs)

    def _get_all_values(self, key, method=None, **method_kwargs):
        if method is None:
            method = self.get_sessions
        data = method(**method_kwargs)
        return data[key].unique().tolist()

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

        manifest_builder.add_path(
            self.PROBE_LFP_NWB_KEY, 'probe_%d_lfp.nwb', parent_key=self.SESSION_DIR_KEY, typename='file'
        )

        manifest_builder.add_path(
            self.NATURAL_MOVIE_DIR_KEY, "natural_movie_templates", parent_key="BASEDIR", typename="dir"
        )

        manifest_builder.add_path(
            self.NATURAL_MOVIE_KEY, "natural_movie_%d.h5", parent_key=self.NATURAL_MOVIE_DIR_KEY, typename="file"
        )

        manifest_builder.add_path(
            self.NATURAL_SCENE_DIR_KEY, "natural_scene_templates", parent_key="BASEDIR", typename="dir"
        )

        manifest_builder.add_path(
            self.NATURAL_SCENE_KEY, "natural_scene_%d.tiff", parent_key=self.NATURAL_SCENE_DIR_KEY, typename="file"
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

    @classmethod
    def fixed(cls, **kwargs):
        return cls(fetch_api=EcephysProjectFixedApi(), **kwargs)
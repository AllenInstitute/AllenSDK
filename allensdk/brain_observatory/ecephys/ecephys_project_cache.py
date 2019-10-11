import functools
from pathlib import Path
from typing import Any, List, Optional
import ast

import pandas as pd
import SimpleITK as sitk
import numpy as np

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
    except Exception:
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

    SESSION_ANALYSIS_METRICS_KEY = "session_analysis_metrics"
    TYPEWISE_ANALYSIS_METRICS_KEY = "typewise_analysis_metrics"

    MANIFEST_VERSION = '0.2.1'

    SUPPRESS_FROM_UNITS = ("air_channel_index",
                           "surface_channel_index",
                           "has_nwb",
                           "lfp_temporal_subsampling_factor",
                           "epoch_name_quality_metrics",
                           "epoch_name_waveform_metrics",
                           "isi_experiment_id")
    SUPPRESS_FROM_CHANNELS = (
        "air_channel_index", "surface_channel_index", "name",
        "date_of_acquisition", "published_at", "specimen_id", "session_type", "isi_experiment_id", "age_in_days",
        "sex", "genotype", "has_nwb", "lfp_temporal_subsampling_factor"
    )
    SUPPRESS_FROM_PROBES = (
        "air_channel_index", "surface_channel_index",
        "date_of_acquisition", "published_at", "specimen_id", "session_type", "isi_experiment_id", "age_in_days",
        "sex", "genotype", "has_nwb", "lfp_temporal_subsampling_factor"
    )
    SUPPRESS_FROM_SESSION_TABLE = (
        "has_nwb",
        "isi_experiment_id",
        "date_of_acquisition"
    )

    def __init__(self, fetch_api, **kwargs):

        kwargs['manifest'] = kwargs.get('manifest', 'ecephys_project_manifest.json')
        kwargs['version'] = kwargs.get('version', self.MANIFEST_VERSION)

        super(EcephysProjectCache, self).__init__(**kwargs)
        self.fetch_api = fetch_api

    def _get_sessions(self):
        path = self.get_cache_path(None, self.SESSIONS_KEY)

        def reader(path):
            response = pd.read_csv(path, index_col='id')
            if "structure_acronyms" in response.columns:  # unfortunately, structure_acronyms is a list of str
                response["ecephys_structure_acronyms"] = [ast.literal_eval(item) for item in response["structure_acronyms"]]
                response.drop(columns=["structure_acronyms"], inplace=True)
            return response

        return call_caching(self.fetch_api.get_sessions, path=path, strategy='lazy', writer=csv_io["writer"], reader=reader)

    def _get_probes(self):
        path = self.get_cache_path(None, self.PROBES_KEY)
        probes = call_caching(self.fetch_api.get_probes, path, strategy='lazy', **csv_io)
        # Divide the lfp sampling by the subsampling factor for clearer presentation (if provided)
        if all(c in list(probes) for c in
               ["lfp_sampling_rate", "lfp_temporal_subsampling_factor"]):
            probes["lfp_sampling_rate"] = (
                probes["lfp_sampling_rate"] / probes["lfp_temporal_subsampling_factor"])
        return probes

    def _get_channels(self):
        path = self.get_cache_path(None, self.CHANNELS_KEY)
        return call_caching(self.fetch_api.get_channels, path, strategy='lazy', **csv_io)

    def _get_units(self, filter_by_validity: bool = True, **unit_filter_kwargs) -> pd.DataFrame:
        path = self.get_cache_path(None, self.UNITS_KEY)
        get_units = functools.partial(
            self.fetch_api.get_units,
            amplitude_cutoff_maximum=None,  # pull down all the units to csv and filter on the way out
            presence_ratio_minimum=None,
            isi_violations_maximum=None,
            filter_by_validity=filter_by_validity
        )
        units = call_caching(get_units, path, strategy='lazy', **csv_io)
        units = units.rename(columns={
            'PT_ratio': 'waveform_PT_ratio',
            'amplitude': 'waveform_amplitude',
            'duration': 'waveform_duration',
            'halfwidth': 'waveform_halfwidth',
            'recovery_slope': 'waveform_recovery_slope',
            'repolarization_slope': 'waveform_repolarization_slope',
            'spread': 'waveform_spread',
            'velocity_above': 'waveform_velocity_above',
            'velocity_below': 'waveform_velocity_below',
            'l_ratio': 'L_ratio',
        })

        units = units[
            (units["amplitude_cutoff"] <= get_unit_filter_value("amplitude_cutoff_maximum", **unit_filter_kwargs))
            & (units["presence_ratio"] >= get_unit_filter_value("presence_ratio_minimum", **unit_filter_kwargs))
            & (units["isi_violations"] <= get_unit_filter_value("isi_violations_maximum", **unit_filter_kwargs))
        ]

        if "quality" in units.columns and filter_by_validity:
            units = units[units["quality"] == "good"]
            units.drop(columns="quality", inplace=True)

        if "ecephys_structure_id" in units.columns and unit_filter_kwargs.get("filter_out_of_brain_units", True):
            units = units[~(units["ecephys_structure_id"].isna())]

        return units

    def _get_annotated_probes(self):
        sessions = self._get_sessions()
        probes = self._get_probes()

        return pd.merge(probes, sessions, left_on="ecephys_session_id", right_index=True, suffixes=['_probe', '_session'])

    def _get_annotated_channels(self):
        channels = self._get_channels()
        probes = self._get_annotated_probes()

        return pd.merge(channels, probes, left_on="ecephys_probe_id", right_index=True, suffixes=['_channel', '_probe'])

    def _get_annotated_units(self, filter_by_validity: bool = True, **unit_filter_kwargs) -> pd.DataFrame:
        units = self._get_units(filter_by_validity=filter_by_validity, **unit_filter_kwargs)
        channels = self._get_annotated_channels()
        annotated_units = pd.merge(units, channels, left_on='ecephys_channel_id', right_index=True, suffixes=['_unit', '_channel'])
        annotated_units = annotated_units.rename(columns={
            'name': 'probe_name',
            'phase': 'probe_phase',
            'sampling_rate': 'probe_sampling_rate',
            'lfp_sampling_rate': 'probe_lfp_sampling_rate',
            'local_index': 'peak_channel'
        })

        return pd.merge(units, channels, left_on='ecephys_channel_id', right_index=True, suffixes=['_unit', '_channel'])

    def get_session_table(self, suppress=None) -> pd.DataFrame:
        sessions = self._get_sessions()

        count_owned(sessions, self._get_annotated_units(), "ecephys_session_id", "unit_count", inplace=True)
        count_owned(sessions, self._get_annotated_channels(), "ecephys_session_id", "channel_count", inplace=True)
        count_owned(sessions, self._get_annotated_probes(), "ecephys_session_id", "probe_count", inplace=True)

        get_grouped_uniques(sessions, self._get_annotated_channels(), "ecephys_session_id", "ecephys_structure_acronym", "ecephys_structure_acronyms", inplace=True)

        if suppress is None:
            suppress = list(self.SUPPRESS_FROM_SESSION_TABLE)
        sessions.drop(columns=suppress, inplace=True, errors="ignore")
        sessions = sessions.rename(columns={'genotype': 'full_genotype'})
        return sessions

    def get_probes(self, suppress=None):
        probes = self._get_annotated_probes()

        count_owned(probes, self._get_annotated_units(), "ecephys_probe_id", "unit_count", inplace=True)
        count_owned(probes, self._get_annotated_channels(), "ecephys_probe_id", "channel_count", inplace=True)

        get_grouped_uniques(probes, self._get_annotated_channels(), "ecephys_probe_id", "ecephys_structure_acronym", "ecephys_structure_acronyms", inplace=True)

        if suppress is None:
            suppress = list(self.SUPPRESS_FROM_PROBES)
        probes.drop(columns=suppress, inplace=True, errors="ignore")

        return probes

    def get_channels(self, suppress=None):
        """ Load (potentially downloading and caching) a table whose rows are individual channels.
        """

        channels = self._get_annotated_channels()
        count_owned(channels, self._get_annotated_units(), "ecephys_channel_id", "unit_count", inplace=True)

        if suppress is None:
            suppress = list(self.SUPPRESS_FROM_CHANNELS)
        channels.drop(columns=suppress, inplace=True, errors="ignore")
        channels.rename(columns={"name": "probe_name"}, inplace=True, errors="ignore")

        return channels

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

    def get_units(self, suppress: Optional[List[str]] = None, filter_by_validity: bool = True, **unit_filter_kwargs) -> pd.DataFrame:
        """Reports a table consisting of all sorted units across the entire extracellular electrophysiology project.

        Parameters
        ----------
        suppress : Optional[List[str]], optional
            A list of dataframe column names to hide, by default None
            (None will hide dataframe columns specified in: SUPPRESS_FROM_UNITS)

        filter_by_validity : bool, optional
            Filter units so that only 'valid' units are returned, by default True

        **unit_filter_kwargs :
            Additional keyword arguments that can be used to filter units (for power users).

        Returns
        -------
        pd.DataFrame
            A table consisting of sorted units across the entire extracellular electrophysiology project
        """
        if suppress is None:
            suppress = list(self.SUPPRESS_FROM_UNITS)

        units = self._get_annotated_units(filter_by_validity=filter_by_validity, **unit_filter_kwargs)
        units.drop(columns=suppress, inplace=True, errors="ignore")

        return units

    def get_session_data(self, session_id, filter_by_validity: bool = True, **unit_filter_kwargs):
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
            filter_by_validity=filter_by_validity,
            **unit_filter_kwargs
        )

        get_analysis_metrics = functools.partial(self.get_unit_analysis_metrics_for_session,
                                                 session_id=session_id,
                                                 annotate=False,
                                                 filter_by_validity=True,
                                                 **unit_filter_kwargs)

        def get_channel_columns():
            channels = self.get_channels()
            return channels.loc[channels["ecephys_session_id"] == session_id, [
                "ecephys_structure_id", 
                "ecephys_structure_acronym", 
                "anterior_posterior_ccf_coordinate",
                "dorsal_ventral_ccf_coordinate", 
                "left_right_ccf_coordinate"
            ]]

        session_api = EcephysNwbSessionApi(
            path=path,
            probe_lfp_paths=probe_promises,
            additional_unit_metrics=get_analysis_metrics,
            external_channel_columns=get_channel_columns,
            filter_by_validity=filter_by_validity,
            **unit_filter_kwargs
        )

        return EcephysSession(api=session_api)

    def get_natural_movie_template(self, number):
        path = self.get_cache_path(None, self.NATURAL_MOVIE_KEY, number)

        def reader(path):
            return np.load(path, allow_pickle=False)

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

    def get_all_session_types(self, **session_kwargs):
        return self._get_all_values("session_type", self.get_session_table, **session_kwargs)

    def get_all_full_genotypes(self, **session_kwargs):
        return self._get_all_values("full_genotype", self.get_session_table, **session_kwargs)

    def get_structure_acronyms(self, **channel_kwargs) -> List[str]:
        return self._get_all_values("ecephys_structure_acronym", self.get_channels, **channel_kwargs)

    def get_all_ages(self, **session_kwargs):
        return self._get_all_values("age_in_days", self.get_session_table, **session_kwargs)

    def get_all_sexes(self, **session_kwargs):
        return self._get_all_values("sex", self.get_session_table, **session_kwargs)

    def _get_all_values(self, key, method=None, **method_kwargs) -> List[Any]:
        if method is None:
            method = self.get_session_table
        data = method(**method_kwargs)
        return data[key].unique().tolist()

    def get_unit_analysis_metrics_for_session(self, session_id, annotate: bool = True, filter_by_validity: bool = True, **unit_filter_kwargs):
        """ Cache and return a table of analysis metrics calculated on each unit from a specified session. See
        get_session_table for a list of sessions.

        Parameters
        ----------
        session_id : int
            identifies the session from which to fetch analysis metrics.
        annotate : bool, optional
            if True, information from the annotated units table will be merged onto the outputs
        filter_by_validity : bool, optional
            Filter units used by analysis so that only 'valid' units are returned, by default True
        **unit_filter_kwargs :
            Additional keyword arguments that can be used to filter units (for power users).

        Returns
        -------
        metrics : pd.DataFrame
            Each row corresponds to a single unit, describing a set of analysis metrics calculated on that unit.

        """

        path = self.get_cache_path(None, self.SESSION_ANALYSIS_METRICS_KEY, session_id, session_id)
        metrics = call_caching(
            self.fetch_api.get_unit_analysis_metrics,
            path,
            strategy='lazy',
            ecephys_session_ids=[session_id],
            reader=lambda path: pd.read_csv(path, index_col='ecephys_unit_id'),
            writer=lambda path, df: df.to_csv(path)
        )

        if annotate:
            units = self.get_units(filter_by_validity=filter_by_validity, **unit_filter_kwargs)
            units = units[units["ecephys_session_id"] == session_id]
            metrics = pd.merge(units, metrics, left_index=True, right_index=True, how="inner")
            metrics.index.rename("ecephys_unit_id", inplace=True)

        return metrics

    def get_unit_analysis_metrics_by_session_type(self, session_type, annotate: bool = True, filter_by_validity: bool = True, **unit_filter_kwargs):
        """ Cache and return a table of analysis metrics calculated on each unit from a specified session type. See
        get_all_session_types for a list of session types.

        Parameters
        ----------
        session_type : str
            identifies the session type for which to fetch analysis metrics.
        annotate : bool, optional
            if True, information from the annotated units table will be merged onto the outputs
        filter_by_validity : bool, optional
            Filter units used by analysis so that only 'valid' units are returned, by default True
        **unit_filter_kwargs :
            Additional keyword arguments that can be used to filter units (for power users).

        Returns
        -------
        metrics : pd.DataFrame
            Each row corresponds to a single unit, describing a set of analysis metrics calculated on that unit.

        """

        known_session_types = self.get_all_session_types()
        if session_type not in known_session_types:
            raise ValueError(f"unrecognized session type: {session_type}. Available types: {known_session_types}")

        path = self.get_cache_path(None, self.TYPEWISE_ANALYSIS_METRICS_KEY, session_type)
        metrics = call_caching(
            self.fetch_api.get_unit_analysis_metrics,
            path,
            strategy='lazy',
            session_types=[session_type],
            reader=lambda path: pd.read_csv(path, index_col='ecephys_unit_id'),
            writer=lambda path, df: df.to_csv(path)
        )

        if annotate:
            units = self.get_units(filter_by_validity=filter_by_validity, **unit_filter_kwargs)
            metrics = pd.merge(units, metrics, left_index=True, right_index=True, how="inner")
            metrics.index.rename("ecephys_unit_id", inplace=True)

        return metrics

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
            self.SESSION_ANALYSIS_METRICS_KEY, 'session_%d_analysis_metrics.csv', parent_key=self.SESSION_DIR_KEY, typename='file'
        )

        manifest_builder.add_path(
            self.PROBE_LFP_NWB_KEY, 'probe_%d_lfp.nwb', parent_key=self.SESSION_DIR_KEY, typename='file'
        )

        manifest_builder.add_path(
            self.NATURAL_MOVIE_DIR_KEY, "natural_movie_templates", parent_key="BASEDIR", typename="dir"
        )

        manifest_builder.add_path(
            self.TYPEWISE_ANALYSIS_METRICS_KEY, "%s_analysis_metrics.csv", parent_key='BASEDIR', typename="file"
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


def count_owned(this, other, foreign_key, count_key, inplace=False):
    if not inplace:
        this = this.copy()

    counts = other.loc[:, foreign_key].value_counts()
    this[count_key] = 0
    this.loc[counts.index.values, count_key] = counts.values

    if not inplace:
        return this


def get_grouped_uniques(this, other, foreign_key, field_key, unique_key, inplace=False):
    if not inplace:
        this = this.copy()

    uniques = other.groupby(foreign_key)\
        .apply(lambda grp: pd.DataFrame(grp)[field_key].unique())
    this[unique_key] = 0
    this.loc[uniques.index.values, unique_key] = uniques.values

    if not inplace:
        return this

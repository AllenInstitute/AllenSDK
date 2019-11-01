from typing import Optional, Generator

import pandas as pd

from .ecephys_project_api import EcephysProjectApi, ArrayLike
from .http_engine import HttpEngine
from .utilities import postgres_macros, build_and_execute

from allensdk.internal.api import PostgresQueryMixin


class EcephysProjectLimsApi(EcephysProjectApi):

    STIMULUS_NAMESPACE = "brain_observatory_1.1"

    def __init__(self, postgres_engine, app_engine):
        self.postgres_engine = postgres_engine
        self.app_engine = app_engine

    def get_session_data(self, session_id, **kwargs):
        nwb_response = build_and_execute(
            """
            select wkf.id, wkf.filename, wkf.storage_directory, wkf.attachable_id from well_known_files wkf 
            join ecephys_analysis_runs ear on (
                ear.id = wkf.attachable_id
                and wkf.attachable_type = 'EcephysAnalysisRun'
            )
            join well_known_file_types wkft on wkft.id = wkf.well_known_file_type_id
            where ear.current
            and wkft.name = 'EcephysNwb'
            and ear.ecephys_session_id = {{session_id}}
        """,
            engine=self.postgres_engine.select,
            session_id=session_id,
        )

        if nwb_response.shape[0] != 1:
            raise ValueError(
                f"expected exactly 1 current NWB file for session {session_id}, "
                f"found {nwb_response.shape[0]}: {pd.DataFrame(nwb_response)}"
            )

        nwb_id = nwb_response.loc[0, "id"]
        return self.app_engine.stream(
            f"well_known_files/download/{nwb_id}?wkf_id={nwb_id}"
        )

    def get_probe_lfp_data(self, probe_id):
        nwb_response = build_and_execute(
            """
            select wkf.id from well_known_files wkf
            join ecephys_analysis_run_probes earp on (
                earp.id = wkf.attachable_id
                and wkf.attachable_type = 'EcephysAnalysisRunProbe'
            )
            join ecephys_analysis_runs ear on ear.id = earp.ecephys_analysis_run_id
            join well_known_file_types wkft on wkft.id = wkf.well_known_file_type_id
            where wkft.name ~ 'EcephysLfpNwb'
            and ear.current
            and earp.ecephys_probe_id = {{probe_id}}
            """,
            engine=self.postgres_engine.select,
            probe_id=probe_id
        )

        if nwb_response.shape[0] != 1:
            raise ValueError(
                f"expected exactly 1 current LFP NWB file for probe {probe_id}, "
                f"found {nwb_response.shape[0]}: {pd.DataFrame(nwb_response)}"
            )

        nwb_id = nwb_response.loc[0, "id"]
        return self.app_engine.stream(
            f"well_known_files/download/{nwb_id}?wkf_id={nwb_id}"
        )

    def get_units(
        self, 
        unit_ids: Optional[ArrayLike] = None, 
        channel_ids: Optional[ArrayLike] = None, 
        probe_ids: Optional[ArrayLike] = None, 
        session_ids: Optional[ArrayLike] = None, 
        published_at: Optional[str] = None
    ):
        """ Query LIMS for records describing sorted ecephys units.

        Parameters
        ----------
        units_ids :
        channel_ids :
        probe_ids :
        session_ids :
        published_at :
            If provided, only units from sessions published prior to this date 
            will be returned. Format should be YYYY-MM-DD

        """

        published_at_not_null = None if published_at is None else True
        published_at = f"'{published_at}'" if published_at is not None else None

        response = build_and_execute(
            """
                {%- import 'postgres_macros' as pm -%}
                {%- import 'macros' as m -%}
                select 
                    eu.id, 
                    eu.ecephys_channel_id,
                    eu.quality,
                    eu.snr,
                    eu.firing_rate,
                    eu.isi_violations,
                    eu.presence_ratio,
                    eu.amplitude_cutoff,
                    eu.isolation_distance,
                    eu.l_ratio,
                    eu.d_prime,
                    eu.nn_hit_rate,
                    eu.nn_miss_rate,
                    eu.silhouette_score,
                    eu.max_drift,
                    eu.cumulative_drift,
                    eu.epoch_name_quality_metrics,
                    eu.epoch_name_waveform_metrics,
                    eu.duration,
                    eu.halfwidth,
                    eu.\"PT_ratio\",
                    eu.repolarization_slope,
                    eu.recovery_slope,
                    eu.amplitude,
                    eu.spread,
                    eu.velocity_above,
                    eu.velocity_below
                from ecephys_units eu
                join ecephys_channels ec on ec.id = eu.ecephys_channel_id
                join ecephys_probes ep on ep.id = ec.ecephys_probe_id
                join ecephys_sessions es on es.id = ep.ecephys_session_id
                where 
                    not es.habituation 
                    and ec.valid_data
                    and ep.workflow_state != 'failed'
                    and es.workflow_state != 'failed'
                    {{pm.optional_not_null('es.published_at', published_at_not_null)}}
                    {{pm.optional_le('es.published_at', published_at)}}
                    {{pm.optional_contains('eu.id', unit_ids) -}}
                    {{pm.optional_contains('ec.id', channel_ids) -}}
                    {{pm.optional_contains('ep.id', probe_ids) -}}
                    {{pm.optional_contains('es.id', session_ids) -}}
            """,
            base=postgres_macros(),
            engine=self.postgres_engine.select,
            unit_ids=unit_ids,
            channel_ids=channel_ids,
            probe_ids=probe_ids,
            session_ids=session_ids,
            published_at_not_null=published_at_not_null,
            published_at=published_at
        )
        return response.set_index("id", inplace=False)

    def get_channels(
        self, 
        channel_ids: Optional[ArrayLike] = None, 
        probe_ids: Optional[ArrayLike] = None, 
        session_ids: Optional[ArrayLike] = None, 
        published_at: Optional[str] = None
    ):

        published_at_not_null = None if published_at is None else True
        published_at = f"'{published_at}'" if published_at is not None else None

        response = build_and_execute(
            """
                {%- import 'postgres_macros' as pm -%}
                select 
                    ec.id, 
                    ec.ecephys_probe_id, 
                    ec.local_index,
                    ec.probe_vertical_position,
                    ec.probe_horizontal_position,
                    ec.manual_structure_id as ecephys_structure_id,
                    st.acronym as ecephys_structure_acronym,
                    ec.anterior_posterior_ccf_coordinate,
                    ec.dorsal_ventral_ccf_coordinate,
                    ec.left_right_ccf_coordinate
                from ecephys_channels ec
                join ecephys_probes ep on ep.id = ec.ecephys_probe_id
                join ecephys_sessions es on es.id = ep.ecephys_session_id
                left join structures st on ec.manual_structure_id = st.id
                where 
                    not es.habituation 
                    and valid_data
                    and ep.workflow_state != 'failed'
                    and es.workflow_state != 'failed'
                    {{pm.optional_not_null('es.published_at', published_at_not_null)}}
                    {{pm.optional_le('es.published_at', published_at)}}
                    {{pm.optional_contains('ec.id', channel_ids) -}}
                    {{pm.optional_contains('ep.id', probe_ids) -}}
                    {{pm.optional_contains('es.id', session_ids) -}}
            """,
            base=postgres_macros(),
            engine=self.postgres_engine.select,
            channel_ids=channel_ids,
            probe_ids=probe_ids,
            session_ids=session_ids,
            published_at_not_null=published_at_not_null,
            published_at=published_at
        )
        return response.set_index("id")

    def get_probes(
        self, 
        probe_ids: Optional[ArrayLike] = None, 
        session_ids: Optional[ArrayLike] = None, 
        published_at: Optional[str] = None
    ):

        published_at_not_null = None if published_at is None else True
        published_at = f"'{published_at}'" if published_at is not None else None

        response = build_and_execute(
            """
                {%- import 'postgres_macros' as pm -%}
                select 
                    ep.id, 
                    ep.ecephys_session_id, 
                    ep.name, 
                    ep.global_probe_sampling_rate as sampling_rate, 
                    ep.global_probe_lfp_sampling_rate as lfp_sampling_rate,
                    ep.phase,
                    ep.air_channel_index,
                    ep.surface_channel_index,
                    ep.use_lfp_data as has_lfp_data,
                    ep.temporal_subsampling_factor as lfp_temporal_subsampling_factor
                from ecephys_probes ep
                join ecephys_sessions es on es.id = ep.ecephys_session_id
                where 
                    not es.habituation 
                    and ep.workflow_state != 'failed'
                    and es.workflow_state != 'failed'
                    {{pm.optional_not_null('es.published_at', published_at_not_null)}}
                    {{pm.optional_le('es.published_at', published_at)}}
                    {{pm.optional_contains('ep.id', probe_ids) -}}
                    {{pm.optional_contains('es.id', session_ids) -}}
            """,
            base=postgres_macros(),
            engine=self.postgres_engine.select,
            probe_ids=probe_ids,
            session_ids=session_ids,
            published_at_not_null=published_at_not_null,
            published_at=published_at
        )
        return response.set_index("id")


    def get_sessions(
        self,
        session_ids: Optional[ArrayLike] = None,
        published_at: Optional[str] = None
    ):

        published_at_not_null = None if published_at is None else True
        published_at = f"'{published_at}'" if published_at is not None else None

        response = build_and_execute(
            """
                {%- import 'postgres_macros' as pm -%}
                {%- import 'macros' as m -%}
                select 
                    es.id, 
                    es.specimen_id, 
                    es.stimulus_name as session_type, 
                    es.isi_experiment_id, 
                    es.date_of_acquisition, 
                    es.published_at,  
                    dn.full_genotype as genotype,
                    gd.name as sex, 
                    ages.days as age_in_days,
                    case 
                        when nwb_id is not null then true
                        else false
                    end as has_nwb
                from ecephys_sessions es
                join specimens sp on sp.id = es.specimen_id 
                join donors dn on dn.id = sp.donor_id 
                join genders gd on gd.id = dn.gender_id 
                join ages on ages.id = dn.age_id
                left join (
                    select ecephys_sessions.id as ecephys_session_id,
                    wkf.id as nwb_id
                    from ecephys_sessions 
                    join ecephys_analysis_runs ear on (
                        ear.ecephys_session_id = ecephys_sessions.id
                        and ear.current
                    )
                    join well_known_files wkf on (
                        wkf.attachable_id = ear.id
                        and wkf.attachable_type = 'EcephysAnalysisRun'
                    )
                    join well_known_file_types wkft on wkft.id = wkf.well_known_file_type_id
                    where wkft.name = 'EcephysNwb'
                ) nwb on es.id = nwb.ecephys_session_id
                where 
                    not es.habituation 
                    and es.workflow_state != 'failed'
                    {{pm.optional_contains('es.id', session_ids) -}}
                    {{pm.optional_not_null('es.published_at', published_at_not_null)}}
                    {{pm.optional_le('es.published_at', published_at)}}
            """,
            base=postgres_macros(),
            engine=self.postgres_engine.select,
            session_ids=session_ids,
            published_at_not_null=published_at_not_null,
            published_at=published_at
        )
        
        response.set_index("id", inplace=True) 
        response["genotype"].fillna("wt", inplace=True)
        return response


    def get_unit_analysis_metrics(self, unit_ids=None, ecephys_session_ids=None, session_types=None):
        response = build_and_execute(
            """
            {%- import 'postgres_macros' as pm -%}
            {%- import 'macros' as m -%}
            select eumb.data, eumb.ecephys_unit_id from ecephys_unit_metric_bundles eumb
            join ecephys_analysis_runs ear on eumb.ecephys_analysis_run_id = ear.id
            join ecephys_units eu on eumb.ecephys_unit_id = eu.id
            join ecephys_channels ec on eu.ecephys_channel_id = ec.id 
            join ecephys_probes ep on ec.ecephys_probe_id = ep.id
            join ecephys_sessions es on es.id = ep.ecephys_session_id
            where ear.current
            {{pm.optional_contains('eumb.id', unit_ids) -}}
            {{pm.optional_contains('es.id', ecephys_session_ids) -}}
            {{pm.optional_contains('es.stimulus_name', session_types, True) -}}
        """,
            base=postgres_macros(),
            engine=self.postgres_engine.select,
            unit_ids=unit_ids,
            ecephys_session_ids=ecephys_session_ids,
            session_types=session_types
        )

        data = pd.DataFrame(response.pop("data").values.tolist(), index=response.index)
        response = pd.merge(response, data, left_index=True, right_index=True)
        response.set_index("ecephys_unit_id", inplace=True)

        return response


    def _get_template(self, name, namespace):
        try:
            well_known_file = build_and_execute(
                f"""
                select 
                    st.well_known_file_id
                from stimuli st
                join stimulus_namespaces sn on sn.id = st.stimulus_namespace_id
                where
                    st.name = '{name}'
                    and sn.name = '{namespace}'
                """,
                base=postgres_macros(),
                engine=self.postgres_engine.select_one
            )
            wkf_id = well_known_file["well_known_file_id"]
        except (KeyError, IndexError):
            raise ValueError(f"expected exactly 1 template for {name}")

        download_link = f"well_known_files/download/{wkf_id}?wkf_id={wkf_id}"
        return self.app_engine.stream(download_link)


    def get_natural_movie_template(self, number: int) -> Generator[bytes, None, None]:
        """ Download a template for the natural movie stimulus. This is the 
        actual movie that was shown during the recording session.

        Parameters
        ----------
        number :
            idenfifier for this movie (note that this is an integer, so to get 
            the template for natural_movie_three you should pass in 3)

        Returns
        -------
        A generator yielding a bytestream of this template's data.

        """

        return self._get_template(f"natural_movie_{number}", self.STIMULUS_NAMESPACE)


    def get_natural_scene_template(self, number: int) -> Generator[bytes, None, None]:
        """ Download a template for the natural scene stimulus. This is the 
        actual image that was shown during the recording session.

        Parameters
        ----------
        number :
            idenfifier for this scene

        Returns
        -------
        A generator yielding a bytestream of this template's data.

        """
        return self._get_template(f"natural_scene_{int(number)}", self.STIMULUS_NAMESPACE)


    @classmethod
    def default(cls, pg_kwargs=None, app_kwargs=None):

        _pg_kwargs = {}
        if pg_kwargs is not None:
            _pg_kwargs.update(pg_kwargs)

        _app_kwargs = {"scheme": "http", "host": "lims2"}
        if app_kwargs is not None:
            _app_kwargs.update(app_kwargs)

        pg_engine = PostgresQueryMixin(**_pg_kwargs)
        app_engine = HttpEngine(**_app_kwargs)
        return cls(pg_engine, app_engine)

from pathlib import Path
import shutil
import warnings

import pandas as pd

from .ecephys_project_api import EcephysProjectApi
from .http_engine import HttpEngine
from .utilities import postgres_macros, build_and_execute

from allensdk.internal.api import PostgresQueryMixin
from allensdk.brain_observatory.ecephys import get_unit_filter_value


class EcephysProjectLimsApi(EcephysProjectApi):
    def __init__(self, postgres_engine, app_engine):
        self.postgres_engine = postgres_engine
        self.app_engine = app_engine

    def get_session_data(self, session_id):
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
        self, unit_ids=None, 
        channel_ids=None, 
        probe_ids=None, 
        session_ids=None, 
        quality="good",
        **kwargs
    ):
        response = build_and_execute(
            """
                {%- import 'postgres_macros' as pm -%}
                {%- import 'macros' as m -%}
                select eu.*
                from ecephys_units eu
                join ecephys_channels ec on ec.id = eu.ecephys_channel_id
                join ecephys_probes ep on ep.id = ec.ecephys_probe_id
                join ecephys_sessions es on es.id = ep.ecephys_session_id 
                where ec.valid_data
                and ep.workflow_state != 'failed'
                and es.workflow_state != 'failed'
                {{pm.optional_equals('eu.quality', quality) -}}
                {{pm.optional_contains('eu.id', unit_ids) -}}
                {{pm.optional_contains('ec.id', channel_ids) -}}
                {{pm.optional_contains('ep.id', probe_ids) -}}
                {{pm.optional_contains('es.id', session_ids) -}}
                {{pm.optional_le('eu.amplitude_cutoff', amplitude_cutoff_maximum) -}}
                {{pm.optional_ge('eu.presence_ratio', presence_ratio_minimum) -}}
                {{pm.optional_le('eu.isi_violations', isi_violations_maximum) -}}
            """,
            base=postgres_macros(),
            engine=self.postgres_engine.select,
            unit_ids=unit_ids,
            channel_ids=channel_ids,
            probe_ids=probe_ids,
            session_ids=session_ids,
            quality=f"'{quality}'" if quality is not None else quality,
            amplitude_cutoff_maximum=get_unit_filter_value("amplitude_cutoff_maximum", replace_none=False, **kwargs),
            presence_ratio_minimum=get_unit_filter_value("presence_ratio_minimum", replace_none=False, **kwargs),
            isi_violations_maximum=get_unit_filter_value("isi_violations_maximum", replace_none=False, **kwargs)
        )

        response.set_index("id", inplace=True)
        response.rename(columns={"ecephys_channel_id": "peak_channel_id"}, inplace=True)

        return response

    def get_channels(self, channel_ids=None, probe_ids=None, session_ids=None, **kwargs):
        response = build_and_execute(
            """
                {%- import 'postgres_macros' as pm -%}
                select 
                    ec.id as id,
                    ec.ecephys_probe_id,
                    ec.local_index,
                    ec.probe_vertical_position,
                    ec.probe_horizontal_position,
                    ec.manual_structure_id,
                    st.acronym as manual_structure_acronym,
                    pc.unit_count
                from ecephys_channels ec 
                join ecephys_probes ep on ep.id = ec.ecephys_probe_id
                join ecephys_sessions es on es.id = ep.ecephys_session_id 
                left join structures st on st.id = ec.manual_structure_id
                join (
                    select ech.id as ecephys_channel_id,
                    count (distinct eun.id) as unit_count
                    from ecephys_channels ech
                    join ecephys_units eun on (
                        eun.ecephys_channel_id = ech.id
                        and eun.quality = 'good'
                        {{pm.optional_le('eun.amplitude_cutoff', amplitude_cutoff_maximum) -}}
                        {{pm.optional_ge('eun.presence_ratio', presence_ratio_minimum) -}}
                        {{pm.optional_le('eun.isi_violations', isi_violations_maximum) -}}
                    )
                    group by ech.id
                ) pc on ec.id = pc.ecephys_channel_id
                where valid_data
                and ep.workflow_state != 'failed'
                and es.workflow_state != 'failed'
                {{pm.optional_contains('ec.id', channel_ids) -}}
                {{pm.optional_contains('ep.id', probe_ids) -}}
                {{pm.optional_contains('es.id', session_ids) -}}
            """,
            base=postgres_macros(),
            engine=self.postgres_engine.select,
            channel_ids=channel_ids,
            probe_ids=probe_ids,
            session_ids=session_ids,
            amplitude_cutoff_maximum=get_unit_filter_value("amplitude_cutoff_maximum", replace_none=False, **kwargs),
            presence_ratio_minimum=get_unit_filter_value("presence_ratio_minimum", replace_none=False, **kwargs),
            isi_violations_maximum=get_unit_filter_value("isi_violations_maximum", replace_none=False, **kwargs)
        )
        return response.set_index("id")

    def get_probes(self, probe_ids=None, session_ids=None, **kwargs):
        response = build_and_execute(
            """
                {%- import 'postgres_macros' as pm -%}
                select 
                    ep.id as id,
                    ep.ecephys_session_id,
                    ep.global_probe_sampling_rate,
                    ep.global_probe_lfp_sampling_rate,
                    total_time_shift,
                    channel_count,
                    unit_count,
                    case 
                        when nwb_id is not null then true
                        else false
                    end as has_lfp_nwb,
                    str.structure_acronyms as structure_acronyms
                from ecephys_probes ep 
                join ecephys_sessions es on es.id = ep.ecephys_session_id 
                join (
                    select epr.id as ecephys_probe_id,
                    count (distinct ech.id) as channel_count,
                    count (distinct eun.id) as unit_count
                    from ecephys_probes epr
                    join ecephys_channels ech on (
                        ech.ecephys_probe_id = epr.id
                        and ech.valid_data
                    )
                    join ecephys_units eun on (
                        eun.ecephys_channel_id = ech.id
                        and eun.quality = 'good'
                        {{pm.optional_le('eun.amplitude_cutoff', amplitude_cutoff_maximum) -}}
                        {{pm.optional_ge('eun.presence_ratio', presence_ratio_minimum) -}}
                        {{pm.optional_le('eun.isi_violations', isi_violations_maximum) -}}
                    )
                    group by epr.id
                ) chc on ep.id = chc.ecephys_probe_id
                left join (
                    select
                        epr.id as ecephys_probe_id,
                        wkf.id as nwb_id
                    from ecephys_probes epr 
                    join ecephys_analysis_runs ear on (
                        ear.ecephys_session_id = epr.ecephys_session_id
                        and ear.current
                    )
                    right join ecephys_analysis_run_probes earp on (
                        earp.ecephys_probe_id = epr.id
                        and earp.ecephys_analysis_run_id = ear.id
                    )
                    right join well_known_files wkf on (
                        wkf.attachable_id = earp.id
                        and wkf.attachable_type = 'EcephysAnalysisRunProbe'
                    )
                    join well_known_file_types wkft on wkft.id = wkf.well_known_file_type_id
                    where wkft.name = 'EcephysLfpNwb'
                ) nwb on ep.id = nwb.ecephys_probe_id
                left join (
                    select epr.id as ecephys_probe_id,
                    array_agg (st.id) as structure_ids,
                    array_agg (distinct st.acronym) as structure_acronyms
                    from ecephys_probes epr
                    join ecephys_channels ech on (
                        ech.ecephys_probe_id = epr.id
                        and ech.valid_data
                    )
                    left join structures st on st.id = ech.manual_structure_id
                    group by epr.id
                ) str on ep.id = str.ecephys_probe_id
                where true
                and ep.workflow_state != 'failed'
                and es.workflow_state != 'failed'
                {{pm.optional_contains('ep.id', probe_ids) -}}
                {{pm.optional_contains('es.id', session_ids) -}}
            """,
            base=postgres_macros(),
            engine=self.postgres_engine.select,
            probe_ids=probe_ids,
            session_ids=session_ids,
            amplitude_cutoff_maximum=get_unit_filter_value("amplitude_cutoff_maximum", replace_none=False, **kwargs),
            presence_ratio_minimum=get_unit_filter_value("presence_ratio_minimum", replace_none=False, **kwargs),
            isi_violations_maximum=get_unit_filter_value("isi_violations_maximum", replace_none=False, **kwargs)
        )
        return response.set_index("id")

    def get_sessions(
        self,
        session_ids=None,
        workflow_states=("uploaded",),
        published=None,
        habituation=False,
        project_names=(
            "BrainTV Neuropixels Visual Behavior",
            "BrainTV Neuropixels Visual Coding",
        ),
        **kwargs
    ):

        response = build_and_execute(
            """
                {%- import 'postgres_macros' as pm -%}
                {%- import 'macros' as m -%}
                select 
                    stimulus_name as session_type,
                    sp.id as specimen_id, 
                    es.id as id, 
                    dn.full_genotype as genotype,
                    gd.name as gender, 
                    ages.days as age_in_days,
                    pr.code as project_code,
                    probe_count,
                    channel_count,
                    unit_count,
                    case 
                        when nwb_id is not null then true
                        else false
                    end as has_nwb,
                    str.structure_acronyms as structure_acronyms
                from ecephys_sessions es
                join specimens sp on sp.id = es.specimen_id 
                join donors dn on dn.id = sp.donor_id 
                join genders gd on gd.id = dn.gender_id 
                join ages on ages.id = dn.age_id
                join projects pr on pr.id = es.project_id
                join (
                    select es.id as ecephys_session_id,
                    count (distinct epr.id) as probe_count,
                    count (distinct ech.id) as channel_count,
                    count (distinct eun.id) as unit_count
                    from ecephys_sessions es
                    join ecephys_probes epr on epr.ecephys_session_id = es.id
                    join ecephys_channels ech on (
                        ech.ecephys_probe_id = epr.id
                        and ech.valid_data
                    )
                    join ecephys_units eun on (
                        eun.ecephys_channel_id = ech.id
                        and eun.quality = 'good'
                        {{pm.optional_le('eun.amplitude_cutoff', amplitude_cutoff_maximum) -}}
                        {{pm.optional_ge('eun.presence_ratio', presence_ratio_minimum) -}}
                        {{pm.optional_le('eun.isi_violations', isi_violations_maximum) -}}
                    )
                    group by es.id
                ) pc on es.id = pc.ecephys_session_id
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
                left join (
                    select es.id as ecephys_session_id,
                    array_agg (st.id) as structure_ids,
                    array_agg (distinct st.acronym) as structure_acronyms
                    from ecephys_sessions es
                    join ecephys_probes epr on epr.ecephys_session_id = es.id
                    join ecephys_channels ech on (
                        ech.ecephys_probe_id = epr.id
                        and ech.valid_data
                    )
                    left join structures st on st.id = ech.manual_structure_id
                    group by es.id
                ) str on es.id = str.ecephys_session_id
                where true
                {{pm.optional_contains('es.id', session_ids) -}}
                {{pm.optional_contains('es.workflow_state', workflow_states, True) -}}
                {{pm.optional_equals('es.habituation', habituation) -}}
                {{pm.optional_not_null('es.published_at', published) -}}
                {{pm.optional_contains('pr.name', project_names, True) -}}
            """,
            base=postgres_macros(),
            engine=self.postgres_engine.select,
            session_ids=session_ids,
            workflow_states=workflow_states,
            published=published,
            habituation=f"{habituation}".lower() if habituation is not None else habituation,
            project_names=project_names,
            amplitude_cutoff_maximum=get_unit_filter_value("amplitude_cutoff_maximum", replace_none=False, **kwargs),
            presence_ratio_minimum=get_unit_filter_value("presence_ratio_minimum", replace_none=False, **kwargs),
            isi_violations_maximum=get_unit_filter_value("isi_violations_maximum", replace_none=False, **kwargs)
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

from pathlib import Path
import shutil
import warnings

import pandas as pd

from .ecephys_project_api import EcephysProjectApi
from .http_engine import HttpEngine
from .utilities import postgres_macros, build_and_execute

from allensdk.internal.api import PostgresQueryMixin


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
        self, unit_ids=None, channel_ids=None, probe_ids=None, session_ids=None
    ):
        return build_and_execute(
            """
                {%- import 'postgres_macros' as pm -%}
                select eu.* from ecephys_units eu
                join ecephys_channels ec on ec.id = eu.ecephys_channel_id
                join ecephys_probes ep on ep.id = ec.ecephys_probe_id
                join ecephys_sessions es on es.id = ep.ecephys_session_id 
                where true
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
        )

    def get_channels(self, channel_ids=None, probe_ids=None, session_ids=None):
        return build_and_execute(
            """
                {%- import 'postgres_macros' as pm -%}
                select ec.* from ecephys_channels ec 
                join ecephys_probes ep on ep.id = ec.ecephys_probe_id
                join ecephys_sessions es on es.id = ep.ecephys_session_id 
                where true
                {{pm.optional_contains('ec.id', channel_ids) -}}
                {{pm.optional_contains('ep.id', probe_ids) -}}
                {{pm.optional_contains('es.id', session_ids) -}}
            """,
            base=postgres_macros(),
            engine=self.postgres_engine.select,
            channel_ids=channel_ids,
            probe_ids=probe_ids,
            session_ids=session_ids,
        )

    def get_probes(self, probe_ids=None, session_ids=None):
        return build_and_execute(
            """
                {%- import 'postgres_macros' as pm -%}
                select ep.* from ecephys_probes ep 
                join ecephys_sessions es on es.id = ep.ecephys_session_id 
                where true
                {{pm.optional_contains('ep.id', probe_ids) -}}
                {{pm.optional_contains('es.id', session_ids) -}}
            """,
            base=postgres_macros(),
            engine=self.postgres_engine.select,
            probe_ids=probe_ids,
            session_ids=session_ids,
        )

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
    ):
        return build_and_execute(
            """
                {%- import 'postgres_macros' as pm -%}
                {%- import 'macros' as m -%}
                select 
                    stimulus_name as stimulus_set_name,
                    sp.id as specimen_id, 
                    es.id as ecephys_session_id, 
                    dn.full_genotype as genotype,
                    gd.name as gender, 
                    ages.name as age,
                    pr.code as project_code
                from ecephys_sessions es 
                join specimens sp on sp.id = es.specimen_id 
                join donors dn on dn.id = sp.donor_id 
                join genders gd on gd.id = dn.gender_id 
                join ages on ages.id = dn.age_id
                join projects pr on pr.id = es.project_id
                where true
                {{pm.optional_contains('es.id', session_ids) -}}
                {{pm.optional_contains('es.workflow_state', workflow_states, True) -}}
                {{pm.optional_equals('es.habituation', m.str(habituation).lower()) -}}
                {{pm.optional_not_null('es.published_at', published) -}}
                {{pm.optional_contains('pr.name', project_names, True) -}}
            """,
            base=postgres_macros(),
            engine=self.postgres_engine.select,
            session_ids=session_ids,
            workflow_states=workflow_states,
            published=published,
            habituation=habituation,
            project_names=project_names,
        )

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

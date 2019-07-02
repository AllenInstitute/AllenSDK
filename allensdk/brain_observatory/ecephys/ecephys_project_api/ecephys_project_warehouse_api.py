import pandas as pd

from .rma_engine import RmaEngine
from .ecephys_project_api import EcephysProjectApi
from .utilities import rma_macros, build_and_execute

class EcephysProjectWarehouseApi(EcephysProjectApi):
    
    def __init__(self, rma_engine):
        self.rma_engine = rma_engine

    def get_session_data(self, session_id):
        well_known_files = build_and_execute(
            (
                "criteria=model::WellKnownFile"
                ",rma::criteria,well_known_file_type[name$eq'EcephysNwb']"
                "[attachable_type$eq'EcephysSession']"
                r"[attachable_id$eq{{session_id}}]"
            ),
            engine=self.rma_engine.get_rma_tabular, session_id=session_id
        )

        if well_known_files.shape[0] != 1:
            raise ValueError(f"expected exactly 1 nwb file for session {session_id}, found: {well_known_files}")
        
        download_link = well_known_files.loc[0, "download_link"]
        return self.rma_engine.stream(download_link)
        
    def get_probe_lfp_data(self, probe_id):
        well_known_files = build_and_execute(
            (
                "criteria=model::WellKnownFile"
                ",rma::criteria,well_known_file_type[name$eq'EcephysNwb']"
                "[attachable_type$eq'EcephysProbe']"
                r"[attachable_id$eq{{probe_id}}]"
            ),
            engine=self.rma_engine.get_rma_tabular, probe_id=probe_id
        )

        if well_known_files.shape[0] != 1:
            raise ValueError(f"expected exactly 1 LFP NWB file for probe {probe_id}, found: {well_known_files}")
        
        download_link = well_known_files.loc[0, "download_link"]
        return self.rma_engine.stream(download_link)


    def get_sessions(self, session_ids=None, has_eye_tracking=None, stimulus_names=None):
        criteria = session_ids is not None or has_eye_tracking is not None or stimulus_names is not None
        return build_and_execute(
            (
                "{% import 'rma_macros' as rm %}"
                "{% import 'macros' as m %}"
                "criteria=model::EcephysSession"
                r"{{',rma::criteria' if criteria}}"
                r"{{rm.optional_contains('id',session_ids)}}"
                r"{%if has_eye_tracking is not none%}[fail_eye_tracking$eq{{m.str(not has_eye_tracking).lower()}}]{%endif%}"
                r"{{rm.optional_contains('stimulus_name',stimulus_names,True)}}"
            ), 
            base=rma_macros(), engine=self.rma_engine.get_rma_tabular, criteria=criteria, session_ids=session_ids, 
            has_eye_tracking=has_eye_tracking, stimulus_names=stimulus_names
        )

    def get_probes(self, probe_ids=None, session_ids=None):
        criteria = probe_ids is not None and session_ids is not None
        return build_and_execute(
            (
                "{% import 'rma_macros' as rm %}"
                "{% import 'macros' as m %}"           
                "criteria=model::EcephysProbe"
                r"{{',rma::criteria' if criteria}}"
                r"{{rm.optional_contains('id',probe_ids)}}"
                r"{{rm.optional_contains('ecephys_session_id',session_ids)}}"
            ),
            base=rma_macros(), engine=self.rma_engine.get_rma_tabular, session_ids=session_ids, probe_ids=probe_ids,
            criteria=criteria
        )

    def get_channels(self, channel_ids=None, probe_ids=None, session_ids=None):
        criteria = probe_ids is not None and session_ids is not None
        return build_and_execute(
            (
                "{% import 'rma_macros' as rm %}"
                "{% import 'macros' as m %}"           
                "criteria=model::EcephysChannel"
                r"{{',rma::criteria' if criteria}}"
                r"{{rm.optional_contains('id',channel_ids)}}"
                r"{{rm.optional_contains('ecephys_probe_id',probe_ids)}}"
                r"{%if session_ids is not none%},rma::criteria,ecephys_probe[ecephys_session_id$in{{m.comma_sep(session_ids)}}]{%endif%}"
            ),
            base=rma_macros(), engine=self.rma_engine.get_rma_tabular, session_ids=session_ids, probe_ids=probe_ids,
            channel_ids=channel_ids, criteria=criteria
        )

    def get_units(self, unit_ids=None, channel_ids=None, probe_ids=None, session_ids=None):
        criteria = probe_ids is not None and session_ids is not None
        return build_and_execute(
            (
                "{% import 'macros' as m %}" 
                "criteria=model::EcephysUnit"
                r"{% if unit_ids is not none %},rma::criteria[id$in{{m.comma_sep(unit_ids)}}]{% endif %}"
                r"{% if channel_ids is not none %},rma::criteria[ecephys_channel_id$in{{m.comma_sep(channel_ids)}}]{% endif %}"
                r"{% if probe_ids is not none %},rma::criteria,ecephys_channel(ecephys_probe[id$in{{m.comma_sep(probe_ids)}}]){% endif %}"
                r"{% if session_ids is not none %},rma::criteria,ecephys_channel(ecephys_probe(ecephys_session[id$in{{m.comma_sep(session_ids)}}])){% endif %}"
            ),
            base=rma_macros(), engine=self.rma_engine.get_rma_tabular, session_ids=session_ids, probe_ids=probe_ids,
            channel_ids=channel_ids, unit_ids=unit_ids, criteria=criteria
        )
    @classmethod
    def default(cls, **rma_kwargs):
        _rma_kwargs = {"scheme": "http", "host": "api.brain-map.org"}
        _rma_kwargs.update(rma_kwargs)
        return cls(RmaEngine(**_rma_kwargs))
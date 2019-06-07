import pandas as pd

from .rma_engine import RmaEngine
from .ecephys_project_api import EcephysProjectApi
from .utilities import rma_macros, build_and_execute

class EcephysProjectWarehouseApi(EcephysProjectApi):
    
    def __init__(self, rma_engine):
        self.rma_engine = rma_engine

    def get_sessions(self, session_ids=None, has_eye_tracking=None, stimulus_names=None):
        criteria = session_ids is not None or has_eye_tracking is not None or stimulus_names is not None
        stream = build_and_execute(
            (
                "{% import 'rma_macros' as rm %}"
                "{% import 'macros' as m %}"
                "criteria=model::EcephysSession"
                r"{{',rma::criteria' if criteria}}"
                r"{{rm.optional_contains('id',session_ids)}}"
                r"{%if has_eye_tracking is not none%}[fail_eye_tracking$eq{{m.str(not has_eye_tracking).lower()}}]{%endif%}"
                r"{{rm.optional_contains('stimulus_name',stimulus_names,True)}}"
            ), 
            base=rma_macros(), engine=self.rma_engine.get, criteria=criteria, session_ids=session_ids, 
            has_eye_tracking=has_eye_tracking, stimulus_names=stimulus_names
        )

        response = []
        for chunk in stream:
            response.extend(chunk)
        return pd.DataFrame(response)



    @classmethod
    def default(cls, **rma_kwargs):
        _rma_kwargs = {"scheme": "http", "host": "api.brain-map.org"}
        _rma_kwargs.update(rma_kwargs)
        return cls(RmaEngine(**_rma_kwargs))
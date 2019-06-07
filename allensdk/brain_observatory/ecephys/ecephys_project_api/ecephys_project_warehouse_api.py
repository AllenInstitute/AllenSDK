from allensdk.api.queries.rma_api import RmaApi
from .ecephys_project_api import EcephysProjectApi


from .utilities import rma_macros, build_and_execute

class EcephysProjectWarehouseApi(EcephysProjectApi):
    
    def __init__(self, rma_engine):
        self.rma_engine = rma_engine

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
            base=rma_macros(), engine=self.rma_engine.get, criteria=criteria, session_ids=session_ids, 
            has_eye_tracking=has_eye_tracking, stimulus_names=stimulus_names
        )


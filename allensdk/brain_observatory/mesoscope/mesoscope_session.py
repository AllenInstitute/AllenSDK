from allensdk.core.lazy_property import LazyProperty, LazyPropertyMixin
import pandas as pd
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession as MesoscopeOphysPlane
from allensdk.internal.api.mesoscope_lims_api import MesoscopeSessionLimsApi, MesoscopePlaneLimsApi

class MesoscopeSession(LazyPropertyMixin):

    @classmethod
    def from_lims(cls, session_id):
        return cls(api=MesoscopeSessionLimsApi(session_id))

    def __init__(self, api=None):

        self.api = api
        self.session_id = LazyProperty(self.api.get_session_id)
        self.session_df = LazyProperty(self.api.get_session_df)
        self.experiments_ids = LazyProperty(self.api.get_session_experiments)
        self.pairs = LazyProperty(self.api.get_paired_experiments)
        self.splitting_json =LazyProperty(self.api.get_splitting_json)
        self.folder = LazyProperty(self.api.get_session_folder)

    def get_exp_by_structure(self, structure):
        return self.experiments.loc[self.session_df.structure == structure]

    def get_planes(self):
        planes = pd.DataFrame(columns=['plane_id', 'plane'], index=range(len(self.experiments_ids['experiment_id'])))
        i=0
        for experiment_id in self.experiments_ids['experiment_id']:
            plane = MesoscopeOphysPlane(api=MesoscopePlaneLimsApi(experiment_id))
            planes.plane_id[i] = experiment_id
            planes.plane[i] = plane
            i += 1
        return planes

if __name__ == "__main__":

    session = MesoscopeSession.from_lims(754606824)
    print(session.experiments_ids)
    pd.options.display.width = 0
    print(session.session_df)
    print(session.folder)
    print(session.session_id)
    print(session.splitting_json)
    print(session.pairs)
    print(session.get_planes())


    



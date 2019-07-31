from allensdk.core.lazy_property import LazyProperty, LazyPropertyMixin
import pandas as pd
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
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
        self.planes_timestamps = LazyProperty(self.api.split_session_timestamps)

    def get_exp_by_structure(self, structure):
        return self.session_df.loc[self.session_df.structure == structure]

    def get_planes(self):
        self.planes = pd.DataFrame(columns=['plane_id', 'plane'], index=range(len(self.experiments_ids['experiment_id'])))
        i=0
        for experiment_id in self.experiments_ids['experiment_id']:
            plane = MesoscopeOphysPlane(api=MesoscopePlaneLimsApi(experiment_id))
            self.planes.plane_id[i] = experiment_id
            self.planes.plane[i] = plane
            i += 1
        return self.planes

    def get_plane_timestamp(self, exp_id):
        return self.planes_timestamps.loc[self.planes_timestamps['plane_id'] == exp_id]

class  MesoscopeOphysPlane(BehaviorOphysSession):

    @classmethod
    def from_lims(cls, experiment_id):
        return cls(api=MesoscopePlaneLimsApi(experiment_id))

    def __init__(self, api=None):
        self.api = api
        self.experiment_df = LazyProperty(self.api.get_experiment_df)
        self.experiment_id = LazyProperty(self.api.get_experiment_id)
        self.session_id = LazyProperty(self.api.get_session_id)


if __name__ == "__main__":

    session = MesoscopeSession.from_lims(754606824)
    # print(session.experiments_ids)
    pd.options.display.width = 0
    # print(session.session_df)
    # print(session.folder)
    # print(session.session_id)
    # print(session.splitting_json)
    # print(session.pairs)
    planes = session.get_planes()


    



from allensdk.core.lazy_property import LazyProperty, LazyPropertyMixin
import pandas as pd
from allensdk.internal.api.mesoscope_session_lims_api import MesoscopeSessionLimsApi
from allensdk.brain_observatory.mesoscope.mesoscope_plane import MesoscopeOphysPlane

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
        super().__init__()

    def get_exp_by_structure(self, structure):
        return self.session_df.loc[self.session_df.structure == structure]

    def get_planes(self):
        self.planes = pd.DataFrame(columns=['plane_id', 'plane'], index=range(len(self.experiments_ids['experiment_id'])))
        i=0
        for experiment_id in self.experiments_ids['experiment_id']:
            plane = MesoscopeOphysPlane(experiment_id)
            self.planes.plane_id[i] = experiment_id
            self.planes.plane[i] = plane
            i += 1
        return self.planes

    def get_plane_timestamps(self, exp_id):
        return self.planes_timestamps[self.planes_timestamps.plane_id == exp_id].reset_index().loc[0, 'ophys_timestamps']


if __name__ == "__main__":

    session = MesoscopeSession.from_lims(754606824)
    pd.options.display.width = 0
    planes = session.get_planes()
    print(planes)
    print(session.session_df)
    plane = planes[planes.plane_id == 807310592].plane.values[0]
    print(plane.licks)



    



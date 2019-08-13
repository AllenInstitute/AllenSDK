from allensdk.core.lazy_property import LazyProperty, LazyPropertyMixin
import pandas as pd
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
afrom allensdk.internal.api.mesoscope_lims_api import MesoscopePlaneLimsApi, MesoscopeSessionLimsApi

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
            plane = MesoscopeOphysPlane(api=MesoscopePlaneLimsApi(experiment_id, session = self))
            self.planes.plane_id[i] = experiment_id
            self.planes.plane[i] = plane
            i += 1
        return self.planes

    def get_plane_timestamps(self, exp_id):
        return self.planes_timestamps[self.planes_timestamps.plane_id == exp_id].reset_index().loc[0, 'ophys_timestamps']

class  MesoscopeOphysPlane(BehaviorOphysSession):

    @classmethod
    def from_lims(cls, experiment_id):
        return cls(api=MesoscopePlaneLimsApi(experiment_id))

    def __init__(self, api=None):

        self.api = api
        self.ophys_experiment_id = LazyProperty(self.api.get_ophys_experiment_id)
        self.max_projection = LazyProperty(self.api.get_max_projection)
        self.stimulus_timestamps = LazyProperty(self.api.get_stimulus_timestamps)
        self.ophys_timestamps = LazyProperty(self.api.get_ophys_timestamps)
        self.metadata = LazyProperty(self.api.get_metadata)
        self.dff_traces = LazyProperty(self.api.get_dff_traces)
        self.cell_specimen_table = LazyProperty(self.api.get_cell_specimen_table)
        self.running_speed = LazyProperty(self.api.get_running_speed)
        self.running_data_df = LazyProperty(self.api.get_running_data_df)
        self.stimulus_presentations = LazyProperty(self.api.get_stimulus_presentations)
        self.stimulus_templates = LazyProperty(self.api.get_stimulus_templates)
        self.licks = LazyProperty(self.api.get_licks)
        self.rewards = LazyProperty(self.api.get_rewards)
        self.task_parameters = LazyProperty(self.api.get_task_parameters)
        self.trials = LazyProperty(self.api.get_trials)
        self.corrected_fluorescence_traces = LazyProperty(self.api.get_corrected_fluorescence_traces)
        self.average_projection = LazyProperty(self.api.get_average_projection)
        self.motion_correction = LazyProperty(self.api.get_motion_correction)
        self.segmentation_mask_image = LazyProperty(self.api.get_segmentation_mask_image)

        self.experiment_df = LazyProperty(self.api.get_experiment_df)
        self.ophys_session_id = LazyProperty(self.api.get_ophys_session_id)

if __name__ == "__main__":

    session = MesoscopeSession.from_lims(754606824)
    pd.options.display.width = 0
    planes = session.get_planes()
    print(planes)
    print(session.session_df)
    plane = planes[planes.plane_id == 807310592].plane.values[0]
    print(plane.licks)



    



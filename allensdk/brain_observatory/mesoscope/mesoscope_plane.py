from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.internal.api.mesoscope_session_lims_api import MesoscopePlaneLimsApi
from allensdk.core.lazy_property import LazyProperty

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

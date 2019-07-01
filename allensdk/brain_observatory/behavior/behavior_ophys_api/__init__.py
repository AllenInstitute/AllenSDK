from allensdk.brain_observatory.behavior.behavior_api import BehaviorApiBase
class BehaviorOphysApiBase(BehaviorApiBase):

    def get_ophys_experiment_id(self) -> int:
        raise RuntimeError('Use ophys_experiment_id not behavior_session_id')

    def get_ophys_experiment_id(self) -> int:
        return self.get_metadata()['ophys_experiment_id']

    def get_max_projection(self):
        raise NotImplementedError

    def get_stimulus_timestamps(self):
        raise NotImplementedError

    def get_ophys_timestamps(self):
        raise NotImplementedError

    def get_dff_traces(self):
        raise NotImplementedError

    def get_cell_specimen_table(self):
        raise NotImplementedError

    def get_corrected_fluorescence_traces(self):
        raise NotImplementedError

    def get_motion_correction(self):
        raise NotImplementedError

    def get_average_projection(self):
        raise NotImplementedError

    def get_segmentation_mask_image(self):
        raise NotImplementedError

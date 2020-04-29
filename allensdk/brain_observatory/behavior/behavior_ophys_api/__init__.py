
class BehaviorOphysApiBase:

    def get_ophys_experiment_id(self) -> int:
        return self.get_metadata()['ophys_experiment_id']

    def get_max_projection(self):
        raise NotImplementedError

    def get_stimulus_timestamps(self):
        raise NotImplementedError

    def get_ophys_timestamps(self):
        raise NotImplementedError

    def get_metadata(self):
        raise NotImplementedError

    def get_dff_traces(self):
        raise NotImplementedError

    def get_cell_specimen_table(self):
        raise NotImplementedError

    def get_running_speed(self):
        raise NotImplementedError

    def get_running_data_df(self):
        raise NotImplementedError

    def get_stimulus_presentations(self):
        raise NotImplementedError

    def get_stimulus_templates(self):
        raise NotImplementedError

    def get_licks(self):
        raise NotImplementedError

    def get_rewards(self):
        raise NotImplementedError

    def get_task_parameters(self):
        raise NotImplementedError

    def get_trials(self):
        raise NotImplementedError

    def get_corrected_fluorescence_traces(self):
        raise NotImplementedError

    def get_motion_correction(self):
        raise NotImplementedError

    def get_average_projection(self):
        raise NotImplementedError

    def get_segmentation_mask_image(self):
        raise NotImplementedError

    def get_eye_tracking_data(self):
        raise NotImplementedError


class BehaviorApiBase:

    def get_behavior_session_id(self) -> int:
        return self.get_metadata()['behavior_session_id']

    def get_stimulus_timestamps(self):
        raise NotImplementedError

    def get_metadata(self):
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

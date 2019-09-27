from allensdk.brain_observatory.behavior.behavior_project_api import BehaviorProjectApi

class MissingDataError(ValueError):
    pass

class BehaviorProjectFixedApi(BehaviorProjectApi):

    def get_session_nwb(self, *args, **kwargs):
        raise MissingDataError(f"Data not found!")

    def get_session_trial_response(self, *args, **kwargs):
        raise MissingDataError(f"Data not found!")

    def get_session_flash_response(self, *args, **kwargs):
        raise MissingDataError(f"Data not found!")

    def get_session_extended_stimulus_columns(self, *args, **kwargs):
        raise MissingDataError(f"Data not found!")

    def get_sessions(self, *args, **kwargs):
        raise MissingDataError(f"Data not found!")

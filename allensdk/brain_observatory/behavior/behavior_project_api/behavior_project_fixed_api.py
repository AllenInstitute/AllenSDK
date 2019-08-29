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

    def get_targeted_regions(self, *args, **kwargs):
        raise MissingDataError(f"Data not found!")

    def get_isi_experiments(self, *args, **kwargs):
        raise MissingDataError(f"Data not found!")

    def get_units(self, *args, **kwargs):
        raise MissingDataError(f"Data not found!")

    def get_channels(self, *args, **kwargs):
        raise MissingDataError(f"Data not found!")

    def get_probes(self, *args, **kwargs):
        raise MissingDataError(f"Data not found!")

    def get_natural_movie_template(self, number, *args, **kwargs):
        raise MissingDataError(f"natural movie template not found for movie {number}")

    def get_natural_scene_template(self, number, *args, **kwargs):
        raise MissingDataError(f"natural scene template not found for scene {number}")

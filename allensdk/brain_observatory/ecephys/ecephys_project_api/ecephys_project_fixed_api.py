from allensdk.brain_observatory.ecephys.ecephys_project_api import EcephysProjectApi


class MissngDataError(ValueError):
    pass


class EcephysProjectFixedApi(EcephysProjectApi):

    SPECIALIZED = ("get_session_data", "get_probe_data")

    def get_session_data(self, session_id):
        raise MissngDataError(f"data for session {session_id} not found!")

    def get_probe_lfp_data(self, probe_id):
        raise MissngDataError(f"lfp data for probe {probe_id} not found!")

    def __getattr__(self, key):
        if key in self.SPECIALIZED:
            return self.__dict__[key]

        def cannot_get(*args, **kwargs):
            raise MissingDataError(f"Data not found (you called {key} with args {args} and kwargs {kwargs})!")
        return cannot_get
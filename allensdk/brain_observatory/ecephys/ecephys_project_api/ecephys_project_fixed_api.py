from allensdk.brain_observatory.ecephys.ecephys_project_api import EcephysProjectApi


class MissingDataError(ValueError):
    pass


class EcephysProjectFixedApi(EcephysProjectApi):

    def get_session_data(self, session_id):
        raise MissingDataError(f"data for session {session_id} not found!")

    def get_probe_lfp_data(self, probe_id):
        raise MissingDataError(f"lfp data for probe {probe_id} not found!")

    def get_sessions(self):
        raise MissingDataError(f"Data not found!")

    def get_targeted_regions(self):
        raise MissingDataError(f"Data not found!")

    def get_isi_experiments(self):
        raise MissingDataError(f"Data not found!")

    def get_units(self):
        raise MissingDataError(f"Data not found!")

    def get_channels(self):
        raise MissingDataError(f"Data not found!")

    def get_probes(self):
        raise MissingDataError(f"Data not found!")

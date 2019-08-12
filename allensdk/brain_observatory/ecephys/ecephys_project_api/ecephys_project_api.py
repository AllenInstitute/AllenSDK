class EcephysProjectApi:
    def get_sessions(self):
        raise NotImplementedError()

    def get_session_data(self, session_id):
        raise NotImplementedError()

    def get_targeted_regions(self):
        raise NotImplementedError()

    def get_isi_experiments(self):
        raise NotImplementedError()

    def get_units(self):
        raise NotImplementedError()

    def get_channels(self):
        raise NotImplementedError()

    def get_probes(self):
        raise NotImplementedError()

    def get_probe_lfp_data(self, probe_id):
        raise NotImplementedError()

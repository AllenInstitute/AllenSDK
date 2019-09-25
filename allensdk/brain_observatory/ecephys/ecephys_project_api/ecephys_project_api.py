class EcephysProjectApi:
    def get_sessions(self, *args, **kwargs):
        raise NotImplementedError()

    def get_session_data(self, session_id, *args, **kwargs):
        raise NotImplementedError()

    def get_targeted_regions(self, *args, **kwargs):
        raise NotImplementedError()

    def get_isi_experiments(self, *args, **kwargs):
        raise NotImplementedError()

    def get_units(self, *args, **kwargs):
        raise NotImplementedError()

    def get_channels(self, *args, **kwargs):
        raise NotImplementedError()

    def get_probes(self, *args, **kwargs):
        raise NotImplementedError()

    def get_probe_lfp_data(self, probe_id, *args, **kwargs):
        raise NotImplementedError()

    def get_natural_movie_template(self, number, *args, **kwargs):
        raise NotImplementedError()

    def get_natural_scene_template(self, number, *args, **kwargs):
        raise NotImplementedError()

    def get_unit_analysis_metrics(self, unit_ids=None, ecephys_session_ids=None, session_types=None, *args, **kwargs):
        raise NotImplementedError()
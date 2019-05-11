from allensdk.brain_observatory.ecephys.stimulus_analysis import StimulusAnalysis


class DriftingGratings(StimulusAnalysis):
    def __init__(self, ecephys_session, **kwargs):
        super(DriftingGratings, self).__init__(ecephys_session, **kwargs)

        self._orivals = None
        self._number_ori = None
        self._tfvals = None
        self._number_tf = None

    @property
    def orivals(self):
        if self._orivals is None:
            self._get_stim_table_stats()

        return self._orivals

    @property
    def number_ori(self):
        if self._number_ori is None:
            self._get_stim_table_stats()

        return self._number_ori

    @property
    def tfvals(self):
        if self._tfvals is None:
            self._get_stim_table_stats()

        return self._tfvals

    def _get_stim_table_stats(self):
        pass

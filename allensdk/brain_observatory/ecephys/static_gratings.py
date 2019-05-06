from .stimulus_analysis import StimulusAnalysis


class StaticGratings(StimulusAnalysis):
    def __init__(self, ecephys_session, **kwargs):
        super(StaticGratings, self).__init__(ecephys_session, **kwargs)



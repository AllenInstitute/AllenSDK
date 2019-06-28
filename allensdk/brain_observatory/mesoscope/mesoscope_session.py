from allensdk.core.lazy_property import LazyProperty, LazyPropertyMixin
from allensdk.internal.api.mesoscope_lims_api import MesoscopeLimsApi

class MesoscopeSession(LazyPropertyMixin):

    @classmethod
    def from_lims(cls, session_id):
        return cls(api=MesoscopeLimsApi(session_id))

    def __init__(self, api=None):

        self.api = api
        self.session_id = LazyProperty(self.api.get_session_id)
        self.metadata = LazyProperty(self.api.get_metadata)
        self.session_df = LazyProperty(self.api.get_mesoscope_session_df)

if __name__ == "__main__":

    session = MesoscopeSession.from_lims(754606824)
    print(session.session_id)
    print(session.session_df)
    # print(session.metadata)
    
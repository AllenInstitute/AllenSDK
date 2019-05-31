import pandas as pd

from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession


class EcephysProjectApi:

    def __init__(self, *args, **kwargs):
        pass

    def get_sessions(self) -> pd.DataFrame:
        raise NotImplementedError()

    def get_session_data(self) -> EcephysSession:
        raise NotImplementedError()

    def get_targeted_regions(self) -> pd.DataFrame:
        raise NotImplementedError()

    def get_isi_experiments(self) -> pd.DataFrame:
        raise NotImplementedError()

    def get_units(self) -> pd.DataFrame:
        raise NotImplementedError()

    def get_channels(self) -> pd.DataFrame:
        raise NotImplementedError()

    def get_probes(self) -> pd.DataFrame:
        raise NotImplementedError()

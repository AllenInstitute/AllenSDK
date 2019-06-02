from typing import Dict

import numpy as np
import pandas as pd

from ...running_speed import RunningSpeed


class EcephysSessionApi:

    session_na = -1

    __slots__: tuple = tuple([])

    def __init__(self, *args, **kwargs):
        pass

    def get_running_speed(self) -> RunningSpeed:
        raise NotImplementedError

    def get_stimulus_presentations(self) -> pd.DataFrame:
        raise NotImplementedError

    def get_probes(self) -> pd.DataFrame:
        raise NotImplementedError

    def get_channels(self) -> pd.DataFrame:
        raise NotImplementedError

    def get_mean_waveforms(self) -> Dict[int, np.ndarray]:
        raise NotImplementedError

    def get_spike_times(self) -> Dict[int, np.ndarray]:
        raise NotImplementedError

    def get_units(self) -> pd.DataFrame:
        raise NotImplementedError

    def get_ecephys_session_id(self) -> int:
        raise NotImplementedError

    def get_actual_sampling_rates(self) -> Dict[int, float]:
        """Returns a dictionary mapping the actual sampling rate for every probe (or channel?)."""
        raise NotImplementedError

    def get_baseline_sampling_rate(self) -> float:
        """Returns the global theoretical sampling rate used during the session."""
        # TODO: Default to 30kHz for now but this value should be stored/cacluated from the NWB
        return 30000.0

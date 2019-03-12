from typing import Dict

import numpy as np
import pandas as pd

from ... import RunningSpeed

class EcephysApi:

    __slots__: tuple = tuple([])

    def __init__(self, *args, **kwargs):
        pass

    def get_running_speed(self) -> RunningSpeed:
        raise NotImplementedError
    
    def get_stimulus_table(self) -> pd.DataFrame:
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

import numpy as np
from xarray import DataArray

from allensdk.brain_observatory.ecephys.file_io.continuous_file import \
    ContinuousFile
from allensdk.core import DataObject


class LFP(DataObject):
    """
    Probe LFP
    """
    def __init__(
            self,
            data: np.ndarray,
            timestamps: np.ndarray,
            channels: np.ndarray,
            sampling_rate: float
    ):
        """

        Parameters
        ----------
        data:
            LFP data
        timestamps:
            LFP timestamps
        channels:
            LFP channels
        sampling_rate
            LFP sampling rate
        """
        super().__init__(name='lfp', value=None, is_value_self=True)
        self._data = data
        self._timestamps = timestamps
        self._channels = channels
        self._sampling_rate = sampling_rate

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def timestamps(self) -> np.ndarray:
        return self._timestamps

    @property
    def channels(self) -> np.ndarray:
        return self._channels

    @property
    def sampling_rate(self):
        return self._sampling_rate

    def to_dataarray(self) -> DataArray:
        return DataArray(
            name="LFP",
            data=self._data,
            dims=['time', 'channel'],
            coords=[self._timestamps, self._channels]
        )

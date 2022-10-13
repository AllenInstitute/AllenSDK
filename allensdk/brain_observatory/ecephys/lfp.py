from typing import Optional

import numpy as np
from xarray import DataArray

from allensdk.brain_observatory.ecephys.file_io.continuous_file import \
    ContinuousFile
from allensdk.core import DataObject, JsonReadableInterface


class LFP(DataObject, JsonReadableInterface):
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

    @classmethod
    def from_json(
            cls,
            probe_meta: dict
    ) -> "LFP":
        """

        Parameters
        ----------
        probe_meta:
            Probe metadata as a dict

        Returns
        -------
        `LFP` instance
        """
        lfp_meta = probe_meta['lfp']
        lfp_channels = np.load(lfp_meta['input_channels_path'],
                               allow_pickle=False)

        # TODO 2nd dimension in lfp_data doesn't match number of lfp_channels

        lfp_data, lfp_timestamps = ContinuousFile(
            data_path=lfp_meta['input_data_path'],
            timestamps_path=lfp_meta['input_timestamps_path'],
            total_num_channels=len(lfp_channels)
        ).load(memmap=False)

        lfp_data = lfp_data.astype(np.float32)
        lfp_data = lfp_data * probe_meta["amplitude_scale_factor"]

        sampling_rate = (
                probe_meta['lfp_sampling_rate'] /
                probe_meta['temporal_subsampling_factor'])

        return cls(
            data=lfp_data,
            timestamps=lfp_timestamps,
            channels=lfp_channels,
            sampling_rate=sampling_rate
        )

    def to_dataarray(self) -> DataArray:
        return DataArray(
            name="LFP",
            data=self._data,
            dims=['time', 'channel'],
            coords=[self._timestamps, self._channels]
        )

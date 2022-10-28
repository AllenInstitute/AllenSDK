import h5py
import numpy as np

from allensdk.core import JsonReadableInterface, DataObject


class CurrentSourceDensity(DataObject, JsonReadableInterface):
    """Current Source Density"""
    def __init__(
            self,
            data: np.ndarray,
            timestamps: np.ndarray,
            interpolated_channel_locations: np.ndarray
    ):
        """

        Parameters
        ----------
        data:
            CSD data. channels X time samples
        timestamps:
            CSD timestamps
        interpolated_channel_locations:
            Array of interpolated channel indices for CSD
        """
        super().__init__(
            name='current_source_density',
            value=None,
            is_value_self=True
        )
        self._data = data
        self._timestamps = timestamps
        self._interpolated_channel_locations = interpolated_channel_locations

    @property
    def data(self) -> np.ndarray:
        """CSD data. channels X time samples"""
        return self._data

    @property
    def timestamps(self) -> np.ndarray:
        return self._timestamps

    @property
    def channel_locations(self) -> np.ndarray:
        """Array of interpolated channel indices for CSD"""
        return self._interpolated_channel_locations

    @classmethod
    def from_json(cls, probe_meta: dict) -> "CurrentSourceDensity":
        with h5py.File(probe_meta['csd_path'], "r") as csd_file:
            return CurrentSourceDensity(
                data=csd_file["current_source_density"][:],
                timestamps=csd_file["timestamps"][:],
                interpolated_channel_locations=csd_file["csd_locations"][:]
            )

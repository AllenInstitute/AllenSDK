from typing import Optional
import numpy as np

from allensdk.core import DataObject


class Channel(DataObject):
    """Probe channel"""
    def __init__(
            self,
            id: int,
            probe_id: int,
            valid_data: bool,
            local_index: int,
            probe_vertical_position: int,
            probe_horizontal_position: int,
            manual_structure_acronym: str = '',
            manual_structure_id: Optional[int] = None,
            anterior_posterior_ccf_coordinate: Optional[float] = None,
            dorsal_ventral_ccf_coordinate: Optional[float] = None,
            left_right_ccf_coordinate: Optional[float] = None,
            impedance: float = np.nan,
            filtering: str = 'AP band: 500 Hz high-pass; '
                             'LFP band: 1000 Hz low-pass'
    ):
        super().__init__(name='channel', value=self)
        self._id = id
        self._probe_id = probe_id
        self._valid_data = valid_data
        self._local_index = local_index
        self._probe_vertical_position = probe_vertical_position
        self._probe_horizontal_position = probe_horizontal_position
        self._manual_structure_acronym = manual_structure_acronym
        self._manual_structure_id = manual_structure_id
        self._anterior_posterior_ccf_coordinate = \
            anterior_posterior_ccf_coordinate
        self._dorsal_ventral_ccf_coordinate = dorsal_ventral_ccf_coordinate
        self._left_right_ccf_coordinate = left_right_ccf_coordinate
        self._impedance = impedance
        self._filtering = filtering

    @property
    def id(self) -> int:
        return self._id

    @property
    def probe_id(self) -> int:
        return self._probe_id

    @property
    def valid_data(self) -> bool:
        return self._valid_data

    @property
    def local_index(self) -> int:
        return self._local_index

    @property
    def probe_vertical_position(self) -> int:
        return self._probe_vertical_position

    @property
    def probe_horizontal_position(self) -> int:
        return self._probe_horizontal_position

    @property
    def manual_structure_acronym(self) -> str:
        return self._manual_structure_acronym

    @property
    def manual_structure_id(self) -> Optional[int]:
        return self._manual_structure_id

    @property
    def anterior_posterior_ccf_coordinate(self) -> Optional[float]:
        return self._anterior_posterior_ccf_coordinate

    @property
    def dorsal_ventral_ccf_coordinate(self) -> Optional[float]:
        return self._dorsal_ventral_ccf_coordinate

    @property
    def left_right_ccf_coordinate(self) -> Optional[float]:
        return self._left_right_ccf_coordinate

    @property
    def impedance(self) -> float:
        return self._impedance

    @property
    def filtering(self) -> str:
        return self._filtering

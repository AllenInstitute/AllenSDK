import logging
from typing import Optional

import numpy as np
import pandas as pd
from pynwb import NWBFile

from allensdk.brain_observatory.ecephys._channels import Channels
from allensdk.brain_observatory.ecephys._units import Units
from allensdk.brain_observatory.ecephys.nwb_util import add_probe_to_nwbfile, \
    add_ecephys_electrodes
from allensdk.core import DataObject, JsonReadableInterface, \
    NwbWritableInterface, NwbReadableInterface


class _Lfp:
    def __init__(self, data: np.ndarray, sampling_rate: float = 2500.0):
        """

        Parameters
        ----------
        data: LFP data
        sampling_rate: sampling rate of LFP data
        """
        self._data = data
        self._sampling_rate = sampling_rate

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def sampling_rate(self) -> float:
        return self._sampling_rate


class Probe(DataObject, JsonReadableInterface, NwbWritableInterface,
            NwbReadableInterface):
    """A single probe"""
    def __init__(
            self,
            id: int,
            name: str,
            channels: Channels,
            units: Units,
            sampling_rate: float = 30000.0,
            lfp: Optional[_Lfp] = None,
            location: str = 'See electrode locations',
            temporal_subsampling_factor: Optional[float] = 2.0
    ):
        """

        Parameters
        ----------
        id: probe id
        name: probe name
        channels: probe channels
        units: units detected by probe
        sampling_rate: probe sampling rate
        lfp: probe LFP
        location: probe location
        temporal_subsampling_factor: subsampling factor applied to lfp data
            (across time)
        """
        self._id = id
        self._name = name
        self._channels = channels
        self._units = units
        self._sampling_rate = sampling_rate
        self._lfp = lfp
        self._location = location
        self._temporal_subsampling_factor = temporal_subsampling_factor
        super().__init__(name=name,
                         value=None,
                         is_value_self=True)

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def channels(self) -> Channels:
        return self._channels

    @property
    def units(self) -> Units:
        return self._units

    @property
    def sampling_rate(self) -> float:
        return self._sampling_rate

    @property
    def lfp(self) -> Optional[_Lfp]:
        return self._lfp

    @property
    def location(self) -> str:
        return self._location

    @property
    def temporal_subsampling_factor(self) -> Optional[float]:
        return self._temporal_subsampling_factor

    @property
    def units_table(self) -> pd.DataFrame:
        df = pd.DataFrame(
                [unit.to_dict()['unit'] for unit in self.units.value])
        df = df.fillna(np.nan)
        return df

    @classmethod
    def from_json(cls, probe: dict) -> "Probe":
        if probe['lfp'] is not None:
            lfp = _Lfp(data=probe['lfp'],
                       sampling_rate=probe['lfp_sampling_rate'])
        else:
            lfp = None

        channels = Channels.from_json(channels=probe['channels'])
        units = Units.from_json(probe=probe)

        return Probe(
            id=probe['id'],
            name=probe['name'],
            channels=channels,
            units=units,
            sampling_rate=probe['sampling_rate'],
            lfp=lfp,
            temporal_subsampling_factor=probe['temporal_subsampling_factor']
        )

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile, **kwargs) -> "Probe":
        if 'probe_name' not in kwargs:
            raise ValueError('Must pass probe_name')
        probe = nwbfile.electrode_groups[kwargs['probe_name']]
        channels = Channels.from_nwb(nwbfile=nwbfile, probe_id=probe.probe_id)
        units = Units.from_nwb(nwbfile=nwbfile, probe_id=probe.probe_id)
        return Probe(
            id=probe.probe_id,
            name=probe.name,
            location=probe.location,
            sampling_rate=probe.device.sampling_rate,
            channels=channels,
            units=units
        )

    def to_nwb(self, nwbfile: NWBFile) -> NWBFile:
        logging.info(f'found probe {self._id} with name {self._name}')

        if self._temporal_subsampling_factor is not None and \
                self._lfp is not None:
            lfp_sampling_rate = self._lfp.sampling_rate / \
                                self._temporal_subsampling_factor
        else:
            lfp_sampling_rate = np.nan

        nwbfile, probe_nwb_device, probe_nwb_electrode_group = \
            add_probe_to_nwbfile(
                nwbfile,
                probe_id=self._id,
                name=self._name,
                sampling_rate=self._sampling_rate,
                lfp_sampling_rate=lfp_sampling_rate,
                has_lfp_data=self._lfp is not None
            )

        channels = [c.to_dict()['channel'] for c in self._channels.value]
        add_ecephys_electrodes(nwbfile,
                               channels,
                               probe_nwb_electrode_group)
        return nwbfile

    def to_dict(self) -> dict:
        return {
            'id': self._id,
            'name': self._name,
            'location': self._location,
            'sampling_rate': self._sampling_rate,
            'lfp_sampling_rate':
                self._lfp.sampling_rate if self._lfp is not None else None,
            'has_lfp_data': self._lfp is not None
        }

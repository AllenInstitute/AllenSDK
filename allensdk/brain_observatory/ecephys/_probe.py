import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import pynwb
from hdmf.backends.hdf5 import H5DataIO
from pynwb import NWBFile

from allensdk.brain_observatory.ecephys._behavior_ecephys_metadata import \
    BehaviorEcephysMetadata
from allensdk.brain_observatory.ecephys._channels import Channels
from allensdk.brain_observatory.ecephys._units import Units
from allensdk.brain_observatory.ecephys.lfp import LFP
from allensdk.brain_observatory.ecephys.nwb_util import add_probe_to_nwbfile, \
    add_ecephys_electrodes
from allensdk.core import DataObject, JsonReadableInterface, \
    NwbWritableInterface, NwbReadableInterface


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
            lfp: Optional[LFP] = None,
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
    def lfp(self) -> Optional[LFP]:
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
        channels = Channels.from_json(channels=probe['channels'])
        units = Units.from_json(probe=probe)

        if probe['lfp'] is not None:
            lfp = LFP.from_json(
                probe_meta=probe,
                num_probe_channels=len(channels.value)
            )
        else:
            lfp = None

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

    def to_nwb(
            self,
            nwbfile: NWBFile,
            lfp_output_path: Optional[Union[str, Path]] = None
    ) -> NWBFile:
        nwbfile = self._add_probe_to_nwb(nwbfile=nwbfile)

        if self._lfp is not None:
            if lfp_output_path is None:
                raise ValueError(
                    'Need output path to save LFP data to separate NWB file')
            self._write_lfp_to_nwb(
                output_path=lfp_output_path,
                session_id=nwbfile.session_id,
                session_metadata=BehaviorEcephysMetadata.from_nwb(
                    nwbfile=nwbfile),
                session_start_time=nwbfile.session_start_time
            )
        return nwbfile

    def _add_probe_to_nwb(
            self, nwbfile: NWBFile,
            add_only_lfp_channels: bool = False
    ):
        logging.info(f'found probe {self._id} with name {self._name}')

        nwbfile, probe_nwb_device, probe_nwb_electrode_group = \
            add_probe_to_nwbfile(
                nwbfile,
                probe_id=self._id,
                name=self._name,
                sampling_rate=self._sampling_rate,
                lfp_sampling_rate=(
                    self._lfp.sampling_rate if self._lfp is not None
                    else np.nan),
                has_lfp_data=self._lfp is not None
            )

        channels = [c.to_dict()['channel'] for c in self._channels.value]

        if self._lfp is not None and add_only_lfp_channels:
            channels = [c for i, c in enumerate(channels)
                        if i in self._lfp.channels]
        add_ecephys_electrodes(nwbfile,
                               channels,
                               probe_nwb_electrode_group)
        return nwbfile

    def _write_lfp_to_nwb(
            self,
            output_path: Union[Path, str],
            session_id: str,
            session_start_time: datetime,
            session_metadata: BehaviorEcephysMetadata
    ):
        logging.info(f'writing lfp file for probe {self._id}')

        output_path = Path(output_path)

        nwbfile = pynwb.NWBFile(
            session_description='LFP data and associated info for one probe',
            identifier=f"{self._id}",
            session_id=f"{session_id}",
            session_start_time=session_start_time,
            institution="Allen Institute for Brain Science"
        )
        nwbfile = session_metadata.to_nwb(nwbfile=nwbfile)

        nwbfile = self._add_probe_to_nwb(
            nwbfile=nwbfile,
            add_only_lfp_channels=True
        )
        lfp_nwb = pynwb.ecephys.LFP(name=f"probe_{self._id}_lfp")

        electrode_table_region = nwbfile.create_electrode_table_region(
            region=np.arange(len(nwbfile.electrodes)).tolist(),
            name='electrodes',
            description=f"lfp channels on probe {self._id}"
        )

        nwbfile.add_acquisition(lfp_nwb.create_electrical_series(
            name=f"probe_{self._id}_lfp_data",
            data=self._lfp.data,
            timestamps=self._lfp.timestamps,
            electrodes=electrode_table_region
        ))
        nwbfile.add_acquisition(lfp_nwb)

        # TODO CSD data

        os.makedirs(output_path.parent, exist_ok=True)

        with pynwb.NWBHDF5IO(output_path, 'w') as f:
            logging.info(f"writing lfp file to {output_path}")
            f.write(nwbfile, cache_spec=True)

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

import typing
from typing import Mapping, TypeVar, Union, Callable, Tuple

from dataclasses import dataclass # backport of 3.7+ dataclasses -> 3.6
import pandas as pd
import numpy as np
import pynwb


T = TypeVar('T')
MaybeCallable = Union[T, Callable[[], T]]


@dataclass(frozen=True)
class EcephysDataset:
    session_id: int
    stimulus_epochs: MaybeCallable[pd.DataFrame]
    probes: MaybeCallable[pd.DataFrame]
    channels: MaybeCallable[pd.DataFrame]
    units: MaybeCallable[pd.DataFrame]
    spike_times: MaybeCallable[Mapping[int, np.ndarray]]
    mean_waveforms: MaybeCallable[Mapping[int, np.ndarray]]


def eager_read_nwbfile_units(nwbfile: pynwb.file.NWBFile) -> Tuple[pd.DataFrame, Mapping[int, np.ndarray], Mapping[int, np.ndarray]]:
    units_table = nwbfile.units.to_dataframe()
    
    spike_times = units_table['spike_times'].to_dict()
    mean_waveforms = units_table['waveform_mean'].to_dict()

    units_table.drop(columns=['spike_times', 'waveform_mean'], inplace=True)
    return units_table, spike_times, mean_waveforms


def eager_read_nwbfile_channels(nwbfile: pynwb.file.NWBFile) -> pd.DataFrame:
    channels = nwbfile.electrodes.to_dataframe()
    channels.drop(columns='group', inplace=True)
    return channels


def eager_read_nwb_probes(nwbfile: pynwb.file.NWBFile) -> pd.DataFrame:
    probes = []
    for k, v in nwbfile.electrode_groups.items():
        probes.append({'id': int(k), 'name': v.description})
    return pd.DataFrame(probes)


def eager_read_nwb_stimulus_epochs(nwbfile: pynwb.file.NWBFile) -> pd.DataFrame:
    stimulus_table = nwbfile.epochs.to_dataframe()
    stimulus_table.drop(columns=['tags', 'timeseries'], inplace=True)
    return stimulus_table


def eager_read_dataset_from_nwbfile(nwbfile: Union[str, pynwb.file.NWBFile]) -> EcephysDataset:

    owns_file = False
    if isinstance(nwbfile, str):
        io = pynwb.NWBHDF5IO(nwbfile, mode='r')
        nwbfile = io.read()
        owns_file = True

    units, spike_times, mean_waveforms = eager_read_nwbfile_units(nwbfile)
    dataset: EcephysDataset = EcephysDataset(
        session_id=int(nwbfile.identifier),
        stimulus_epochs=eager_read_nwb_stimulus_epochs(nwbfile),
        probes=eager_read_nwb_probes(nwbfile),
        channels=eager_read_nwbfile_channels(nwbfile),
        units=units,
        spike_times=spike_times,
        mean_waveforms=mean_waveforms
    )

    if owns_file:
        io.close()

    return dataset
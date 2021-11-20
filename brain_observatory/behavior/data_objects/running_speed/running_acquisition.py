
import json
from typing import Optional

from cachetools import cached, LRUCache
from cachetools.keys import hashkey

import pandas as pd

from pynwb import NWBFile, ProcessingModule
from pynwb.base import TimeSeries

from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    LimsReadableInterface, NwbReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base \
    .writable_interfaces import \
    JsonWritableInterface, NwbWritableInterface
from allensdk.internal.api import PostgresQueryMixin
from allensdk.brain_observatory.behavior.data_objects import (
    DataObject, StimulusTimestamps
)
from allensdk.brain_observatory.behavior.data_files import (
    StimulusFile
)
from allensdk.brain_observatory.behavior.data_objects.running_speed.running_processing import (  # noqa: E501
    get_running_df
)


def from_json_cache_key(
    cls, dict_repr: dict
):
    return hashkey(json.dumps(dict_repr))


def from_lims_cache_key(
    cls, db,
    behavior_session_id: int, ophys_experiment_id: Optional[int] = None
):
    return hashkey(
        behavior_session_id, ophys_experiment_id
    )


class RunningAcquisition(DataObject, LimsReadableInterface,
                         NwbReadableInterface, NwbWritableInterface,
                         JsonWritableInterface):
    """A DataObject which contains properties and methods to load, process,
    and represent running acquisition data.

    Running aquisition data is represented as:

    Pandas Dataframe with an index of timestamps and the following columns:
        "dx": Angular change, computed during data collection
        "v_sig": Voltage signal from the encoder
        "v_in": The theoretical maximum voltage that the encoder
            will reach prior to "wrapping". This should
            theoretically be 5V (after crossing 5V goes to 0V, or
            vice versa). In practice the encoder does not always
            reach this value before wrapping, which can cause
            transient spikes in speed at the voltage "wraps".
    """

    def __init__(
        self,
        running_acquisition: pd.DataFrame,
        stimulus_file: Optional[StimulusFile] = None,
        stimulus_timestamps: Optional[StimulusTimestamps] = None,
    ):
        super().__init__(name="running_acquisition", value=running_acquisition)
        self._stimulus_file = stimulus_file
        self._stimulus_timestamps = stimulus_timestamps

    @classmethod
    @cached(cache=LRUCache(maxsize=10), key=from_json_cache_key)
    def from_json(
        cls,
        dict_repr: dict,
    ) -> "RunningAcquisition":
        stimulus_file = StimulusFile.from_json(dict_repr)
        stimulus_timestamps = StimulusTimestamps.from_json(dict_repr)
        running_acq_df = get_running_df(
            data=stimulus_file.data, time=stimulus_timestamps.value,
        )
        running_acq_df.drop("speed", axis=1, inplace=True)

        return cls(
            running_acquisition=running_acq_df,
            stimulus_file=stimulus_file,
            stimulus_timestamps=stimulus_timestamps,
        )

    def to_json(self) -> dict:
        """[summary]

        Returns
        -------
        dict
            [description]

        Raises
        ------
        RuntimeError
            [description]
        """
        if self._stimulus_file is None or self._stimulus_timestamps is None:
            raise RuntimeError(
                "RunningAcquisition DataObject lacks information about the "
                "StimulusFile or StimulusTimestamps. This is likely due to "
                "instantiating from NWB which prevents to_json() functionality"
            )
        output_dict = dict()
        output_dict.update(self._stimulus_file.to_json())
        output_dict.update(self._stimulus_timestamps.to_json())
        return output_dict

    @classmethod
    @cached(cache=LRUCache(maxsize=10), key=from_lims_cache_key)
    def from_lims(
        cls,
        db: PostgresQueryMixin,
        behavior_session_id: int,
        ophys_experiment_id: Optional[int] = None,
    ) -> "RunningAcquisition":

        stimulus_file = StimulusFile.from_lims(db, behavior_session_id)
        stimulus_timestamps = StimulusTimestamps.from_stimulus_file(
            stimulus_file=stimulus_file
        )
        running_acq_df = get_running_df(
            data=stimulus_file.data, time=stimulus_timestamps.value,
        )
        running_acq_df.drop("speed", axis=1, inplace=True)

        return cls(
            running_acquisition=running_acq_df,
            stimulus_file=stimulus_file,
            stimulus_timestamps=stimulus_timestamps,
        )

    @classmethod
    def from_nwb(
        cls,
        nwbfile: NWBFile
    ) -> "RunningAcquisition":
        running_module = nwbfile.modules['running']
        dx_interface = running_module.get_data_interface('dx')

        dx = dx_interface.data
        v_in = nwbfile.get_acquisition('v_in').data
        v_sig = nwbfile.get_acquisition('v_sig').data
        timestamps = dx_interface.timestamps[:]

        running_acq_df = pd.DataFrame(
            {
                'dx': dx,
                'v_in': v_in,
                'v_sig': v_sig
            },
            index=pd.Index(timestamps, name='timestamps')
        )
        return cls(running_acquisition=running_acq_df)

    def to_nwb(self, nwbfile: NWBFile) -> NWBFile:
        running_acquisition_df: pd.DataFrame = self.value

        running_dx_series = TimeSeries(
            name='dx',
            data=running_acquisition_df['dx'].values,
            timestamps=running_acquisition_df.index.values,
            unit='cm',
            description=(
                'Running wheel angular change, computed during data collection'
            )
        )
        v_sig = TimeSeries(
            name='v_sig',
            data=running_acquisition_df['v_sig'].values,
            timestamps=running_acquisition_df.index.values,
            unit='V',
            description='Voltage signal from the running wheel encoder'
        )
        v_in = TimeSeries(
            name='v_in',
            data=running_acquisition_df['v_in'].values,
            timestamps=running_acquisition_df.index.values,
            unit='V',
            description=(
                'The theoretical maximum voltage that the running wheel '
                'encoder will reach prior to "wrapping". This should '
                'theoretically be 5V (after crossing 5V goes to 0V, or '
                'vice versa). In practice the encoder does not always '
                'reach this value before wrapping, which can cause '
                'transient spikes in speed at the voltage "wraps".')
        )

        if 'running' in nwbfile.processing:
            running_mod = nwbfile.processing['running']
        else:
            running_mod = ProcessingModule('running',
                                           'Running speed processing module')
            nwbfile.add_processing_module(running_mod)

        running_mod.add_data_interface(running_dx_series)
        nwbfile.add_acquisition(v_sig)
        nwbfile.add_acquisition(v_in)

        return nwbfile

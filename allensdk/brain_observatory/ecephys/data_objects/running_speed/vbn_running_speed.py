from typing import Optional

from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    JsonReadableInterface, LimsReadableInterface, NwbReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base \
    .writable_interfaces import \
    JsonWritableInterface, NwbWritableInterface
from allensdk.brain_observatory.behavior.data_objects import (
    DataObject
)

import pandas as pd

from allensdk.brain_observatory.behavior.data_objects.\
    running_speed.running_speed import RunningSpeedNWBMixin

from allensdk.brain_observatory.ecephys.data_objects.\
    running_speed.multi_stim_running_processing import (
        multi_stim_running_df_from_raw_data)


def _get_multi_stim_running_df(
        sync_path: str,
        behavior_pkl_path: str,
        mapping_pkl_path: str,
        replay_pkl_path: str,
        use_lowpass_filter: bool,
        zscore_threshold: float) -> pd.DataFrame:
    """
    Parameters
    ----------
    sync_path: str
        The path to the sync file
    behavior_pkl_path: str
        path to behavior pickle file
    mapping_pkl_path: str
        path to mapping pickle file
    replay_pkl_path: str
        path to replay pickle file
    use_lowpass_filter: bool
        whther or not to apply a low pass filter to the
        running speed results
    zscore_threshold: float
        The threshold to use for removing outlier
        running speeds which might be noise and not true signal

    Returns
    -------
    A dataframe with columns 'timestamps' and 'speed'
    """

    (velocity_data,
     _) = multi_stim_running_df_from_raw_data(
             sync_path=sync_path,
             behavior_pkl_path=behavior_pkl_path,
             mapping_pkl_path=mapping_pkl_path,
             replay_pkl_path=replay_pkl_path,
             use_lowpass_filter=use_lowpass_filter,
             zscore_threshold=zscore_threshold,
             behavior_start_frame=0)
    return pd.DataFrame(data={'timestamps': velocity_data.frame_time.values,
                              'speed': velocity_data.velocity.values})


class VBNRunningSpeed(RunningSpeedNWBMixin,
                      DataObject, LimsReadableInterface, NwbReadableInterface,
                      NwbWritableInterface, JsonReadableInterface,
                      JsonWritableInterface):
    """A DataObject which contains properties and methods to load, process,
    and represent running speed data.

    Running speed data is represented as:

    Pandas Dataframe with the following columns:
        "timestamps": Timestamps (in s) for calculated speed values
        "speed": Computed running speed in cm/s

    The difference betwen this class and the RunningSpeed class implemented
    in behavior/data_objects/running_speed is that this object concatenates
    the running speed data from multiple stimulus pickle files
    """

    def __init__(
            self,
            running_speed: pd.DataFrame,
            sync_path: Optional[str] = None,
            behavior_pkl_path: Optional[str] = None,
            replay_pkl_path: Optional[str] = None,
            mapping_pkl_path: Optional[str] = None,
            use_lowpass_filter: bool = True,
            zscore_threshold: float = 10.0,
            filtered: bool = False):

        super().__init__(name='running_speed', value=running_speed)
        self._sync_path = sync_path
        self._behavior_pkl_path = behavior_pkl_path
        self._replay_pkl_path = replay_pkl_path
        self._mapping_pkl_path = mapping_pkl_path
        self._use_lowpass_filter = use_lowpass_filter
        self._zscore_threshold = zscore_threshold
        self._filtered = filtered

    @classmethod
    def from_json(
            cls,
            dict_repr: dict) -> "VBNRunningSpeed":

        behavior_pkl_path = dict_repr['behavior_pkl_path']
        replay_pkl_path = dict_repr['replay_pkl_path']
        mapping_pkl_path = dict_repr['mapping_pkl_path']
        sync_path = dict_repr['sync_h5_path']
        if 'use_lowpass_filter' in dict_repr:
            use_lowpass_filter = dict_repr['use_lowpass_filter']
        else:
            use_lowpass_filter = True

        if 'zscore_threshold' in dict_repr:
            zscore_threshold = dict_repr['zscore_threshold']
        else:
            zscore_threshold = 10.0

        df = _get_multi_stim_running_df(
                sync_path=sync_path,
                behavior_pkl_path=behavior_pkl_path,
                replay_pkl_path=replay_pkl_path,
                mapping_pkl_path=mapping_pkl_path,
                use_lowpass_filter=use_lowpass_filter,
                zscore_threshold=zscore_threshold)

        return cls(
                running_speed=df,
                sync_path=sync_path,
                behavior_pkl_path=behavior_pkl_path,
                mapping_pkl_path=mapping_pkl_path,
                replay_pkl_path=replay_pkl_path,
                use_lowpass_filter=use_lowpass_filter,
                zscore_threshold=zscore_threshold)

    def to_json(self) -> dict:
        output = dict()
        msg = ""
        for value, key in zip((self._sync_path,
                               self._behavior_pkl_path,
                               self._mapping_pkl_path,
                               self._replay_pkl_path),
                              ('sync_h5_path',
                               'behavior_pkl_path',
                               'mapping_pkl_path',
                               'replay_pkl_path')):
            if value is None:
                msg += (f"{key} is None; must be specified "
                        "for VBNRunningSpeed.to_json")

            output[key] = value

        if len(msg) > 0:
            raise RuntimeError(msg)

        output["use_lowpass_filter"] = self._use_lowpass_filter
        output["zscore_threshold"] = self._zscore_threshold
        output["filtered"] = self._filtered
        return output

    @classmethod
    def from_lims(cls):
        raise NotImplementedError()

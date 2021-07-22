import warnings
from typing import Optional

import pandas as pd
from pynwb import NWBFile, TimeSeries, ProcessingModule

from allensdk.brain_observatory.behavior.data_files import StimulusFile
from allensdk.brain_observatory.behavior.data_objects import DataObject, \
    StimulusTimestamps
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    StimulusFileReadableInterface, NwbReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base\
    .writable_interfaces import \
    NwbWritableInterface


class Rewards(DataObject, StimulusFileReadableInterface, NwbReadableInterface,
              NwbWritableInterface):
    def __init__(self, rewards: pd.DataFrame):
        super().__init__(name='rewards', value=rewards)

    @classmethod
    def from_stimulus_file(
            cls, stimulus_file: StimulusFile,
            stimulus_timestamps: StimulusTimestamps) -> "Rewards":
        """Get reward data from pkl file, based on timestamps
        (not sync file).
        """
        data = stimulus_file.data

        trial_df = pd.DataFrame(data["items"]["behavior"]["trial_log"])
        rewards_dict = {"volume": [], "timestamps": [], "autorewarded": []}
        for idx, trial in trial_df.iterrows():
            rewards = trial["rewards"]
            # as i write this there can only ever be one reward per trial
            if rewards:
                rewards_dict["volume"].append(rewards[0][0])
                rewards_dict["timestamps"].append(
                    stimulus_timestamps.value[rewards[0][2]])
                auto_rwrd = trial["trial_params"]["auto_reward"]
                rewards_dict["autorewarded"].append(auto_rwrd)

        df = pd.DataFrame(rewards_dict)
        return cls(rewards=df)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> Optional["Rewards"]:
        if 'rewards' in nwbfile.processing:
            rewards = nwbfile.processing['rewards']
            time = rewards.get_data_interface('autorewarded').timestamps[:]
            autorewarded = rewards.get_data_interface('autorewarded').data[:]
            volume = rewards.get_data_interface('volume').data[:]
            df = pd.DataFrame({
                'volume': volume, 'timestamps': time,
                'autorewarded': autorewarded})
        else:
            warnings.warn("This session "
                          f"'{int(nwbfile.identifier)}' has no rewards data.")
            return None
        return cls(rewards=df)

    def to_nwb(self, nwbfile: NWBFile) -> NWBFile:
        reward_volume_ts = TimeSeries(
            name='volume',
            data=self.value['volume'].values,
            timestamps=self.value['timestamps'].values,
            unit='mL'
        )

        autorewarded_ts = TimeSeries(
            name='autorewarded',
            data=self.value['autorewarded'].values,
            timestamps=reward_volume_ts.timestamps,
            unit='mL'
        )

        rewards_mod = ProcessingModule('rewards',
                                       'Licking behavior processing module')
        rewards_mod.add_data_interface(reward_volume_ts)
        rewards_mod.add_data_interface(autorewarded_ts)
        nwbfile.add_processing_module(rewards_mod)

        return nwbfile

import warnings
from typing import Optional, List

import pandas as pd
from pynwb import NWBFile

from allensdk.brain_observatory import dict_to_indexed_array
from allensdk.brain_observatory.behavior.data_files import StimulusFile, \
    SyncFile
from allensdk.brain_observatory.behavior.data_objects import DataObject, \
    StimulusTimestamps
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    StimulusFileReadableInterface, NwbReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base\
    .writable_interfaces import \
    NwbWritableInterface
from allensdk.brain_observatory.behavior.data_objects.licks import Licks
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .behavior_metadata.equipment import \
    Equipment
from allensdk.brain_observatory.behavior.data_objects.rewards import Rewards
from allensdk.brain_observatory.behavior.data_objects.trials.trial import Trial
from allensdk.internal.brain_observatory.time_sync import OphysTimeAligner


class TrialTable(DataObject, StimulusFileReadableInterface,
                 NwbReadableInterface, NwbWritableInterface):
    def __init__(self, trials: pd.DataFrame):
        super().__init__(name='trials', value=trials)

    def to_nwb(self, nwbfile: NWBFile) -> NWBFile:
        trials = self.value
        order = list(trials.index)
        for _, row in trials[['start_time', 'stop_time']].iterrows():
            row_dict = row.to_dict()
            nwbfile.add_trial(**row_dict)

        for c in trials.columns:
            if c in ['start_time', 'stop_time']:
                continue
            index, data = dict_to_indexed_array(trials[c].to_dict(), order)
            if data.dtype == '<U1':  # data type is composed of unicode
                # characters
                data = trials[c].tolist()
            if not len(data) == len(order):
                if len(data) == 0:
                    data = ['']
                nwbfile.add_trial_column(
                    name=c,
                    description='NOT IMPLEMENTED: %s' % c,
                    data=data,
                    index=index)
            else:
                nwbfile.add_trial_column(
                    name=c,
                    description='NOT IMPLEMENTED: %s' % c,
                    data=data)
        return nwbfile

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "TrialTable":
        trials = nwbfile.trials.to_dataframe()
        if 'lick_events' in trials.columns:
            trials.drop('lick_events', inplace=True, axis=1)
        trials.index = trials.index.rename('trials_id')
        return TrialTable(trials=trials)

    @classmethod
    def from_stimulus_file(cls, stimulus_file: StimulusFile,
                           stimulus_timestamps: StimulusTimestamps,
                           licks: Licks,
                           rewards: Rewards,
                           monitor_delay: Optional[float] = None,
                           sync_file: Optional[SyncFile] = None,
                           equipment: Optional[Equipment] = None
                           ) -> "TrialTable":
        if monitor_delay is None:
            if sync_file is None or equipment is None:
                raise ValueError('Need sync file and equipment in order to '
                                 'calculate monitor delay')
            monitor_delay = cls._calculate_monitor_delay(sync_file=sync_file,
                                                         equipment=equipment)
        bsf = stimulus_file.data

        stimuli = bsf["items"]["behavior"]["stimuli"]
        trial_log = bsf["items"]["behavior"]["trial_log"]

        trial_bounds = cls._get_trial_bounds(trial_log=trial_log)

        all_trial_data = [None] * len(trial_log)

        for idx, trial in enumerate(trial_log):
            trial_start, trial_end = trial_bounds[idx]
            t = Trial(trial=trial, start=trial_start, end=trial_end,
                      behavior_stimulus_file=stimulus_file,
                      index=idx,
                      monitor_delay=monitor_delay,
                      stimulus_timestamps=stimulus_timestamps,
                      licks=licks, rewards=rewards,
                      stimuli=stimuli
                      )
            all_trial_data[idx] = t.data

        trials = pd.DataFrame(all_trial_data).set_index('trial')
        trials.index = trials.index.rename('trials_id')
        del trials["sham_change"]

        return TrialTable(trials=trials)

    @staticmethod
    def _calculate_monitor_delay(sync_file: SyncFile,
                                 equipment: Equipment) -> float:
        aligner = OphysTimeAligner(sync_file=sync_file.filepath)

        try:
            delay = aligner.monitor_delay
        except ValueError as ee:
            equipment_name = equipment.value

            warning_msg = 'Monitory delay calculation failed '
            warning_msg += 'with ValueError\n'
            warning_msg += f'    "{ee}"'
            warning_msg += '\nlooking monitor delay up from table '
            warning_msg += f'for rig: {equipment_name} '

            # see
            # https://github.com/AllenInstitute/AllenSDK/issues/1318
            # https://github.com/AllenInstitute/AllenSDK/issues/1916
            delay_lookup = {'CAM2P.1': 0.020842,
                            'CAM2P.2': 0.037566,
                            'CAM2P.3': 0.021390,
                            'CAM2P.4': 0.021102,
                            'CAM2P.5': 0.021192,
                            'MESO.1': 0.03613}

            if equipment_name not in delay_lookup:
                msg = warning_msg
                msg += f'\nequipment_name {equipment_name} not in lookup table'
                raise RuntimeError(msg)
            delay = delay_lookup[equipment_name]
            warning_msg += f'\ndelay: {delay} seconds'
            warnings.warn(warning_msg)

        return delay

    @staticmethod
    def _get_trial_bounds(trial_log: List) -> List:
        """
        Adjust trial boundaries from a trial_log so that there is no dead time
        between trials.

        Parameters
        ----------
        trial_log: list
            The trial_log read in from the well known behavior stimulus
            pickle file

        Returns
        -------
        list
            Each element in the list is a tuple of the form
            (start_frame, end_frame) so that the ith element
            of the list gives the start and end frames of
            the ith trial. The endframe of the last trial will
            be -1, indicating that it should map to the last
            timestamp in the session
        """
        start_frames = []

        for trial in trial_log:
            start_f = None
            for event in trial['events']:
                if event[0] == 'trial_start':
                    start_f = event[-1]
                    break
            if start_f is None:
                msg = "Could not find a 'trial_start' event "
                msg += "for all trials in the trial log\n"
                msg += f"{trial}"
                raise ValueError(msg)

            if len(start_frames) > 0 and start_f < start_frames[-1]:
                msg = "'trial_start' frames in trial log "
                msg += "are not in ascending order"
                msg += f"\ntrial_log: {trial_log}"
                raise ValueError(msg)

            start_frames.append(start_f)

        end_frames = [idx for idx in start_frames[1:] + [-1]]
        return list([(s, e) for s, e in zip(start_frames, end_frames)])

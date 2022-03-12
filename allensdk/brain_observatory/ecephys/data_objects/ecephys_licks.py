import logging
from typing import Optional, Union, Tuple
import numpy as np

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

from allensdk.brain_observatory.behavior.data_files import SyncFile

from allensdk.brain_observatory.ecephys.data_files \
    .ecephys_stimulus_file import \
    EcephysStimulusFile

from allensdk.brain_observatory.sync_dataset import Dataset as SyncDataset


class EcephysLicks(DataObject, StimulusFileReadableInterface, NwbReadableInterface,
            NwbWritableInterface):
    _logger = logging.getLogger(__name__)

    def __init__(self, licks: pd.DataFrame):
        """
        :param licks
            dataframe containing the following columns:
                - timestamps: float
                    stimulus timestamps in which there was a lick
                - frame: int
                    frame number in which there was a lick
        """
        super().__init__(name='licks', value=licks)


    # @classmethod
    # def from_sync_file(cls, sync_file: SyncFile, behavior_stimulus_file: EcephysStimulusFile)  -> "Licks":
    #     sync_data = SyncFile.load_data(sync_file.filepath)
    #     lick_times = sync_data['lick_times']

    #     self._stim_data = behavior_stimulus_file.load_data(filepath=self._filepath)

    #     #TODO - create a new class for this
    #     # vsyncs = sync_data['ophys_frames']
    #     # vsyncs_lookup = {}

    #     # frame = 0
    #     # for vsync in vsyncs:
    #     #     vsyncs_lookup[vsync] = frame
    #     #     frame+=1

    #     # print('vsyncs' , vsyncs)

    #     # lick_frames = []
    #     # for lick_time in lick_times:
    #     #     lick_frames.append(vsyncs_lookup[lick_time])

    #     #TODO update this
    #     lick_frames = range(len(lick_times))

    #     df = pd.DataFrame({"timestamps": lick_times, "frame": lick_frames})
    #     # df = pd.DataFrame({"timestamps": lick_times})
    #     return cls(licks=df)

    def get_indexes(start_time, end_time, times):
        start_index = None
        end_index = None

        index = 0
        while start_index is None or end_index is None:
            if index == len(times):
                if end_time is None:
                    end_time = len(times) - 1
                if start_index is None:
                    start_index = len(times) - 1

                break

            if start_index is None:
                if start_time >= times[index]:
                    start_index = index
            elif end_index is None and end_time < times[index]:
                end_index = index - 1
                break

            index+=1

        return start_index, end_index

    def get_stim_starts_and_ends(
        sync_dataset: SyncFile, fallback_line: Union[int, str] = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get stimulus presentation start and end times from a loaded session
        *.sync datset.

        Parameters
        ----------
        sync_dataset : Dataset
            A loaded *.sync file for a session (contains events from
            different data streams logged on a global time basis)
        fallback_line : Union[int, str], optional
            The sync dataset line label to use if named line labels could not
            be found, by default 5.

            For more details about line labels see:
            https://alleninstitute.sharepoint.com/:x:/s/Instrumentation/ES2bi1xJ3E9NupX-zQeXTlYBS2mVVySycfbCQhsD_jPMUw?e=Z9jCwH  # noqa: E501

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple of numpy arrays containing
            (stimulus_start_times, stimulus_end_times) in seconds.
        """

        # Look for 'stim_running' line in sync dataset line labels
        stim_line: Union[int, str] = fallback_line
        for line in sync_dataset.line_labels:
            if line == 'stim_running':
                stim_line = line
                break
            if line == 'sweep':
                stim_line = line
                break

        if stim_line == fallback_line:
            logging.warning(
                f"Could not find 'stim_running' nor 'sweep' line labels in "
                f"sync dataset ({sync_dataset.dfile.filename}). Defaulting to "
                f"using fallback line label index ({fallback_line}) which "
                f"is not guaranteed to be correct!"
            )

        # 'stim_running'/'sweep' line is high while visual stimulus is being
        # displayed and low otherwise
        stim_starts = sync_dataset.get_rising_edges(stim_line, units='seconds')
        stim_ends = sync_dataset.get_falling_edges(stim_line, units='seconds')

        return stim_starts, stim_ends

    @classmethod
    def from_stimulus_file(cls, sync_file: SyncFile, stimulus_file: EcephysStimulusFile,
                           stimulus_timestamps: StimulusTimestamps) -> "Licks":
        """Get lick data from pkl file.
        This function assumes that the first sensor in the list of
        lick_sensors is the desired lick sensor.

        Since licks can occur outside of a trial context, the lick times
        are extracted from the vsyncs and the frame number in `lick_events`.
        Since we don't have a timestamp for when in "experiment time" the
        vsync stream starts (from self.get_stimulus_timestamps), we compute
        it by fitting a linear regression (frame number x time) for the
        `start_trial` and `end_trial` events in the `trial_log`, to true
        up these time streams.

        :returns: pd.DataFrame
            Two columns: "time", which contains the sync time
            of the licks that occurred in this session and "frame",
            the frame numbers of licks that occurred in this session
        """
        data = stimulus_file.data

        sync_data = SyncFile.load_data(sync_file.filepath)

        sync_dataset = SyncDataset(sync_file.filepath)

        lick_times = sync_data['lick_times']

        # stim_starts, stim_ends = cls.get_stim_starts_and_ends(sync_dataset)

        # behavior_start = stim_starts[0]
        # behavior_end = stim_ends[0]

        # start_index, end_index = cls.get_indexes(behavior_start, behavior_end, lick_times)

        

        # epoch_frames = np.where((lick_times > behavior_start) & (lick_times < behavior_end))[0]

        # print('epoch_frames', epoch_frames)



        lick_frames = (data["items"]["behavior"]["lick_sensors"][0]
                       ["lick_events"])

        # print('lick_times', len(lick_times))
        # print('lick_frames', len(lick_frames))
        # print('stimulus_timestamps', len(stimulus_timestamps.value))

        max_length = min(len(lick_times), len(lick_frames))

        lick_frames = lick_frames[0:max_length]
        lick_times = lick_times[0:max_length]
        # print('len', start_index,  - end_index)

        # there's an occasional bug where the number of logged
        # frames is one greater than the number of vsync intervals.
        # If the animal licked on this last frame it will cause an
        # error here. This fixes the problem.
        # see: https://github.com/AllenInstitute/visual_behavior_analysis
        # /issues/572  # noqa: E501
        #    & https://github.com/AllenInstitute/visual_behavior_analysis
        #    /issues/379  # noqa:E501
        #
        # This bugfix copied from
        # https://github.com/AllenInstitute/visual_behavior_analysis/blob
        # /master/visual_behavior/translator/foraging2/extract.py#L640-L647

        # if len(lick_frames) > 0:
        #     if lick_frames[-1] == len(lick_times.value):
        #         lick_frames = lick_frames[:-1]
        #         cls._logger.error('removed last lick - '
        #                           'it fell outside of stimulus_timestamps '
        #                           'range')

        # lick_times = \
        #     [stimulus_timestamps.value[frame] for frame in lick_frames]

        

        df = pd.DataFrame({"timestamps": lick_times, "frame": lick_frames})
        return cls(licks=df)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> Optional["Licks"]:
        if 'licking' in nwbfile.processing:
            lick_module = nwbfile.processing['licking']
            licks = lick_module.get_data_interface('licks')
            timestamps = licks.timestamps[:]
            frame = licks.data[:]
        else:
            timestamps = []
            frame = []

        df = pd.DataFrame({
            'timestamps': timestamps,
            'frame': frame
        })

        return cls(licks=df)

    def to_nwb(self, nwbfile: NWBFile) -> NWBFile:

        # If there is no lick data, do not write
        # anything to the NWB file (this is
        # expected for passive sessions)
        if len(self.value['frame']) == 0:
            return nwbfile

        lick_timeseries = TimeSeries(
            name='licks',
            data=self.value['frame'].values,
            timestamps=self.value['timestamps'].values,
            description=('Timestamps and stimulus presentation '
                         'frame indices for lick events'),
            unit='N/A'
        )

        # if len(self.value['timestamps']) == 0:
        #     return nwbfile

        # lick_timeseries = TimeSeries(
        #     name='licks',
        #     data=self.value['frame'].values,
        #     timestamps=self.value['timestamps'].values,
        #     description=('Timestamps and stimulus presentation'),
        #     unit='N/A'
        # )

        # Add lick interface to nwb file, by way of a processing module:
        licks_mod = ProcessingModule('licking',
                                     'Licking behavior processing module')
        licks_mod.add_data_interface(lick_timeseries)
        nwbfile.add_processing_module(licks_mod)

        return nwbfile

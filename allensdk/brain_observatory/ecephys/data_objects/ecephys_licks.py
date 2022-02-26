import logging
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

from allensdk.brain_observatory.behavior.data_files import SyncFile


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


    @classmethod
    def from_sync_file(cls, sync_file: SyncFile)  -> "Licks":
        sync_data = SyncFile.load_data(sync_file.filepath)
        lick_times = sync_data['lick_times']

        #TODO - create a new class for this
        # vsyncs = sync_data['ophys_frames']
        # vsyncs_lookup = {}

        # frame = 0
        # for vsync in vsyncs:
        #     vsyncs_lookup[vsync] = frame
        #     frame+=1

        # print('vsyncs' , vsyncs)

        # lick_frames = []
        # for lick_time in lick_times:
        #     lick_frames.append(vsyncs_lookup[lick_time])

        #TODO update this
        lick_frames = range(len(lick_times))

        df = pd.DataFrame({"timestamps": lick_times, "frame": lick_frames})
        # df = pd.DataFrame({"timestamps": lick_times})
        return cls(licks=df)

    @classmethod
    def from_stimulus_file(cls, stimulus_file: StimulusFile,
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

        lick_frames = (data["items"]["behavior"]["lick_sensors"][0]
                       ["lick_events"])

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

        if len(lick_frames) > 0:
            if lick_frames[-1] == len(stimulus_timestamps.value):
                lick_frames = lick_frames[:-1]
                cls._logger.error('removed last lick - '
                                  'it fell outside of stimulus_timestamps '
                                  'range')

        lick_times = \
            [stimulus_timestamps.value[frame] for frame in lick_frames]
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

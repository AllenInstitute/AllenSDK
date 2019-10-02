from itertools import product
import functools
from collections import defaultdict
import logging
import warnings

import numpy as np

from allensdk.brain_observatory.sync_dataset import Dataset
from allensdk.brain_observatory.ecephys import stimulus_sync
from allensdk.brain_observatory import sync_utilities


class EcephysSyncDataset(Dataset):
        
    @property
    def sample_frequency(self):
        return self.meta_data['ni_daq']['counter_output_freq']


    @sample_frequency.setter
    def sample_frequency(self, value):
        if not hasattr(self, 'meta_data'):
            self.meta_data = defaultdict(dict)
        self.meta_data['ni_daq']['counter_output_freq'] = value


    def __init__(self):
        '''In-memory representation of a sync h5 file as produced by the sync package. 

        Notes
        -----
        base is from here: http://aibspi/mpe_apps/sync/blob/master/sync/dataset.py
        Construction works slightly differently for this class as its base. In particular, 
        this class' __init__ method merely constructs the object. To make a new SyncDataset in client code, use the 
        factory classmethod. This is done for ease of testability.

        '''
        pass


    def extract_led_times(self, keys=Dataset.OPTOGENETIC_STIMULATION_KEYS, fallback_line=18):

        try:
            led_times = self.get_edges(
                kind="rising",
                keys=keys,
                units="seconds"
            )
        except KeyError:
            warnings.warn(f"unable to find LED times using line labels {keys}, returning line {fallback_line}")
            led_times = self.get_rising_edges(fallback_line, units="seconds")

        return led_times


    def extract_frame_times_from_photodiode(self, photodiode_cycle=60, frame_keys=Dataset.FRAME_KEYS, photodiode_keys=Dataset.PHOTODIODE_KEYS):
        photodiode_times = self.get_edges('all', photodiode_keys)
        vsync_times = self.get_edges('falling', frame_keys)
        vsync_times = sync_utilities.trim_discontiguous_times(vsync_times)
        
        logging.info(f"Total vsyncs: {len(vsync_times)}")

        photodiode_times = stimulus_sync.trim_border_pulses(photodiode_times, vsync_times)
        photodiode_times = stimulus_sync.correct_on_off_effects(photodiode_times)
        photodiode_times = stimulus_sync.fix_unexpected_edges(photodiode_times, cycle=photodiode_cycle)

        frame_duration = stimulus_sync.estimate_frame_duration(photodiode_times, cycle=photodiode_cycle)
        irregular_interval_policy = functools.partial(stimulus_sync.allocate_by_vsync, np.diff(vsync_times))
        frame_indices, frame_start_times, frame_end_times = stimulus_sync.compute_frame_times(
            photodiode_times, frame_duration, len(vsync_times), 
            cycle=photodiode_cycle, irregular_interval_policy=irregular_interval_policy
        )

        return frame_start_times


    def extract_frame_times_from_vsyncs(self, photodiode_cycle=60, 
        frame_keys=Dataset.FRAME_KEYS, photodiode_keys=Dataset.PHOTODIODE_KEYS
    ):
        raise NotImplementedError()


    def extract_frame_times(self, strategy, photodiode_cycle=60, 
        frame_keys=Dataset.FRAME_KEYS, photodiode_keys=Dataset.PHOTODIODE_KEYS
    ):

        if strategy == 'use_photodiode':
            return self.extract_frame_times_from_photodiode(
                photodiode_cycle=photodiode_cycle, frame_keys=frame_keys, photodiode_keys=photodiode_keys
                )
        elif strategy == 'use_vsyncs':
            return self.extract_frame_times_from_vsyncs(
                photodiode_cycle=photodiode_cycle, frame_keys=frame_keys, photodiode_keys=photodiode_keys
                )
        else:
            raise ValueError('unrecognized strategy: {}'.format(strategy))


    @classmethod
    def factory(cls, path):
        ''' Build a new SyncDataset.

        Parameters
        ----------
        path : str
            Filesystem path to the h5 file containing sync information to be loaded.

        '''

        obj = cls()
        obj.load(path)
        return obj


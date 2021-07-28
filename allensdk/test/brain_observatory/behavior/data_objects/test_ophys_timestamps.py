from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pynwb
import pytest

from allensdk.brain_observatory.behavior.data_files import StimulusFile, \
    SyncFile
from allensdk.brain_observatory.behavior.data_objects import StimulusTimestamps
from allensdk.brain_observatory.behavior.data_objects.licks import Licks
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.equipment import \
    Equipment
from allensdk.brain_observatory.behavior.data_objects.rewards import Rewards
from allensdk.brain_observatory.behavior.data_objects.timestamps \
    .ophys_timestamps import \
    OphysTimestamps, OphysTimestampsMultiplane
from allensdk.brain_observatory.behavior.data_objects.trials.trial_table \
    import TrialTable
from allensdk.internal.brain_observatory.time_sync import OphysTimeAligner
from allensdk.test.brain_observatory.behavior.data_objects.lims_util import \
    LimsTest


class TestFromSyncFile(LimsTest):
    @classmethod
    def setup_class(cls):
        dir = Path(__file__).parent.resolve()
        test_data_dir = dir / 'test_data'

        cls.sync_file = SyncFile(filepath=str(test_data_dir / 'sync.h5'))

    def test_from_sync_file(self):
        self.sync_file._data = {'ophys_frames': np.array([.1, .2, .3])}
        ts = OphysTimestamps.from_sync_file(sync_file=self.sync_file,
                                            number_of_frames=3)
        expected = np.array([.1, .2, .3])
        np.testing.assert_equal(ts.value, expected)

    def test_too_long(self):
        """test when timestamps longer than # frames"""
        self.sync_file._data = {'ophys_frames': np.array([.1, .2, .3])}
        ts = OphysTimestamps.from_sync_file(sync_file=self.sync_file,
                                            number_of_frames=2)
        expected = np.array([.1, .2])
        np.testing.assert_equal(ts.value, expected)

    def test_too_short(self):
        """test when timestamps shorter than # frames"""
        self.sync_file._data = {'ophys_frames': np.array([.1, .2, .3])}
        with pytest.raises(RuntimeError):
            OphysTimestamps.from_sync_file(sync_file=self.sync_file,
                                                number_of_frames=4)

    def test_multiplane(self):
        """test timestamps properly extracted when multiplane"""
        self.sync_file._data = {'ophys_frames': np.array([.1, .2, .3, .4])}
        ts = OphysTimestampsMultiplane.from_sync_file(sync_file=self.sync_file,
                                                      group_count=2,
                                                      plane_group=0,
                                                      number_of_frames=2)
        expected = np.array([.1, .3])
        np.testing.assert_equal(ts.value, expected)

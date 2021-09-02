from pathlib import Path

import numpy as np
import pytest

from allensdk.brain_observatory.behavior.data_files import SyncFile
from allensdk.brain_observatory.behavior.data_objects.timestamps \
    .ophys_timestamps import \
    OphysTimestamps, OphysTimestampsMultiplane
from allensdk.test.brain_observatory.behavior.data_objects.lims_util import \
    LimsTest


class TestFromSyncFile(LimsTest):
    def setup_method(self, method):
        dir = Path(__file__).parent.resolve()
        test_data_dir = dir / 'test_data'

        self.sync_file = SyncFile(filepath=str(test_data_dir / 'sync.h5'))

    def test_from_sync_file(self):
        self.sync_file._data = {'ophys_frames': np.array([.1, .2, .3])}
        ts = OphysTimestamps.from_sync_file(sync_file=self.sync_file)\
            .validate(number_of_frames=3)
        expected = np.array([.1, .2, .3])
        np.testing.assert_equal(ts.value, expected)

    def test_too_long_single_plane(self):
        """test that timestamps are truncated for single plane data"""
        self.sync_file._data = {'ophys_frames': np.array([.1, .2, .3])}
        ts = OphysTimestamps.from_sync_file(sync_file=self.sync_file)\
            .validate(number_of_frames=2)
        expected = np.array([.1, .2])
        np.testing.assert_equal(ts.value, expected)

    def test_too_long_multi_plane(self):
        """test that exception raised when timestamps longer than # frames
        for multiplane data"""
        self.sync_file._data = {'ophys_frames': np.array([.1, .2, .3])}
        with pytest.raises(RuntimeError):
            OphysTimestampsMultiplane.from_sync_file(sync_file=self.sync_file,
                                                     group_count=2,
                                                     plane_group=0)\
                .validate(number_of_frames=1)

    def test_too_short(self):
        """test when timestamps shorter than # frames"""
        self.sync_file._data = {'ophys_frames': np.array([.1, .2, .3])}
        with pytest.raises(RuntimeError):
            OphysTimestamps.from_sync_file(sync_file=self.sync_file)\
                .validate(number_of_frames=4)

    def test_multiplane(self):
        """test timestamps properly extracted when multiplane"""
        self.sync_file._data = {'ophys_frames': np.array([.1, .2, .3, .4])}
        ts = OphysTimestampsMultiplane.from_sync_file(sync_file=self.sync_file,
                                                      group_count=2,
                                                      plane_group=0)\
            .validate(number_of_frames=2)
        expected = np.array([.1, .3])
        np.testing.assert_equal(ts.value, expected)

    @pytest.mark.parametrize(
        "timestamps,plane_group,group_count,expected",
        [
            (np.ones(10), 1, 0, np.ones(10)),
            (np.ones(10), 1, 0, np.ones(10)),
            # middle
            (np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]), 1, 3, np.ones(4)),
            # first
            (np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]), 0, 4, np.ones(3)),
            # last
            (np.array([0, 1, 0, 1, 0, 1, 0, 1]), 1, 2, np.ones(4)),
            # only one group
            (np.ones(10), 0, 1, np.ones(10))
        ]
    )
    def test_process_ophys_plane_timestamps(
            self, timestamps, plane_group, group_count, expected):
        """Various test cases"""
        self.sync_file._data = {'ophys_frames': timestamps}
        number_of_frames = len(timestamps) if group_count == 0 else \
            len(timestamps) / group_count
        if group_count == 0:
            ts = OphysTimestamps.from_sync_file(sync_file=self.sync_file)
        else:
            ts = OphysTimestampsMultiplane.from_sync_file(
                sync_file=self.sync_file, group_count=group_count,
                plane_group=plane_group)
        ts = ts.validate(number_of_frames=number_of_frames)
        np.testing.assert_array_equal(expected, ts.value)

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pynwb
import pytest

from allensdk.brain_observatory.behavior.data_files import \
    SyncFile
from allensdk.brain_observatory.behavior.data_files.eye_tracking_file import \
    EyeTrackingFile
from allensdk.brain_observatory.behavior.data_objects.eye_tracking \
    .eye_tracking_table import \
    EyeTrackingTable
from allensdk.test.brain_observatory.behavior.data_objects.lims_util import \
    LimsTest
from allensdk.test.brain_observatory.behavior.test_eye_tracking_processing \
    import \
    create_refined_eye_tracking_df


class TestFromDataFile(LimsTest):
    @classmethod
    def setup_class(cls):
        cls.ophys_experiment_id = 994278291

        dir = Path(__file__).parent.parent.resolve()
        test_data_dir = dir / 'test_data'

        df = pd.read_pickle(str(test_data_dir / 'eye_tracking_table.pkl'))
        cls.expected = EyeTrackingTable(eye_tracking=df)

    @pytest.mark.requires_bamboo
    def test_from_data_file(self):
        etf = EyeTrackingFile.from_lims(
            ophys_experiment_id=self.ophys_experiment_id, db=self.dbconn)
        sync_file = SyncFile.from_lims(
            ophys_experiment_id=self.ophys_experiment_id, db=self.dbconn)
        ett = EyeTrackingTable.from_data_file(data_file=etf,
                                              sync_file=sync_file)

        # filter to first 100 values for testing
        ett = EyeTrackingTable(eye_tracking=ett.value.iloc[:100])
        assert ett == self.expected


class TestNWB:
    @classmethod
    def setup_class(cls):
        dir = Path(__file__).parent.parent.resolve()
        cls.test_data_dir = dir / 'test_data'

        df = create_refined_eye_tracking_df(
            np.array([[0.1, 12 * np.pi, 72 * np.pi, 196 * np.pi, False,
                       196 * np.pi, 12 * np.pi, 72 * np.pi,
                       1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.,
                       13., 14., 15.],
                      [0.2, 20 * np.pi, 90 * np.pi, 225 * np.pi, False,
                       225 * np.pi, 20 * np.pi, 90 * np.pi,
                       2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.,
                       14., 15., 16.]])
        )
        cls.eye_tracking_table = EyeTrackingTable(eye_tracking=df)

    def setup_method(self, method):
        self.nwbfile = pynwb.NWBFile(
            session_description='asession',
            identifier='1234',
            session_start_time=datetime.now()
        )

    @pytest.mark.parametrize('roundtrip', [True, False])
    def test_read_write_nwb(self, roundtrip,
                            data_object_roundtrip_fixture):
        self.eye_tracking_table.to_nwb(nwbfile=self.nwbfile)

        if roundtrip:
            obt = data_object_roundtrip_fixture(
                nwbfile=self.nwbfile,
                data_object_cls=EyeTrackingTable)
        else:
            obt = EyeTrackingTable.from_nwb(nwbfile=self.nwbfile)

        assert obt == self.eye_tracking_table

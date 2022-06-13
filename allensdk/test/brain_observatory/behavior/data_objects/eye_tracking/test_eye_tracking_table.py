import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pynwb
import pytest
from allensdk.brain_observatory.behavior.data_files\
    .eye_tracking_metadata_file import \
    EyeTrackingMetadataFile

from allensdk.brain_observatory.sync_dataset import Dataset as SyncDataset
from allensdk.brain_observatory.behavior.data_objects import StimulusTimestamps
from allensdk.brain_observatory import sync_utilities

from allensdk.brain_observatory.behavior.data_files import \
    SyncFile
from allensdk.brain_observatory.behavior.data_files.eye_tracking_file import \
    EyeTrackingFile
from allensdk.brain_observatory.behavior.data_objects import BehaviorSessionId
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
        behavior_session_id = BehaviorSessionId.from_lims(
            db=self.dbconn, ophys_experiment_id=self.ophys_experiment_id)
        etf = EyeTrackingFile.from_lims(
            behavior_session_id=behavior_session_id.value, db=self.dbconn)
        sync_file = SyncFile.from_lims(
            behavior_session_id=behavior_session_id.value, db=self.dbconn)

        sync_path = Path(sync_file.filepath)

        frame_times = sync_utilities.get_synchronized_frame_times(
            session_sync_file=sync_path,
            sync_line_label_keys=SyncDataset.EYE_TRACKING_KEYS,
            drop_frames=None,
            trim_after_spike=False)

        stimulus_timestamps = StimulusTimestamps(
                timestamps=frame_times,
                monitor_delay=0.0)

        ett = EyeTrackingTable.from_data_file(
                    data_file=etf,
                    stimulus_timestamps=stimulus_timestamps)

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


class TestTimeFrameAlignment:
    @classmethod
    def setup_class(cls):
        with open('/allen/aibs/informatics/module_test_data/ecephys/'
                  'BEHAVIOR_ECEPHYS_WRITE_NWB_QUEUE_1044594870_input.json') \
                as f:
            input_data = json.load(f)

        cls.input_data = input_data['session_data']

        cls.eye_tracking_file = EyeTrackingFile.from_json(
            dict_repr=cls.input_data)

        cls.metadata_file = EyeTrackingMetadataFile.from_json(
            dict_repr=cls.input_data)
        # Making up timestamps
        cls.stimulus_timestamps = StimulusTimestamps(
            timestamps=(
                np.linspace(1.33955, 9.72219322e+03,
                            cls.eye_tracking_file.data.shape[0])),
            monitor_delay=0)

    @pytest.mark.requires_bamboo
    def test_metadata_frame_is_dropped(self):
        """Tests that when an eye tracking movie produced by MVR
        (adds extra metadata frame at front), that this extra frame is
        dropped"""
        # Make it 1 shorter than # frames
        timestamps = self.stimulus_timestamps.update_timestamps(
            timestamps=self.stimulus_timestamps.value[:-1])

        ett = EyeTrackingTable.from_data_file(
            data_file=self.eye_tracking_file,
            stimulus_timestamps=timestamps,
            metadata_file=self.metadata_file
        )
        assert (ett.value.shape[0] ==
                # Subtract 1 for the metadata frame
                self.eye_tracking_file.data.shape[0] - 1)

    @pytest.mark.requires_bamboo
    def test_timestamps_are_truncated(self):
        """Tests that when the sync file contains more timestamps than frames,
        that the timestamps are truncated"""
        # Make it 2 longer than # frames
        timestamps = self.stimulus_timestamps.update_timestamps(
            timestamps=np.concatenate([self.stimulus_timestamps.value,
                                       self.stimulus_timestamps.value[-2:]]))

        ett = EyeTrackingTable.from_data_file(
            data_file=self.eye_tracking_file,
            stimulus_timestamps=timestamps,
            metadata_file=self.metadata_file
        )

        assert (ett.value.shape[0] ==
                # subtract off metadata frame
               self.eye_tracking_file.data.shape[0] - 1)

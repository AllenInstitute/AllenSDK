import pickle
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pynwb
import pytest

from allensdk.brain_observatory.behavior.data_files import (
    BehaviorStimulusFile, SyncFile)
from allensdk.brain_observatory.behavior.data_objects import StimulusTimestamps
from allensdk.brain_observatory.behavior.data_objects.licks import Licks


class TestFromBehaviorStimulusFile:
    @classmethod
    def setup_class(cls):
        dir = Path(__file__).parent.resolve()
        test_data_dir = dir / 'test_data'

        cls.stimulus_file = BehaviorStimulusFile(
            filepath=test_data_dir / 'behavior_stimulus_file.pkl')
        cls.sync_file = SyncFile(
            filepath=test_data_dir / 'sync.h5')
        expected = pd.read_pickle(str(test_data_dir / 'licks.pkl'))
        cls.expected = Licks(licks=expected)

    def test_monitor_delay_error(self):
        """
        Test that an error is raised if Licks are instantiated with
        non-zero monitor delay
        """
        timestamps = StimulusTimestamps(
                        np.arange(10),
                        0.1)
        with pytest.raises(RuntimeError,
                           match="monitor_delay should be zero"):
            Licks.from_stimulus_file(
                     stimulus_file=self.stimulus_file,
                     stimulus_timestamps=timestamps)

    def test_from_stimulus_file(self):
        st = StimulusTimestamps.from_stimulus_file(
            stimulus_file=self.stimulus_file,
            monitor_delay=0.0)
        licks = Licks.from_stimulus_file(stimulus_file=self.stimulus_file,
                                         stimulus_timestamps=st)
        assert licks == self.expected

    def test_from_stimulus_and_sync_file(self):
        """Test that the expected data is loaded from the sync and stim file.
        Test is slightly different from other tests as the sync file data is
        not matched up to the stim data.
        """
        lick_times = self.sync_file.data['lick_times']
        licks = Licks.from_stimulus_file(stimulus_file=self.stimulus_file,
                                         stimulus_timestamps=lick_times)
        assert licks.value['timestamps'][0] == lick_times[0]
        assert licks.value['frame'][0] == self.expected.value['frame'][0]

    def test_from_stimulus_file2(self, tmpdir):
        """
        Test that Licks.from_stimulus_file returns a dataframe
        of licks whose timestamps are based on their frame number
        with respect to the stimulus_timestamps
        """
        stimulus_filepath = self._create_test_stimulus_file(
            lick_events=[12, 15, 90, 136], tmpdir=tmpdir)
        stimulus_file = BehaviorStimulusFile.from_json(
            dict_repr={'behavior_stimulus_file': str(stimulus_filepath)})
        timestamps = StimulusTimestamps(timestamps=np.arange(0, 2.0, 0.01),
                                        monitor_delay=0.0)
        licks = Licks.from_stimulus_file(stimulus_file=stimulus_file,
                                         stimulus_timestamps=timestamps)

        expected_dict = {'timestamps': [0.12, 0.15, 0.90, 1.36],
                         'frame': [12, 15, 90, 136]}
        expected_df = pd.DataFrame(expected_dict)
        assert expected_df.columns.equals(licks.value.columns)
        np.testing.assert_array_almost_equal(
            expected_df.timestamps.to_numpy(),
            licks.value['timestamps'].to_numpy(),
            decimal=10)
        np.testing.assert_array_almost_equal(expected_df.frame.to_numpy(),
                                             licks.value['frame'].to_numpy(),
                                             decimal=10)

    def test_empty_licks(self, tmpdir):
        """
        Test that Licks.from_stimulus_file in the case where
        there are no licks
        """

        stimulus_filepath = self._create_test_stimulus_file(
            lick_events=[], tmpdir=tmpdir)
        stimulus_file = BehaviorStimulusFile.from_json(
            dict_repr={'behavior_stimulus_file': str(stimulus_filepath)})
        timestamps = StimulusTimestamps(timestamps=np.arange(0, 2.0, 0.01),
                                        monitor_delay=0.0)
        licks = Licks.from_stimulus_file(stimulus_file=stimulus_file,
                                         stimulus_timestamps=timestamps)

        expected_dict = {'timestamps': [],
                         'frame': []}
        expected_df = pd.DataFrame(expected_dict)
        assert expected_df.columns.equals(licks.value.columns)
        np.testing.assert_array_equal(expected_df.timestamps.to_numpy(),
                                      licks.value['timestamps'].to_numpy())
        np.testing.assert_array_equal(expected_df.frame.to_numpy(),
                                      licks.value['frame'].to_numpy())

    def test_get_licks_excess(self, tmpdir):
        """
        Test that Licks.from_stimulus_file
        in the case where
        there is an extra frame at the end of the trial log and the mouse
        licked on that frame

        https://github.com/AllenInstitute/visual_behavior_analysis/blob
        /master/visual_behavior/translator/foraging2/extract.py#L640-L647
        """
        stimulus_filepath = self._create_test_stimulus_file(
            lick_events=[12, 15, 90, 136, 200],  # len(timestamps) == 200,
            tmpdir=tmpdir)
        stimulus_file = BehaviorStimulusFile.from_json(
            dict_repr={'behavior_stimulus_file': str(stimulus_filepath)})
        timestamps = StimulusTimestamps(timestamps=np.arange(0, 2.0, 0.01),
                                        monitor_delay=0.0)
        licks = Licks.from_stimulus_file(stimulus_file=stimulus_file,
                                         stimulus_timestamps=timestamps)

        expected_dict = {'timestamps': [0.12, 0.15, 0.90, 1.36],
                         'frame': [12, 15, 90, 136]}
        expected_df = pd.DataFrame(expected_dict)
        assert expected_df.columns.equals(licks.value.columns)
        np.testing.assert_array_almost_equal(
            expected_df.timestamps.to_numpy(),
            licks.value['timestamps'].to_numpy(),
            decimal=10)
        np.testing.assert_array_almost_equal(expected_df.frame.to_numpy(),
                                             licks.value['frame'].to_numpy(),
                                             decimal=10)

    def test_get_licks_failure(self, tmpdir):
        stimulus_filepath = self._create_test_stimulus_file(
            lick_events=[12, 15, 90, 136, 201],  # len(timestamps) == 200,
            tmpdir=tmpdir)
        stimulus_file = BehaviorStimulusFile.from_json(
            dict_repr={'behavior_stimulus_file': str(stimulus_filepath)})
        timestamps = StimulusTimestamps(timestamps=np.arange(0, 2.0, 0.01),
                                        monitor_delay=0.0)

        with pytest.raises(IndexError):
            Licks.from_stimulus_file(stimulus_file=stimulus_file,
                                     stimulus_timestamps=timestamps)

    @staticmethod
    def _create_test_stimulus_file(lick_events, tmpdir):
        trial_log = [
            {'licks': [(-1.0, 100), (-1.0, 200)]},
            {'licks': [(-1.0, 300), (-1.0, 400)]},
            {'licks': [(-1.0, 500), (-1.0, 600)]}
        ]

        lick_events = [{'lick_events': lick_events}]

        data = {
            'items': {
                'behavior': {
                    'trial_log': trial_log,
                    'lick_sensors': lick_events
                }
            },
        }
        tmp_path = tmpdir / 'stimulus_file.pkl'
        with open(tmp_path, 'wb') as f:
            pickle.dump(data, f)
            f.seek(0)

        return tmp_path


class TestNWB:
    @classmethod
    def setup_class(cls):
        dir = Path(__file__).parent.resolve()
        test_data_dir = dir / 'test_data'

        stimulus_file = BehaviorStimulusFile(
            filepath=test_data_dir / 'behavior_stimulus_file.pkl')
        ts = StimulusTimestamps.from_stimulus_file(
            stimulus_file=stimulus_file,
            monitor_delay=0.0)
        cls.licks = Licks.from_stimulus_file(stimulus_file=stimulus_file,
                                             stimulus_timestamps=ts)

    def setup_method(self, method):
        self.nwbfile = pynwb.NWBFile(
            session_description='asession',
            identifier='1234',
            session_start_time=datetime.now()
        )

    @pytest.mark.parametrize('roundtrip', [True, False])
    def test_read_write_nwb(self, roundtrip,
                            data_object_roundtrip_fixture):
        self.licks.to_nwb(nwbfile=self.nwbfile)

        if roundtrip:
            obt = data_object_roundtrip_fixture(
                nwbfile=self.nwbfile,
                data_object_cls=Licks)
        else:
            obt = self.licks.from_nwb(nwbfile=self.nwbfile)

        assert obt == self.licks
